import tensorboard
from configs.utils.config import load_config


from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from tqdm import tqdm


import torch
import os

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import warnings
import argparse
# import model
from models.AROccFlowNet.occupancy_flow_model import AROccFlowNet
# import loss and metrics
from evaluation.losses.occupancy_flow_map_loss import OccupancyFlowMapLoss
from evaluation.losses.trajectory_loss import TrajectoryLoss
from evaluation.metrics.occupancy_flow_map_metrics import OccupancyFlowMapMetrics

from utils.training_utils import load_checkpoint, save_checkpoint
from datasets.I24Motion.utils.dataset_utils import get_dataloader, get_dataloader_ddp, 
from datasets.I24Motion.utils.training_utils import parse_data, parse_outputs
from evaluation.metrics.trajectory_metrics import TrajectoryMetrics

warnings.filterwarnings("ignore")




def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)



def setup(config, gpu_id, ddp=True):
    """
    """
    model_config = config.models
    model = AROccFlowNet(model_config).to(gpu_id)
    if ddp:
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    
    
    optimizer = torch.optim.NAdam(params=model.parameters(), 
                                  lr=config.training_settings.optimizer.learning_rate,
                                #   weight_decay=config.training_settings.optimizer.weight_decay
                                ) 
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                step_size=config.training_settings.scheduler.step_size,
                                                gamma=config.training_settings.scheduler.gamma) 
    # get the losses Config
    config_losses = config.losses
    occupancy_flow_map_loss_config = config_losses.occupancy_flow_map_loss
    trajectory_loss_config = config_losses.trajectory_loss

    occupancy_flow_map_loss = OccupancyFlowMapLoss(device=gpu_id, config=occupancy_flow_map_loss_config)
    trajectory_loss = TrajectoryLoss(device=gpu_id, config=trajectory_loss_config)

    return model, optimizer, scheduler, occupancy_flow_map_loss, trajectory_loss







def model_training(gpu_id, world_size, config, ddp=True):
    proj_name = config.proj_name
    exp_dir = config.exp_dir
    proj_exp_dir = os.path.join(exp_dir, proj_name)
    dataloaders_config = config.dataloaders
    os.path.exists(proj_exp_dir) or os.makedirs(proj_exp_dir)
    if ddp:
        ddp_setup(gpu_id, world_size)
        train_dataloader, val_dataloader, _ = get_dataloader_ddp(dataloaders_config)
    else:
        train_dataloader, val_dataloader, _ = get_dataloader(dataloaders_config)

    logger_config = config.loggers
    tensorboard_config = logger_config.tensorboard
    logger = SummaryWriter(log_dir=tensorboard_config.log_dir)
    model, optimizer, scheduler, occupancy_flow_map_loss, trajectory_loss = setup(config, gpu_id)
    
    
    continue_ep, global_step = load_checkpoint(model, optimizer, scheduler, proj_exp_dir, gpu_id)
    
    
    
    for epoch in range(config.training_settings.epochs):
        if epoch<continue_ep:
            if gpu_id == 0:
                print("\nskip epoch {}/{}".format(epoch+1, config.training_settings.epochs))
            continue
        
        model.train()
        
        
        train_dataloader.sampler.set_epoch(epoch)
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, data in loop:
            
            input_dict, ground_truth_dict = parse_data(data, gpu_id, config)
            
            # get the input
            his_occupancy_map = input_dict['his/observed_occupancy_map']
            his_flow_map = input_dict['his/flow_map']
            his_observed_agent_features = input_dict['his/observed_agent_features']
            flow_origin_occupancy = input_dict['flow_origin_occupancy_map']
            his_valid_mask = input_dict['his/valid_mask']
            agent_types = input_dict['agent_types']

            # get the ground truth
            gt_occluded_occupancy_logits = ground_truth_dict['pred/occluded_occupancy_map']
            gt_observed_occupancy_logits = ground_truth_dict['pred/observed_occupancy_map']
            gt_flow = ground_truth_dict['pred/flow_map']
            gt_trajectories = ground_truth_dict['pred/trajectories']
            gt_valid_mask = ground_truth_dict['pred/valid_mask']
            
            

            outputs = model(his_occupancy_map, his_flow_map, his_observed_agent_features, his_valid_mask, agent_types)

            pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits, predicted_trajectories, predicted_trajectories_score = parse_outputs(outputs, xxx)
    
            
            occupancy_flow_map_loss_dict = occupancy_flow_map_loss.compute(pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits, gt_observed_occupancy_logits, gt_occluded_occupancy_logits, gt_flow, flow_origin_occupancy)
            trajectory_loss_dict = trajectory_loss.compute(predicted_trajectories, predicted_trajectories_score, gt_trajectories, gt_valid_mask)
            # TODO: merge the occupancy_flow_map_loss and trajectory_loss
            loss_value = torch.sum(sum(occupancy_flow_map_loss_dict.values()))

            # backward pass
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            occupancy_flow_map_loss.update(occupancy_flow_map_loss_dict)
            trajectory_loss_dict.update(trajectory_loss_dict)

            global_step += 1
            occupancy_flow_map_loss_res_dict = occupancy_flow_map_loss.get_result()
            trajectory_loss_res_dict = trajectory_loss.get_result()

            if gpu_id == 0:
                if batch_idx % 20 == 0:
                    logger.add_scalars(main_tag="occupancy_flow_map_loss", tag_scalar_dict=occupancy_flow_map_loss_res_dict, global_step=global_step)  
                    logger.add_scalars(main_tag="trajectory_loss", tag_scalar_dict=trajectory_loss_res_dict, global_step=global_step)
        scheduler.step()
        
        ## validate
        
        occupancy_flow_map_loss.reset()
        trajectory_loss.reset()
        valid_metrics = OccupancyFlowMapMetrics(gpu_id, no_warp=False)
        # TODO: Add the trajectory metrics

        model.eval()
        
        with torch.no_grad():
            loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
            for batch_idx, data in loop:
                
                input_dict, ground_truth_dict = parse_data(data, gpu_id, config)

                # get the input
                his_occupancy_map = input_dict['his/observed_occupancy_map']
                his_flow_map = input_dict['his/flow_map']
                his_observed_agent_features = input_dict['his/observed_agent_features']
                flow_origin_occupancy = input_dict['flow_origin_occupancy_map']
                his_valid_mask = input_dict['his/valid_mask']
                agent_types = input_dict['agent_types']

                # get the ground truth
                gt_occluded_occupancy_logits = ground_truth_dict['pred/occluded_occupancy_map']
                gt_observed_occupancy_logits = ground_truth_dict['pred/observed_occupancy_map']
                gt_flow = ground_truth_dict['pred/flow_map']
                gt_trajectories = ground_truth_dict['pred/trajectories']
                gt_valid_mask = ground_truth_dict['pred/valid_mask']

                outputs = model(his_occupancy_map, his_flow_map, his_observed_agent_features, his_valid_mask, agent_types)
                pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits, predicted_trajectories, predicted_trajectories_score = parse_outputs(outputs, config.task_config.num_waypoints)

                
                occupancy_flow_map_loss_dict = occupancy_flow_map_loss.compute(pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits, gt_observed_occupancy_logits, gt_occluded_occupancy_logits, gt_flow, flow_origin_occupancy)
                trajectory_loss_dict = trajectory_loss.compute(predicted_trajectories, predicted_trajectories_score, gt_trajectories, gt_valid_mask)
                # TODO: merge the occupancy_flow_map_loss and trajectory_loss
                loss_value = torch.sum(sum(occupancy_flow_map_loss_dict.values()))
                

                
                pred_observed_occupancy_logits = torch.sigmoid(pred_observed_occupancy_logits)
                pred_occluded_occupancy_logits = torch.sigmoid(pred_occluded_occupancy_logits)
                
                metrics_dict = valid_metrics.compute(pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits, gt_observed_occupancy_logits, gt_occluded_occupancy_logits, gt_flow, flow_origin_occupancy, no_warp=False)

                valid_metrics.update(metrics_dict)
                occupancy_flow_map_loss_dict.update(occupancy_flow_map_loss_dict)
                trajectory_loss_dict.update(trajectory_loss_dict)

            occupancy_flow_map_loss_res_dict = occupancy_flow_map_loss.get_result()
            trajectory_loss_res_dict = trajectory_loss.get_result()
            val_res_dict = valid_metrics.get_result()
            if gpu_id == 0:
                logger.add_scalars(main_tag="val_metrics", tag_scalar_dict=val_res_dict, global_step=global_step)
                logger.add_scalars(main_tag="val_occupancy_flow_map_loss", tag_scalar_dict=occupancy_flow_map_loss_res_dict, global_step=global_step)
                logger.add_scalars(main_tag="val_trajectory_loss", tag_scalar_dict=trajectory_loss_res_dict, global_step=global_step)
            
        if gpu_id == 0:
            if (epoch+1) % config.training_settings.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, config.paths.checkpoints, global_step)
    destroy_process_group()










if __name__ == "__main__":
    # ============= Parse Argument =============
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("--config", type=str, default="configs/model_configs/AROccFlowNetS.py", help="config file")
    args = parser.parse_args()
    # ============= Load Configuration =============
    config = load_config(args.config)
    model_training(0, 1, config, ddp=False)
    # os.environ["NCCL_P2P_DISABLE"] = "1"

    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # world_size = torch.cuda.device_count()
    # mp.spawn(model_training, args=(world_size, config), nprocs=world_size)

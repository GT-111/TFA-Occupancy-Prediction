import os
import torch
import warnings
import argparse
# import model
from models.OFMPNet import OFMPNet
from evaluation.losses.occupancy_flow_map_loss import OccupancyFlowMapLoss
from evaluation.metrics.occupancy_flow_map_metrics import OccupancyFlowMapMetrics
# import training utils
from tqdm import tqdm
from utils.training_utils import load_checkpoint, save_checkpoint
from datasets.I24Motion.utils.dataset_utils import get_dataloader
from datasets.I24Motion.utils.training_utils import parse_data, parse_outputs_OFMPNet
# import distributed training
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datasets.I24Motion.utils.dataset_utils import get_dataloader_ddp
# import config
from configs.utils.config import load_config
# import tensorboard
from torch.utils.tensorboard import SummaryWriter


# warnings.filterwarnings("ignore")

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

def setup(config, gpu_id, enable_ddp=True):
    """
    Setup the model, optimizer, scheduler, and losses.
    """
    # //[ ] The model config need to be updated if the model is changed
    model_config = config.models
    model = OFMPNet(model_config.ofmpnet).to(gpu_id)
    if enable_ddp:
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)

    # Load the optimizer
    optimizer_config = config.optimizer
    def get_optimizer(optimizer_config):
        optimizer_type = optimizer_config.type
        if optimizer_type == "AdamW":
            return torch.optim.AdamW(
                model.parameters(), 
                lr=optimizer_config.learning_rate,            # Base learning rate
                betas=optimizer_config.betas,  # Slightly higher β2 for smoother updates
                eps=optimizer_config.eps,           # Avoids division by zero
                weight_decay=optimizer_config.weight_decay   # Encourages generalization
            )
        if optimizer_type == "NAdam":
            return torch.optim.NAdam(
                params=model.parameters(), 
                lr=optimizer_config.learning_rate,
                weight_decay=optimizer_config.weight_decay
            )
    optimizer = get_optimizer(optimizer_config)
    # Load the scheduler
    scheduler_config = config.scheduler
    def get_scheduler(scheduler_config):
        scheduler_type = scheduler_config.type
        if scheduler_type == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, 
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma
            )
        if scheduler_type == "CosineAnnealingWarmRestarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer, 
                T_0=scheduler_config.T_0,
                T_mult=scheduler_config.T_mult,
                eta_min=scheduler_config.eta_min
            )
    scheduler = get_scheduler(scheduler_config)
    # Get the losses config
    config_losses = config.losses
    occupancy_flow_map_loss_config = config_losses.occupancy_flow_map_loss

    occupancy_flow_map_loss = OccupancyFlowMapLoss(device=gpu_id, config=occupancy_flow_map_loss_config)

    return model, optimizer, scheduler, occupancy_flow_map_loss

def model_training(gpu_id, world_size, config, enable_ddp=True):
    
    project_dir = config.project_dir
    dataloaders_config = config.dataloaders
    os.path.exists(project_dir) or os.makedirs(project_dir)

    train_config = config.train
    checkpoint_dir = train_config.checkpoint_dir
    max_epochs = train_config.max_epochs
    checkpoint_interval = train_config.checkpoint_interval
    checkpoint_total_limit = train_config.checkpoint_total_limit
    log_interval = train_config.log_interval
    torch.autograd.set_detect_anomaly(True)
    
    if enable_ddp:
        ddp_setup(gpu_id, world_size)
        train_dataloader, val_dataloader, _ = get_dataloader_ddp(dataloaders_config)
    else:
        train_dataloader, val_dataloader, _ = get_dataloader(dataloaders_config)

    logger_config = config.loggers
    tensorboard_config = logger_config.tensorboard
    logger = SummaryWriter(log_dir=tensorboard_config.log_dir)

    model, optimizer, scheduler, occupancy_flow_map_loss = setup(config, gpu_id, enable_ddp=enable_ddp)
    continue_ep, global_step = load_checkpoint(model, optimizer, scheduler, checkpoint_dir, gpu_id)
    

    for epoch in range(continue_ep, max_epochs):
        if epoch<continue_ep:
            if gpu_id == 0:
                print("\nskip epoch {}/{}".format(epoch + 1, max_epochs))
            continue
        
        model.train()
        # //NOTE In distributed mode, calling the set_epoch method at the beginning of each epoch before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.
        if enable_ddp:
            train_dataloader.sampler.set_epoch(epoch)

        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, data in loop:
            input_dict, ground_truth_dict = parse_data(data, gpu_id, config)
            # //[ ]Currently, only the current scene is being used
            input_dict = input_dict['cur']
            ground_truth_dict = ground_truth_dict['cur']

            # get the input
            his_occupancy_map = input_dict['his/observed_occupancy_map']
            his_flow_map = input_dict['his/flow_map'][:, : ,:, -1, :]  # only use the last time step for flow map
            his_observed_agent_features = input_dict['his/observed_agent_features']
            flow_origin_occupancy = input_dict['flow_origin_occupancy_map']
            his_valid_mask = input_dict['his/valid_mask']
            agent_types = input_dict['agent_types']

            # get the ground truth
            gt_occluded_occupancy_logits = ground_truth_dict['pred/occluded_occupancy_map']
            gt_observed_occupancy_logits = ground_truth_dict['pred/observed_occupancy_map']
            gt_flow = ground_truth_dict['pred/flow_map']
            gt_valid_mask = ground_truth_dict['pred/valid_mask']
            gt_occupancy_flow_map_mask = (torch.sum(gt_valid_mask, dim=-2) > 0)

            outputs = model.forward(his_occupancy_map, his_flow_map, None, his_observed_agent_features, his_observed_agent_features)
            pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits = parse_outputs_OFMPNet(outputs, train=True)

            occupancy_flow_map_loss_dict = occupancy_flow_map_loss.compute(
                pred_observed_occupancy_logits, 
                pred_occluded_occupancy_logits, 
                pred_flow_logits, 
                gt_observed_occupancy_logits, 
                gt_occluded_occupancy_logits, 
                gt_flow, 
                flow_origin_occupancy,
                gt_occupancy_flow_map_mask,
            )
            loss_value = torch.sum(sum(occupancy_flow_map_loss_dict.values()))
    

            # backward pass
            optimizer.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            occupancy_flow_map_loss.update(occupancy_flow_map_loss_dict)
            # trajectory_loss.update(trajectory_loss_dict)

            global_step += 1
            occupancy_flow_map_loss_res_dict = occupancy_flow_map_loss.get_result()
            # trajectory_loss_res_dict = trajectory_loss.get_result()

            if gpu_id == 0:
                if batch_idx % log_interval == 0:
                    logger.add_scalars(main_tag="train_occupancy_flow_map_loss", tag_scalar_dict=occupancy_flow_map_loss_res_dict, global_step=global_step)  
                    # logger.add_scalars(main_tag="train_trajectory_loss", tag_scalar_dict=trajectory_loss_res_dict, global_step=global_step)
            
        scheduler.step()
        
        ## validate
        
        occupancy_flow_map_loss.reset()
        # trajectory_loss.reset()
        occupancy_flow_map_metrics = OccupancyFlowMapMetrics(gpu_id, no_warp=False)
        
        # TODO: Add the trajectory metrics
        torch.cuda.empty_cache()
        model.eval()
        if enable_ddp:
            val_dataloader.sampler.set_epoch(epoch)
        with torch.no_grad():
            loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
            for batch_idx, data in loop:
                
                input_dict, ground_truth_dict = parse_data(data, gpu_id, config)
                # //[ ]Currently, only the current scene is being used
                input_dict = input_dict['cur']
                ground_truth_dict = ground_truth_dict['cur']
                # get the input
                his_occupancy_map = input_dict['his/observed_occupancy_map']
                his_flow_map = input_dict['his/flow_map'][:, : ,:, -1, :]  # only use the last time step for flow map
                his_observed_agent_features = input_dict['his/observed_agent_features']
                flow_origin_occupancy = input_dict['flow_origin_occupancy_map']

                # get the ground truth
                # [B, H, W, T ,1]
                gt_occluded_occupancy_logits = ground_truth_dict['pred/occluded_occupancy_map']
                gt_observed_occupancy_logits = ground_truth_dict['pred/observed_occupancy_map']
                gt_flow = ground_truth_dict['pred/flow_map']
                # gt_trajectories = ground_truth_dict['pred/trajectories']
                gt_valid_mask = ground_truth_dict['pred/valid_mask']
                gt_occupancy_flow_map_mask = (torch.sum(gt_valid_mask, dim=-2) > 0)

                outputs = model.forward(his_occupancy_map, his_flow_map, None, his_observed_agent_features, his_observed_agent_features)
                pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits = parse_outputs_OFMPNet(outputs, train=False)
                
                occupancy_flow_map_loss_dict = occupancy_flow_map_loss.compute(
                    pred_observed_occupancy_logits, 
                    pred_occluded_occupancy_logits, 
                    pred_flow_logits, 
                    gt_observed_occupancy_logits, 
                    gt_occluded_occupancy_logits, 
                    gt_flow, 
                    flow_origin_occupancy,
                    gt_occupancy_flow_map_mask,
                )

                occupancy_flow_map_metrics_dict = occupancy_flow_map_metrics.compute(
                    pred_observed_occupancy_logits, 
                    pred_occluded_occupancy_logits, 
                    pred_flow_logits, 
                    gt_observed_occupancy_logits, 
                    gt_occluded_occupancy_logits, 
                    gt_flow, 
                    flow_origin_occupancy, 
                    gt_occupancy_flow_map_mask,
                )

                occupancy_flow_map_metrics.update(occupancy_flow_map_metrics_dict)

                occupancy_flow_map_loss.update(occupancy_flow_map_loss_dict)

            occupancy_flow_map_loss_res_dict = occupancy_flow_map_loss.get_result()
            # trajectory_loss_res_dict = trajectory_loss.get_result()
            occupancy_flow_map_metrics_res_dict = occupancy_flow_map_metrics.get_result()
            # trajectory_metrics_res_dict = trajectory_metrics.get_result()
            if gpu_id == 0:
                    logger.add_scalars(main_tag="val_occupancy_flow_map_metrics", tag_scalar_dict=occupancy_flow_map_metrics_res_dict, global_step=global_step)
                    logger.add_scalars(main_tag="val_occupancy_flow_map_loss", tag_scalar_dict=occupancy_flow_map_loss_res_dict, global_step=global_step)

        if gpu_id == 0:
            if (epoch+1) % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_dir, checkpoint_total_limit)
    destroy_process_group()


if __name__ == "__main__":
    # ============= Parse Argument =============
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("--config", type=str, default="configs/model_configs/OFMPNet.py", help="config file")
    args = parser.parse_args()
    # ============= Load Configuration =============
    config = load_config(args.config)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    world_size = torch.cuda.device_count()
    mp.spawn(model_training, args=(world_size, config), nprocs=world_size)
    # model_training(0, world_size, config, enable_ddp=False)

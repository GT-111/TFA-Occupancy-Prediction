import os
import torch
import warnings
import argparse
# import model
from models.AROccFlowNet.occupancy_flow_model_auto_regressive_three_scenes import AutoRegWrapper
# import loss and metrics

from evaluation.losses.occupancy_flow_map_loss import OccupancyFlowMapLoss
from evaluation.metrics.occupancy_flow_map_metrics import OccupancyFlowMapMetrics
# import training utils
from tqdm import tqdm
from utils.training_utils import load_checkpoint, save_checkpoint
from datasets.I24Motion.utils.dataset_utils import get_dataloader
from datasets.I24Motion.utils.training_utils import parse_data
# import distributed training
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datasets.I24Motion.utils.dataset_utils import get_dataloader_ddp
# import config
from configs.utils.config import load_config
# import tensorboard
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
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
    model = AutoRegWrapper(model_config.auto_regressive_predictor).to(gpu_id)
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
        
        def get_teacher_forcing_prob(current_step, total_steps, p_start=0.4, p_end=0.0):
                    fraction = min(current_step / total_steps, 1.0)
                    p = p_start + fraction * (p_end - p_start)
                    return p
                
        tf_prob = get_teacher_forcing_prob(epoch, max_epochs)
        # print(f"Teacher forcing probability: {tf_prob}")
        tf_prob = 0.0
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, data in loop:
            
            input_dict, ground_truth_dict = parse_data(data, gpu_id, config)

            # get the input
            prv_occupancy_map = input_dict['prv']['his/observed_occupancy_map']
            cur_occupancy_map = input_dict['cur']['his/observed_occupancy_map']
            nxt_occupancy_map = input_dict['nxt']['his/observed_occupancy_map']
            
            # get the ground truth
            gt_prv_occupancy_map = ground_truth_dict['prv']['pred/observed_occupancy_map']
            gt_nxt_occupancy_map = ground_truth_dict['nxt']['pred/observed_occupancy_map']
            gt_observed_occupancy_logits = ground_truth_dict['cur']['pred/observed_occupancy_map']
            gt_valid_mask = ground_truth_dict['cur']['pred/valid_mask']
            gt_occupancy_flow_map_mask = (torch.sum(gt_valid_mask, dim=-2) > 0)

            pred_observed_occupancy_logits = model.forward(prv_occupancy_map=prv_occupancy_map,
                                                           cur_occupancy_map=cur_occupancy_map,
                                                           nxt_occupancy_map=nxt_occupancy_map,
                                                           gt_prv_occupancy_map=gt_prv_occupancy_map,
                                                           gt_nxt__occupancy_map=gt_nxt_occupancy_map,
                                                           training=True)

            loss_dic = occupancy_flow_map_loss.compute_occypancy_map_loss(pred_observed_occupancy_logits, gt_observed_occupancy_logits, gt_occupancy_flow_map_mask)

            loss_value = loss_dic['observed_occupancy_cross_entropy']
            # backward pass
            optimizer.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            

            global_step += 1
            

            if gpu_id == 0:
                if batch_idx % log_interval == 0:
                    logger.add_scalars(main_tag="train_occupancy_flow_map_loss", tag_scalar_dict=loss_dic, global_step=global_step)  
            
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
        vehicles_observed_occupancy_auc = []
        vehicles_observed_occupancy_iou = []
        observed_occupancy_cross_entropy = []
        with torch.no_grad():
            loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
            for batch_idx, data in loop:
                
                input_dict, ground_truth_dict = parse_data(data, gpu_id, config)
                # //[ ]Currently, only the current scene is being used
                # get the input
                prv_occupancy_map = input_dict['prv']['his/observed_occupancy_map']
                cur_occupancy_map = input_dict['cur']['his/observed_occupancy_map']
                nxt_occupancy_map = input_dict['nxt']['his/observed_occupancy_map']
                
                # get the ground truth
                gt_prv_occupancy_map = ground_truth_dict['prv']['pred/observed_occupancy_map']
                gt_nxt_occupancy_map = ground_truth_dict['nxt']['pred/observed_occupancy_map']
                gt_observed_occupancy_logits = ground_truth_dict['cur']['pred/observed_occupancy_map']
                gt_valid_mask = ground_truth_dict['cur']['pred/valid_mask']
                gt_occupancy_flow_map_mask = (torch.sum(gt_valid_mask, dim=-2) > 0)

                pred_observed_occupancy_logits = model.forward(prv_occupancy_map=prv_occupancy_map,
                                                           cur_occupancy_map=cur_occupancy_map,
                                                           nxt_occupancy_map=nxt_occupancy_map,
                                                           gt_prv_occupancy_map=gt_prv_occupancy_map,
                                                           gt_nxt__occupancy_map=gt_nxt_occupancy_map,
                                                           training=True)

                loss_dic = occupancy_flow_map_loss.compute_occypancy_map_loss(pred_observed_occupancy_logits, gt_observed_occupancy_logits, gt_occupancy_flow_map_mask)

                observed_occupancy_cross_entropy.append(loss_dic['observed_occupancy_cross_entropy'])
                pred_observed_occupancy_logits = torch.sigmoid(pred_observed_occupancy_logits)
                
                occupancy_flow_map_metrics_dict = occupancy_flow_map_metrics.compute_occupancy_metrics(pred_observed_occupancy_logits, gt_observed_occupancy_logits, gt_occupancy_flow_map_mask)
                # print(occupancy_flow_map_metrics_dict)
                vehicles_observed_occupancy_auc.append(occupancy_flow_map_metrics_dict['vehicles_observed_occupancy_auc'])
                vehicles_observed_occupancy_iou.append(occupancy_flow_map_metrics_dict['vehicles_observed_occupancy_iou'])
                
            occupancy_flow_map_metrics_res_dict = {'vehicles_observed_occupancy_auc': torch.mean(torch.stack(vehicles_observed_occupancy_auc)),
                                                    'vehicles_observed_occupancy_iou': torch.mean(torch.stack(vehicles_observed_occupancy_iou))}
            occupancy_flow_map_loss_res_dict = {'observed_occupancy_cross_entropy': torch.mean(torch.stack(observed_occupancy_cross_entropy))}
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
    parser.add_argument("--config", type=str, default="configs/model_configs/AROccFlowNetAutoRegressiveThreeScenes.py", help="config file")
    args = parser.parse_args()
    # ============= Load Configuration =============
    config = load_config(args.config)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    world_size = torch.cuda.device_count()
    mp.spawn(model_training, args=(world_size, config), nprocs=world_size)
    # model_training(0, world_size, config, enable_ddp=False)

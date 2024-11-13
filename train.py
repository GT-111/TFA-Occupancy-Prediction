from utils.file_utils import get_config, get_last_checkpoint
from utils.dataset_utils import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from tqdm import tqdm
import utils.training_utils as training_utils
from losses.occupancy_flow_loss import OGMFlow_loss
import torch
import os
from metrics.occu_metrics import compute_occupancy_flow_metrics
from modules.model import OFMPNet
from metrics.metric import OGMFlowMetrics
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
ogm_weight  = 100
occ_weight  = 100
flow_weight = 1.0
flow_origin_weight = 500



def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)



def setup(config, gpu_id):
    """
    """
    
    cfg=dict(input_size=(256,128), window_size=8, embed_dim=96, depths=[2,2,2], num_heads=[3,6,12])
    model = OFMPNet(cfg,actor_only=True,sep_actors=False,fg_msa=True).to(gpu_id)
    model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    loss_fn = OGMFlow_loss(config=config, 
                           ogm_weight=ogm_weight,
                           occ_weight=occ_weight,
                           flow_weight=flow_weight,
                           flow_origin_weight=flow_origin_weight,
                           replica=1.0,
                           no_use_warp=False,
                           use_pred=False,
                           use_focal_loss=True,
                           use_gt=False)
    
    optimizer = torch.optim.NAdam(params=model.parameters(), 
                                  lr=config.training.optimizer.lr,
                                  weight_decay=config.training.optimizer.weight_decay) 
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                step_size=config.training.scheduler.step_size,
                                                gamma=config.training.scheduler.gamma) 
    
    return model, loss_fn, optimizer, scheduler



def model_training(gpu_id, world_size, config):
    
    ddp_setup(gpu_id, world_size)
    logger = SummaryWriter(log_dir=config.paths.logs)
    model, loss_fn, optimizer, scheduler = setup(config, gpu_id)
    train_dataloader, val_dataloader, _ = get_dataloader(config)
    global_step = 0
    if get_last_checkpoint(config.paths.checkpoints) is not None:
        
        checkpoint = torch.load(get_last_checkpoint(config.paths.checkpoints))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        continue_ep = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        if gpu_id == 0:
            print(f'Continue_training...ep:{continue_ep+1}')
    else:
        continue_ep = 0
    
    
    
    for epoch in range(config.training.epochs):
        if epoch<continue_ep:
            if gpu_id == 0:
                print("\nskip epoch {}/{}".format(epoch+1, config.training.epochs))
            continue
        
        model.train()
        train_loss = MeanMetric().to(gpu_id)
        train_loss_occ = MeanMetric().to(gpu_id)
        train_loss_flow = MeanMetric().to(gpu_id)
        train_loss_warp = MeanMetric().to(gpu_id)
        
        train_dataloader.sampler.set_epoch(epoch)
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch, data in loop:
            
            input_dict, ground_truth_dict = training_utils.parse_data(data, gpu_id, config)
            for key, val in input_dict.items():
                # print if value has nan
                if torch.isnan(val).any():
                    input_dict[key] = torch.where(torch.isnan(val), torch.zeros_like(val), val)
            his_occ = input_dict['cur/state/his/observed_occupancy_map']
            flow_origin_occupancy = his_occ[:, -1, :, :, torch.newaxis]
            map_size = (256,128)
            map_img = torch.zeros([his_occ.size(0),*map_size,3]).to(gpu_id)
            outputs = model(input_dict, map_img, training=True)
            pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits = training_utils.parse_outputs(outputs, config.task.num_waypoints)
            gt_occluded_occupancy_logits = ground_truth_dict['cur/state/pred/occluded_occupancy_map']
            gt_observed_occupancy_logits = ground_truth_dict['cur/state/pred/observed_occupancy_map']
            gt_flow = ground_truth_dict['cur/state/pred/flow_map']
            
            loss_dict = loss_fn(pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits, gt_observed_occupancy_logits, gt_occluded_occupancy_logits, gt_flow, flow_origin_occupancy)
            loss_value = torch.sum(sum(loss_dict.values()))
            
            # backward pass
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            train_loss.update(loss_dict['observed_xe'])
            train_loss_occ.update(loss_dict['occluded_xe'])
            train_loss_flow.update(loss_dict['flow'])
            train_loss_warp.update(loss_dict['flow_warp_xe'])
            global_step += 1
            obs_loss  = train_loss.compute()/ogm_weight
            occ_loss  = train_loss_occ.compute()/occ_weight
            flow_loss = train_loss_flow.compute()/flow_weight
            warp_loss = train_loss_warp.compute()/flow_origin_weight
            if gpu_id == 0:
                if batch % 20 == 0:
                    logger.add_scalars(main_tag="train_loss",
                                    tag_scalar_dict={
                                        "observed_xe": obs_loss * ogm_weight,
                                        "occluded_xe": occ_loss * occ_weight,
                                        "flow": flow_loss * flow_weight,
                                        "flow_warp_xe": warp_loss * flow_origin_weight,
                                        "loss": loss_value
                                    },
                                    global_step=global_step
                                    )   
            
        scheduler.step()
        
        ## validate
        
        valid_loss      = MeanMetric().to(gpu_id)
        valid_loss_occ  = MeanMetric().to(gpu_id)
        valid_loss_flow = MeanMetric().to(gpu_id)
        valid_loss_warp = MeanMetric().to(gpu_id)

        valid_metrics = OGMFlowMetrics(gpu_id, no_warp=False)


        model.eval()
        
        with torch.no_grad():
            loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
            for batch, data in loop:
                input_dict, ground_truth_dict = training_utils.parse_data(data, gpu_id, config)
                for key, val in input_dict.items():
                    # print if value has nan
                    if torch.isnan(val).any():
                        input_dict[key] = torch.where(torch.isnan(val), torch.zeros_like(val), val)
                his_occ = input_dict['cur/state/his/observed_occupancy_map']
                flow_origin_occupancy = his_occ[:, -1, :, :, torch.newaxis]
                map_size = (256,128)
                map_img = torch.zeros([1,*map_size,3]).to(gpu_id)
                # forward pass
                outputs = model(input_dict, map_img, training=False)

                # compute losses
                pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits = training_utils.parse_outputs(outputs, config.task.num_waypoints)
                gt_occluded_occupancy_logits = ground_truth_dict['cur/state/pred/occluded_occupancy_map']
                gt_observed_occupancy_logits = ground_truth_dict['cur/state/pred/observed_occupancy_map']
                B, T, H, W = gt_observed_occupancy_logits.shape
                gt_occluded_occupancy_logits = gt_occluded_occupancy_logits.reshape(B, T, H, W, 1)
                gt_observed_occupancy_logits = gt_observed_occupancy_logits.reshape(B, T, H, W, 1)
                
                gt_flow = ground_truth_dict['cur/state/pred/flow_map']
                loss_dict = loss_fn(pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits, gt_observed_occupancy_logits, gt_occluded_occupancy_logits, gt_flow, flow_origin_occupancy)
                loss_value = torch.sum(sum(loss_dict.values()))
                
                valid_loss.update(loss_dict['observed_xe'])
                valid_loss_occ.update(loss_dict['occluded_xe'])
                valid_loss_flow.update(loss_dict['flow'])
                valid_loss_warp.update(loss_dict['flow_warp_xe'])
                
                obs_loss  = valid_loss.compute()/ogm_weight
                occ_loss  = valid_loss_occ.compute()/occ_weight
                flow_loss = valid_loss_flow.compute()/flow_weight
                warp_loss = valid_loss_warp.compute()/flow_origin_weight
                
                pred_observed_occupancy_logits = torch.sigmoid(pred_observed_occupancy_logits)
                pred_occluded_occupancy_logits = torch.sigmoid(pred_occluded_occupancy_logits)
                
                metrics_dict = compute_occupancy_flow_metrics(config, pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits, gt_observed_occupancy_logits, gt_occluded_occupancy_logits, gt_flow, flow_origin_occupancy, no_warp=False)

                valid_metrics.update(metrics_dict)
                break
            val_res_dict = valid_metrics.compute()
            if gpu_id == 0:
                logger.add_scalars(main_tag="val_metrics",
                    tag_scalar_dict={
                        'vehicles_observed_auc': val_res_dict['vehicles_observed_auc'],
                        'vehicles_occluded_auc': val_res_dict['vehicles_occluded_auc'],
                        'vehicles_observed_iou': val_res_dict['vehicles_observed_iou'],
                        'vehicles_occluded_iou': val_res_dict['vehicles_occluded_iou'],
                        'vehicles_flow_epe': val_res_dict['vehicles_flow_epe'],
                        'vehicles_flow_warped_occupancy_auc': val_res_dict['vehicles_flow_warped_occupancy_auc'],
                        'vehicles_flow_warped_occupancy_iou': val_res_dict['vehicles_flow_warped_occupancy_iou'],
                    },
                    global_step=global_step
                )
                
                logger.add_scalars(main_tag="val_loss",
                                tag_scalar_dict={
                                "observed_xe": obs_loss * ogm_weight,
                                "occluded_xe": occ_loss * occ_weight,
                                "flow": flow_loss * flow_weight,
                                "flow_warp_xe": warp_loss * flow_origin_weight
                                },
                                global_step=global_step
                    )
            
        if gpu_id == 0:
            if (epoch+1) % config.training.checkpoint_interval == 0:
                training_utils.save_checkpoint(model, optimizer, scheduler, epoch, config.paths.checkpoints, global_step)
    destroy_process_group()










if __name__ == "__main__":
    config = get_config('./config.json')
    checkpoints_path = config.paths.checkpoints
    os.path.exists(checkpoints_path) or os.makedirs(checkpoints_path)
    # os.environ["NCCL_P2P_DISABLE"] = "1"

    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    world_size = torch.cuda.device_count()
    mp.spawn(model_training, args=(world_size, config), nprocs=world_size)

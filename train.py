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
import time
ogm_weight  = 1000.0
occ_weight  = 1000.0
flow_weight = 1.0
flow_origin_weight = 1000.0

def setup(config, gpu_id):
    """
    """
    
    cfg=dict(input_size=(256,128), window_size=8, embed_dim=96, depths=[2,2,2], num_heads=[3,6,12])
    model = OFMPNet(cfg,actor_only=True,sep_actors=False,fg_msa=True).to(gpu_id)

    loss_fn = OGMFlow_loss(config=config, 
                           ogm_weight=1000.0,
                           occ_weight=1000.0,
                           flow_weight=1.0,
                           flow_origin_weight=1000.0,
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



def model_training(config, gpu_id, logger:SummaryWriter):
    
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
    
    
    model, loss_fn, optimizer, scheduler = setup(config, gpu_id)
    
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
        
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch, data in loop:
            prv_time = time.time()
            input_dict, ground_truth_dict = training_utils.parse_data(data, gpu_id, config)
            for key, val in input_dict.items():
                # print if value has nan
                if torch.isnan(val).any():
                    input_dict[key] = torch.where(torch.isnan(val), torch.zeros_like(val), val)
            his_occ = input_dict['cur/state/his/observed_occupancy_map']
            flow_origin_occupancy = his_occ[:, -1, :, :, torch.newaxis]
            map_size = (256,128)
            map_img = torch.zeros([1,*map_size,3]).to(gpu_id)
            print(time.time()-prv_time)
            prv_time = time.time()
            outputs = model(input_dict, map_img, training=True)
            print(time.time()-prv_time)
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
            if batch % 20 == 0:
                logger.add_scalars(main_tag="train_loss",
                                tag_scalar_dict={
                                    "observed_xe": train_loss.compute()/ogm_weight,
                                    "occluded_xe": train_loss_occ.compute()/occ_weight,
                                    "flow": train_loss_flow.compute()/flow_weight,
                                    "flow_warp_xe": train_loss_warp.compute()/flow_origin_weight
                                },
                                global_step=global_step
                                )   
            break
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
                
                pred_observed_occupancy_logits = torch.sigmoid(pred_observed_occupancy_logits)
                pred_occluded_occupancy_logits = torch.sigmoid(pred_occluded_occupancy_logits)
                
                metrics_dict = compute_occupancy_flow_metrics(config, pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits, gt_observed_occupancy_logits, gt_occluded_occupancy_logits, gt_flow, flow_origin_occupancy, no_warp=False)

                valid_metrics.update(metrics_dict)
                
            val_res_dict = valid_metrics.compute()
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
                            "observed_xe": valid_loss.compute()/ogm_weight,
                            "occluded_xe": valid_loss_occ.compute()/occ_weight,
                            "flow": valid_loss_flow.compute()/flow_weight,
                            "flow_warp_xe": valid_loss_warp.compute()/flow_origin_weight
                            },
                            global_step=global_step
                )
            
            
        if (epoch+1) % config.training.checkpoint_interval == 0:
            training_utils.save_checkpoint(model, optimizer, scheduler, epoch, config.paths.checkpoints)










config = get_config('./config.json')
checkpoints_path = config.paths.checkpoints
os.path.exists(checkpoints_path) or os.makedirs(checkpoints_path)

if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    # add tensor board
    logger = SummaryWriter(log_dir=config.paths.logs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_training(config=config, gpu_id=device, logger=logger)
else:
    print('No GPU available')
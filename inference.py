from utils.file_utils import get_config, get_last_checkpoint
from utils.dataset_utils import get_dataloader_ddp, get_road_map, get_trajs
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from tqdm import tqdm
from  utils.training_utils import parse_data, parse_outputs
from losses.occupancy_flow_loss import OGMFlow_loss
import torch
import os
from metrics.occu_metrics import compute_occupancy_flow_metrics
from modules.OFMPNet import OFMPNet
from metrics.metric import OGMFlowMetrics
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from visualize.animation import visualize
ogm_weight  = 1000.0
occ_weight  = 1000.0
flow_weight = 1.0
flow_origin_weight = 1000.0


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
    
    model = OFMPNet(config).to(gpu_id)
    model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    loss_fn = OGMFlow_loss(config=config, 
                           ogm_weight=ogm_weight,
                           occ_weight=occ_weight,
                           flow_weight=flow_weight,
                           flow_origin_weight=flow_origin_weight,
                           replica=1.0,
                           no_use_warp=True,
                           use_pred=False,
                           use_focal_loss=True,
                           use_gt=False)
    
    optimizer = torch.optim.NAdam(params=model.parameters(), 
                                  lr=config.training_settings.optimizer.learning_rate,
                                #   weight_decay=config.training_settings.optimizer.weight_decay
                                ) 
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                step_size=config.training_settings.scheduler.step_size,
                                                gamma=config.training_settings.scheduler.gamma) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5) 
    return model, loss_fn, optimizer, scheduler



def model_inference(gpu_id, world_size, config):
    ddp_setup(gpu_id, world_size)
    logger = SummaryWriter(log_dir=config.paths.logs)
    model, loss_fn, optimizer, scheduler = setup(config, gpu_id)
    _, val_dataloader, _ = get_dataloader_ddp(config)
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
    
        
        
    ## validate
    
    valid_loss      = MeanMetric().to(gpu_id)
    valid_loss_occ  = MeanMetric().to(gpu_id)
    valid_loss_flow = MeanMetric().to(gpu_id)
    valid_loss_warp = MeanMetric().to(gpu_id)
    valid_metrics = OGMFlowMetrics(gpu_id, no_warp=False)
    model.eval()
    road_map = torch.from_numpy(get_road_map(config)).to(gpu_id, dtype=torch.float32)
    with torch.no_grad():
        loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for batch, data in loop:
            input_dict, ground_truth_dict = parse_data(data, gpu_id, config)
            for key, val in input_dict.items():
                # print if value has nan
                if torch.isnan(val).any():
                    input_dict[key] = torch.where(torch.isnan(val), torch.zeros_like(val), val)
            his_occupancy_map = input_dict['cur/state/his/observed_occupancy_map']
            his_flow_map = input_dict['cur/state/his/flow_map']
            flow_origin_occupancy = his_occupancy_map[:, -1, :, :, torch.newaxis]
            obs_traj, occ_traj= get_trajs(input_dict, config)
            # self,occupancy_map, flow_map, road_map, obs_traj, occ_traj
            his_occupancy_map = his_occupancy_map.permute([0,2,3,1]) # B H W T
            his_flow_map = his_flow_map[:,-1,:,:,:] # B H W 2
            outputs = model(his_occupancy_map, his_flow_map, road_map, obs_traj, occ_traj)
            # compute losses
            pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits = parse_outputs(outputs, config.task_config.num_waypoints)
            
            input_dict_np = {key: val.cpu().numpy().squeeze() for key, val in input_dict.items()}
            # add ground truth to the input_dict_np
            ground_truth_dict_np = {key: val.cpu().numpy().squeeze() for key, val in ground_truth_dict.items()}
            input_dict_np.update(ground_truth_dict_np)
            pred_dict = {'observed_occupancy_map': pred_observed_occupancy_logits, 'flow_map': pred_flow_logits, 'occluded_occupancy_map': pred_occluded_occupancy_logits}
            pred_dict_np = {key: val.cpu().numpy().squeeze() for key, val in pred_dict.items()}
            # visualize(config, input_dict_np, str(batch), vis_occ=True, vis_flow=True,vis_optical_flow=False, pred_dict=pred_dict_np, ground_truth=False, valid_dict={'cur': 1})
            visualize(config, input_dict_np, str(batch), vis_occ=False, vis_flow=True, vis_optical_flow=False,pred_dict=pred_dict_np, ground_truth=False, valid_dict={'cur': 1})
            visualize(config, input_dict_np, str(batch), vis_occ=True, vis_flow=False, vis_optical_flow=False,pred_dict=pred_dict_np, ground_truth=False, valid_dict={'cur': 1})
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
            global_step+=1
        val_res_dict = valid_metrics.compute()
    destroy_process_group()








if __name__ == "__main__":
    config = get_config('./config_12_inference.yaml')
    checkpoints_path = config.paths.checkpoints
    os.path.exists(checkpoints_path) or os.makedirs(checkpoints_path)
    # os.environ["NCCL_P2P_DISABLE"] = "1"

    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    world_size = torch.cuda.device_count()
    mp.spawn(model_inference, args=(world_size, config), nprocs=world_size)

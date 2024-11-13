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



def model_inference(gpu_id, config):
    
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
            
            









if __name__ == "__main__":
    config = get_config('./config.json')
    checkpoints_path = config.paths.checkpoints
    os.path.exists(checkpoints_path) or os.makedirs(checkpoints_path)
    # os.environ["NCCL_P2P_DISABLE"] = "1"

    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    gpu_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_inference(gpu_id, config)


import os
import math
import argparse
from tqdm import tqdm

from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2

import torch
from torch.distributed import init_process_group, destroy_process_group

from torchmetrics import MeanMetric
from modules.model import OFMPNet
from dataset.dataset_utils import get_dataloader
from utils.file_utils import get_config
config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
dataset_config = get_config()
with open('./waymo_ofp.config', 'r') as f:
    config_text = f.read()
    text_format.Parse(config_text, config)

# loss weights
ogm_weight  = 1000.0
occ_weight  = 1000.0
flow_weight = 1.0
flow_origin_weight = 1000.0

def setup(gpu_id):
    """
    Setup model, DDP, loss, optimizer and scheduler
    """
    cfg=dict(input_size=(256,128), window_size=8, embed_dim=96, depths=[2,2,2], num_heads=[3,6,12])
    model = OFMPNet(cfg,actor_only=True,sep_actors=False, fg_msa=True, fg=True).to(gpu_id)
    optimizer = torch.optim.NAdam(model.parameters(), lr=LR) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5) 
    return model, loss_fn, optimizer, scheduler


def model_training(gpu_id):
    """
    Model training and validation
    """

    model, loss_fn, optimizer, scheduler = setup(gpu_id)
    train_loader = get_dataloader(config=dataset_config)
   
    if CHECKPOINT_PATH is not None:
        # if checkpoint path given, load weights
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        continue_ep = checkpoint['epoch'] + 1
        if gpu_id == 0:
            print(f'Continue_training...ep:{continue_ep+1}')
    else:
        continue_ep = 0

    
    train_size = 0
    val_size = 0
    for epoch in range(EPOCHS):

        if epoch<continue_ep:
            if gpu_id == 0:
                print("\nskip epoch {}/{}".format(epoch+1, EPOCHS))
            continue

        # TRAINING
        if gpu_id == 0:
            print(f"Epoch {epoch+1}\n-------------------------------")
        size = train_size or 0
        train_loss = MeanMetric().to(gpu_id)
        train_loss_occ = MeanMetric().to(gpu_id)
        train_loss_flow = MeanMetric().to(gpu_id)
        train_loss_warp = MeanMetric().to(gpu_id)

        model.train()

        train_loader.sampler.set_epoch(epoch)

        loop = tqdm(enumerate(train_loader), total=math.ceil(BATCH_SIZE))if gpu_id == 0 else enumerate(train_loader)
        for batch, data in loop:
            batch_size = dataset_config.dataset.batch_size
            # inputs: will automatically be put on right device when passed to model 
            gria_size_x = config.dataset.grid_size_x
            grid_size_y = config.dataset.grid_size_y
            map_size = (gria_size_x, grid_size_y)
            map_img = torch.zeros([batch_size,*map_size,3])

            road_lines_y = [12, 24, 36, 48, 60, -12, -24, -36, -48, -60]
            # Loop through each road line y-coordinate and draw horizontal lines
            for road_line_y in road_lines_y:
                y_coord = road_line_y * grid_size_y / 160 + grid_size_y / 2
                map_img[:,:,int(y_coord),:] = 1
            
            input_dic = {}
            input_dic['prv/his/occupancy_map'] = data['prv/his/occupancy_map'].to(gpu_id)
            input_dic['prv/his/flow_map'] = data['prv/his/flow_map'].to(gpu_id)
            
            input_dic['cur/his/occupancy_map'] = data['cur/his/occupancy_map'].to(gpu_id)
            input_dic['cur/his/flow_map'] = data['cur/his/flow_map'].to(gpu_id)

            input_dic['nxt/his/occupancy_map'] = data['nxt/his/occupancy_map'].to(gpu_id)
            input_dic['nxt/his/flow_map'] = data['nxt/his/flow_map'].to(gpu_id)
            
            input_dic['cur/his/x_position'] = data['cur/his/x_position'][..., None].to(gpu_id)
            input_dic['cur/his/y_position'] = data['cur/his/y_position'][..., None].to(gpu_id)
            input_dic['cur/his/x_velocity'] = data['cur/his/x_velocity'][..., None].to(gpu_id)
            input_dic['cur/his/y_velocity'] = data['cur/his/y_velocity'][..., None].to(gpu_id)
            input_dic['cur/his/yaw_angle'] = data['cur/his/yaw_angle'][..., None].to(gpu_id)
            
            gt_flow = data['cur/pred/flow_map'].to(gpu_id)
            gt_occ_ogm = data['cur/pred/occupancy_map'].to(gpu_id)
            gt_obs_ogm = data['cur/pred/occupancy_map'].to(gpu_id)

            # forward pass
            outputs = model(input_dic, map_img, training=True)

            loss_dict = loss_fn(outputs, gt_flow, gt_occ_ogm, gt_obs_ogm)
            loss_value = torch.sum(sum(loss_dict.values()))
    
            # backward pass
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            # update losses
            train_loss.update(loss_dict['observed_xe'])
            train_loss_occ.update(loss_dict['occluded_xe'])
            train_loss_flow.update(loss_dict['flow'])
            train_loss_warp.update(loss_dict['flow_warp_xe'])

            obs_loss  = train_loss.compute()/ogm_weight
            occ_loss  = train_loss_occ.compute()/occ_weight
            flow_loss = train_loss_flow.compute()/flow_weight
            warp_loss = train_loss_warp.compute()/flow_origin_weight
            
            if gpu_id == 0:
                # print training losses
                
                print(f"\nobs. loss: {obs_loss:>7f}, occl. loss:  {occ_loss:>7f}, flow loss: {flow_loss:>7f}, warp loss: {warp_loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)

    
        scheduler.step()
        if gpu_id == 0:
            # save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_value,
            }, f'{SAVE_DIR}/epoch_{epoch+1}.pt')
    
    destroy_process_group()

    print(f'Finished training. Model saved in {args.save_dir}!')

parser = argparse.ArgumentParser(description='OFMPNet Training')
parser.add_argument('--save_dir', type=str,
                    help='saving directory', default="./experiments")
parser.add_argument('--file_dir', type=str, help='Training Val Dataset directory',
                    default="./Waymo_Dataset/preprocessed_data")
parser.add_argument('--model_path', type=str,
                    help='loaded weight path', default='./checkpoints/')
parser.add_argument('--batch_size', type=int, help='batch_size', default=20)
parser.add_argument('--epochs', type=int, help='training eps', default=15)
parser.add_argument('--lr', type=float,
                    help='initial learning rate', default=1e-4)
parser.add_argument('--wandb', type=bool, help='wandb logging', default=False)
parser.add_argument(
    '--title', help='choose a title for your wandb/log process', default="ofmpnet")
args = parser.parse_args()


# Parameters
SAVE_DIR = args.save_dir + f"/{args.title}"
FILES_DIR = args.file_dir
CHECKPOINT_PATH = args.model_path

# Hyper parameters
NUM_PRED_CHANNELS = 4
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr

os.path.exists(SAVE_DIR) or os.makedirs(SAVE_DIR)

if __name__ == "__main__":

    model_training()
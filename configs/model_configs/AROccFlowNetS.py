from random import shuffle
from tabnanny import check

from lark import logger
from configs.utils.config import load_config
import os

from evaluation import losses
from evaluation.losses import occupancy_flow_map_loss, trajectory_loss
# ============= Seed ===================
random_seed = 42
# ============= Path ===================
project_name = 'AROccFlowNetS'
# checkpoints = "./checkpoints/"
# logs = "./logs/"
exp_dir = './exp/'  # PATH TO YOUR EXPERIMENT FOLDER
project_dir = os.path.join(exp_dir, project_name)
# ============= Dataset Parameters=================
dataset_config = load_config("configs/dataset_configs/I24Motion_config.py")
occupancy_flow_map_config = dataset_config.occupancy_flow_map
occupancy_flow_map_height = occupancy_flow_map_config.occupancy_flow_map_height
occupancy_flow_map_width = occupancy_flow_map_config.occupancy_flow_map_width

task_config = dataset_config.task
num_his_points = task_config.num_his_points
num_waypoints = task_config.num_waypoints

paths_config = dataset_config.paths
generated_data_path = paths_config.generated_data_path
total_data_samples = 40000
# ============= Model Parameters =================
input_dim = 3 # occupancy, flow_x, flow_y
hidden_dim = 64
num_states = 9# TODO: Define the number of states
num_heads = 4
dropout_prob=0.1
num_motion_mode=6 # number of future motion modes

# ============= Train Parameters =================
num_machines = 1
gpu_ids = [0,1]
activation_checkpointing = True
max_epochs = 30
batch_size = 4
# ============= Test Parameters =================
guidance_scale = 7.5
weight_path = None  # None is the last ckpt you have trained
# ============= Config ===================
config = dict(
    project_name=project_name,
    project_dir=project_dir,
    dataloaders=dict(
        datasets=dict(
            train_ratio = 0.8,
            validation_ratio = 0.1,
            test_ratio = 0.1,
            data_path=generated_data_path,
            total_data_samples=1000,
        ),
        train=dict(
            batch_size=batch_size,
            num_workers=1,
            shuffle=True,
        ),
        val=dict(
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        ),
        test=dict(
            batch_size=1,
            num_workers=1,
            shuffle=False,
        ),
    ),
    models=dict(
        aroccflownet=dict(
            img_size=(occupancy_flow_map_height, occupancy_flow_map_width),
            num_time_steps=num_waypoints,
            hidden_dim=hidden_dim,
            nhead=num_heads,
            dropout_prob=dropout_prob,
            num_layers=1,
            
            convlstm=dict(
                input_dim=input_dim, 
                hidden_dim=[64, hidden_dim],
                kernel_size=(3, 3), 
                num_layers=2,
                batch_first=True, 
                bias=True, 
                return_all_layers=False
            ),
            convnextunet=dict(
                img_size=(occupancy_flow_map_height, occupancy_flow_map_width),
                in_chans=input_dim,
                out_channels=hidden_dim,
                embed_dims=[96, 192, 384, 768],
                temporal_depth=num_his_points - 1,
            ),
            motionpredictor=dict(
                num_states=num_states,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout_prob=dropout_prob,
                num_layers=1,
                num_motion_mode=num_motion_mode,
                num_time_steps=num_waypoints,
            ),
        ),
        
        # pretrained='runwayml/stable-diffusion-v1-5',
        # pipeline_name='StableDiffusionControlPipeline',
        # checkpoint=ckpt_2d,
        # with_ema=with_ema,
        # weight_path=weight_path,
    ),
    losses=dict(
        occupancy_flow_map_loss=dict(
            ogm_weight  = 200,
            occ_weight  = 200,
            flow_weight = 10,
            flow_origin_weight = 500,
            replica=1.0,
            no_use_warp=False,
            use_pred=False,
            use_focal_loss=True,
            use_gt=False
        ),
        trajectory_loss=dict(
            regression_weight=200,
            classification_weight=1.0
        ),
    ),
    optimizers=dict(
        type='NAdam',
        learning_rate = 0.0001,
        weight_decay = 0.0001
    ),
    schedulers=dict(
        type='StepLR',
        step_size = 3,
        gamma = 0.5
    ),
    train=dict(
        max_epochs=max_epochs,
        checkpoint_interval=1,
        checkpoint_dir=os.path.join(project_dir, 'checkpoints'),
        checkpoint_total_limit=10,
        log_interval=100,
    ),
    test=dict(

    ),
    loggers=dict(
        tensorboard=dict(
            type='Tensorboard',
            log_dir=os.path.join(project_dir, 'logs'),
        ),
    ),
)
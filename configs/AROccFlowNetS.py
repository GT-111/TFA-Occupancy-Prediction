import os
# ============= SEED ===================
random_seed = 42
# ============= PATH ===================
proj_name = 'AROccFlowNetS'
raw_data = './raw_data/'
auxiliary_data = './auxiliary_data/'
processed_data = './processed_data/'
# checkpoints = "./checkpoints/"
# logs = "./logs/"
exp_dir = './exp/'  # PATH TO YOUR EXPERIMENT FOLDER
project_dir = os.path.join(exp_dir, proj_name)
# ============= Data Parameters =================
dataset = 'I24Motion'
sample_frequency = 25 # the frequency of the input data
start_position = 58.5 # the start position of the input data in miles
end_position = 63.5 # the end position of the input data in miles
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1
# ============= TASK =================
history_length = 20
num_his_points = 10
prediction_length = 400
num_waypoints = 20
# ============= Model Parameters =================
input_dim = 3 # occupancy, flow_x, flow_y
hidden_dim = 64
img_size = (256, 256)

num_states = 10# TODO: Define the number of states
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
    project_dir=project_dir,
    # launch=dict(
    #     gpu_ids=gpu_ids,
    #     num_machines=num_machines,
    #     distributed_type=distributed_type,
    #     deepspeed_config=deepspeed_config,
    # ),
    dataloaders=dict(
        train=dict(
            # data_or_config=train_data,
            # batch_size_per_gpu=batch_size_frame_per_gpu,
            # num_workers=2,
            # transform=dict(
            #     type='DriveDreamerTransform',
            #     dst_size=img_width,
            #     mode='long',
            #     pos_name=pos_name,
            #     max_objs=max_objs_num,
            #     random_choice=True,
            #     default_prompt='a realistic driving scene.',
            #     prompt_name='sd',
            #     dd_name='image_hdmap',
            #     is_train=True,
            # ),
            # sampler=dict(
            #     type='NuscVideoSampler',
            #     cam_num=num_cams,
            #     frame_num=num_frames,
            #     hz_factor=hz_factor,
            #     video_split_rate=video_split_rate,
            #     mv_video=mv_video,
            #     view=view,
            # ),
        ),
        test=dict(

        ),
    ),
    models=dict(
        aroccflownet=dict(
            img_size=img_size,
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
                img_size=img_size,
                in_chans=input_dim,
                out_channels=hidden_dim,
                embed_dims=[96, 192, 384, 768],
                temporal_depth=num_his_points,
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
    optimizers=dict(
        type='NAdam',
        learning_rate = 0.0001,
        weight_decay = 0.0001
    ),
    schedulers=dict(
        name='StepLR',
        step_size = 3,
        gamma = 0.5
    ),
    train=dict(
        max_epochs=max_epochs,
        checkpoint_interval=1,
        checkpoint_total_limit=10,
        log_with='tensorboard',
        log_interval=100,
    ),
    test=dict(

    ),
)
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
    launch=dict(
        gpu_ids=gpu_ids,
        num_machines=num_machines,
        distributed_type=distributed_type,
        deepspeed_config=deepspeed_config,
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=train_data,
            batch_size_per_gpu=batch_size_frame_per_gpu,
            num_workers=2,
            transform=dict(
                type='DriveDreamerTransform',
                dst_size=img_width,
                mode='long',
                pos_name=pos_name,
                max_objs=max_objs_num,
                random_choice=True,
                default_prompt='a realistic driving scene.',
                prompt_name='sd',
                dd_name='image_hdmap',
                is_train=True,
            ),
            sampler=dict(
                type='NuscVideoSampler',
                cam_num=num_cams,
                frame_num=num_frames,
                hz_factor=hz_factor,
                video_split_rate=video_split_rate,
                mv_video=mv_video,
                view=view,
            ),
        ),
        test=dict(
            data_or_config=test_data,
            batch_size_per_gpu=num_frames * num_cams,
            num_workers=0,
            transform=dict(
                type='DriveDreamerTransform',
                dst_size=img_width,
                mode='long',
                pos_name=pos_name,
                max_objs=max_objs_num,
                random_choice=False,
                prompt_name='sd',
                default_prompt='a realistic driving scene.',
                dd_name='image_hdmap',
                is_train=False,
            ),
            sampler=dict(
                type='NuscVideoSampler',
                cam_num=num_cams,
                frame_num=num_frames,
                hz_factor=hz_factor,
                video_split_rate=1,
                mv_video=mv_video,
                view=view,
            ),
        ),
    ),
    models=dict(
        drivedreamer=dict(
            unet_type='UNet2DConditionModel',
            noise_scheduler_type='DDPMScheduler',
            tune_all_unet_params=tune_all_unet_params,
            unet_from_2d_to_3d=True,
            num_frames=num_frames,
            num_cams=num_cams,
            position_net_cfg=dict(
                type='PositionNet',
                in_dim=768,
                mid_dim=512,
                box_dim=16 if 'corner' in pos_name else 4,
                feature_type='text_image' if 'image' in pos_name else 'text_only',
            ),
            grounding_downsampler_cfg=dict(
                type='GroundingDownSampler',
                in_dim=3,
                mid_dim=4,
                out_dim=hdmap_dim,
            ),
            add_in_channels=hdmap_dim,
        ),
        pretrained='runwayml/stable-diffusion-v1-5',
        pipeline_name='StableDiffusionControlPipeline',
        checkpoint=ckpt_2d,
        with_ema=with_ema,
        weight_path=weight_path,
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
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='fp16',
        checkpoint_interval=1,
        checkpoint_total_limit=10,
        log_with='tensorboard',
        log_interval=100,
        activation_checkpointing=activation_checkpointing,
        resume=resume,
        with_ema=with_ema,
        # max_grad_norm=1.0,
    ),
    test=dict(
        mixed_precision='fp16',
        save_dir=os.path.join(project_dir, 'vis'),
        guidance_scale=guidance_scale,
    ),
)
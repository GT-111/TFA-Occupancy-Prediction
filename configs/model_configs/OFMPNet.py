from configs.utils.config import load_config
import os

# ============= Seed ===================
random_seed = 42
# ============= Path ===================
project_name = 'OFMPNet'  # Name of your project
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
total_data_samples = 30000
# ============= Model Parameters =================
input_dim = 3 # occupancy, flow_x, flow_y
hidden_dim = 96
num_states = 9# TODO: Define the number of states
num_heads = 4
dropout_prob=0.1
num_motion_mode=6 # number of future motion modes

shallow_decode = 1
# ============= Train Parameters =================
num_machines = 1
gpu_ids = [0,1]
max_epochs = 30
batch_size = 8
# ============= Optimizer Parameters =================
optimizer_type = 'AdamW'
optimizers_dic = dict(
    AdamW=dict(
        type='AdamW',       # AdamW optimizer
        learning_rate=3e-4,            # Base learning rate
        betas=(0.9, 0.95),  # Slightly higher β2 for smoother updates
        eps=1e-8,           # Avoids division by zero
        weight_decay=1e-6   # Encourages generalization
    ),
    NAdam=dict(
        type='NAdam',       # NAdam optimizer
        learning_rate = 1e-4,
        weight_decay = 1e-4
    )
)
assert optimizer_type in optimizers_dic, f"Optimizer type {optimizer_type} is not supported"
# ============= Scheduler Parameters =================
scheduler_type = 'CosineAnnealingWarmRestarts'
schedulers_dic = dict(
    StepLR=dict(
        type='StepLR',      # StepLR scheduler
        step_size = 3,
        gamma = 0.5
    ),
    CosineAnnealingWarmRestarts=dict(
        type='CosineAnnealingWarmRestarts',  # CosineAnnealingWarmRestarts scheduler
        T_0=2,    # First restart at 2 epochs
        T_mult=2,  # Restart period doubles (2 → 4 → 8 epochs)
        eta_min=1e-6  # Minimum LR to avoid vanishing updates
    )
)
assert scheduler_type in schedulers_dic, f"Scheduler type {scheduler_type} is not supported"


# ============= Test Parameters =================
guidance_scale = 7.5
weight_path = None  # None is the last ckpt you have trained
# ============= Config ===================
config = dict(
    project_name=project_name,
    project_dir=project_dir,
    dataset_config=dataset_config,
    dataloaders=dict(
        datasets=dict(
            train_ratio = 0.8,
            validation_ratio = 0.1,
            test_ratio = 0.1,
            data_path=generated_data_path,
            total_data_samples=total_data_samples,
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
        ofmpnet=dict(
            input_size=(occupancy_flow_map_height, occupancy_flow_map_width),
            history_length=num_his_points,  # Number of historical frames

            num_waypoints=num_waypoints,  # Number of waypoints to predict
            swin_transformer=dict(
                input_size=(occupancy_flow_map_height, occupancy_flow_map_width),
                history_length=num_his_points, 
                patch_size=[4, 4],  # Patch size for the Swin Transformer
                embedding_dimension=hidden_dim,
                transformer_depths=[2, 2, 1],  # Depths of the transformer layers
                window_size=4,
                attention_heads=[3, 6 , 12],
                mlp_ratio=4.,  # hidden size ratio
                drop_path_rate=0.0,
                # Encoder Settings
                use_flow=False,
                flow_sep=True,
                sep_encode=True,
                no_map=True,  # not using the map
                ape=True,  # Absolute Position Embedding
                patch_norm=True,
                large_input=False,
                use_checkpoint=True,

                basic_layer=dict(
                    qk_scale=None,
                    qkv_bias=True,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                ),
            ),
            fg_msa=True,
            FlowGuidedMultiHeadSelfAttention=dict(
            # input size should be the output size of the previous layer
              query_size=[6,32],
              key_value_size=[6,32],
              num_attention_heads=8,
              num_attention_head_channels=48, # attention head channels * num heads should be the same as the input dimension
              num_groups=8,
              input_dimension=4*hidden_dim, # 4 * swin_transformer.embed_dim
              output_dimension=4*hidden_dim,
              attn_drop=0.,
              proj_drop=0.,
              stride=1,
              offset_range_factor=2,
              use_positional_encoding=True,
              dwc_pe=False,
              no_offset=False,
              fixed_positional_encoding=False,
              stage_idx=3,
              use_last_ref=False,
              fg=True,
            ),

            TrajNetCrossAttention=dict(
                num_waypoints=num_waypoints,  # Number of waypoints to predict
                pic_size=[6,32],
                pic_dim=4*hidden_dim,
                sep_actors=False,
                CrossAttention=dict(
                # key dimension * num heads should be the same as the input dimension
                  key_dimension=48, # This should match the output dimension of the previous layer
                  num_heads=8,
                ),
                TrajNet=dict(
                    TrajEncoder=dict(
                        node_feature_dim=6,
                        vector_feature_dim=3,
                        att_heads=6,
                        out_dim=4*hidden_dim,
                    ),
                    no_attention=False,
                    double_net=False,
                    
                    att_heads=6,
                    out_dim=4*hidden_dim,
              ),
            ),
            Pyramid3DDecoder=dict(
              pic_dim=4*hidden_dim,
              use_pyramid=True,
              timestep_split=True,
              shallow_decode=1, # (4-len(cfg['depths'][:])),,
              flow_sep_decode=True,
              conv_cnn=False,
              stp_grad=False,
              rep_res=True,
              sep_conv=False,
              num_waypoints=num_waypoints,  # Number of waypoints to predict
            ),
        ),
    ),
    losses=dict(
        occupancy_flow_map_loss=dict(
            ogm_weight  = 1000,
            occ_weight  = 0,
            flow_weight = 1,
            flow_origin_weight = 1000,
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
    optimizer = optimizers_dic[optimizer_type],
    scheduler = schedulers_dic[scheduler_type],
    train=dict(
        max_epochs=max_epochs,
        checkpoint_interval=1,
        checkpoint_dir=os.path.join(project_dir, 'checkpoints'),
        checkpoint_total_limit=10,
        log_interval=10,
    ),
    test=dict(
        occupancy_flow_map_height=occupancy_flow_map_height,
        occupancy_flow_map_width=occupancy_flow_map_width,
    ),
    loggers=dict(
        tensorboard=dict(
            type='Tensorboard',
            log_dir=os.path.join(project_dir, 'logs'),
        ),
    ),
)
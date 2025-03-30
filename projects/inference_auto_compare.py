from multiprocessing import context
import os
import torch
import warnings
import argparse
# import model
from models.AROccFlowNet.occupancy_flow_model_auto_regressive_three_scenes import AutoRegWrapperContext
from models.AROccFlowNet.occupancy_flow_model_auto_regressive import AutoRegWrapper
# import loss and metrics
from evaluation.metrics.occupancy_flow_map_metrics import OccupancyFlowMapMetrics
# import training utils
from tqdm import tqdm
from datasets.I24Motion.utils.dataset_utils import get_dataloader
from datasets.I24Motion.utils.training_utils import parse_data, parse_outputs
# import config
from configs.utils.config import load_config
# import tensorboard
import numpy as np
from utils.file_utils import get_last_file_with_extension
warnings.filterwarnings("ignore")



def setup_models(gpu_id):
    """
    Setup the model, optimizer, scheduler, and losses.
    """
    # //[ ] The model config need to be updated if the model is changed
    base_config = load_config('configs/model_configs/AROccFlowNetAutoRegressive.py')
    context_config = load_config('configs/model_configs/AROccFlowNetAutoRegressiveThreeScenes.py')
    
    base_checkpoint_dir = get_last_file_with_extension(base_config.train.checkpoint_dir, '.pth')
    context_checkpoint_dir = get_last_file_with_extension(context_config.train.checkpoint_dir, '.pth')

    def ddp2single(checkpoint_dir):
        checkpoint = torch.load(checkpoint_dir, map_location="cpu")
        ddp_state_dict = checkpoint["model_state_dict"]  # or whatever key you used
        # If your keys have "module." prefix, strip it:
        new_state_dict = {}
        for k, v in ddp_state_dict.items():
            new_k = k.replace("module.", "")
            new_state_dict[new_k] = v
        return new_state_dict
    model_base = AutoRegWrapper(base_config.models.auto_regressive_predictor).to(gpu_id)
    model_contextual = AutoRegWrapperContext(context_config.models.auto_regressive_predictor).to(gpu_id)   
    
    model_base.load_state_dict(ddp2single(base_checkpoint_dir))
    model_contextual.load_state_dict(ddp2single(context_checkpoint_dir))

    return model_base, model_contextual
def model_training(gpu_id, world_size, config, enable_ddp=True):
    
    dataloaders_config = config.dataloaders

    _, _, test_dataloader = get_dataloader(dataloaders_config)

    
    model_base, model_contextual = setup_models(gpu_id)
    
    occupancy_flow_map_metrics = OccupancyFlowMapMetrics(gpu_id, no_warp=False)
   
    best_auc = 0

    timestep2test = [1, 3, 5, 10, 15]
    base_auc_dic = {timestep:[] for timestep in timestep2test}
    base_iou_dic = {timestep:[] for timestep in timestep2test}
    context_auc_dic = {timestep:[] for timestep in timestep2test}
    context_iou_dic = {timestep:[] for timestep in timestep2test}

    with torch.no_grad():
        loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        
        for batch_idx, data in loop:
            input_dict, ground_truth_dict = parse_data(data, gpu_id, config)
            # //[ ]Currently, only the current scene is being used
            prv_occupancy_map = input_dict['prv']['his/observed_occupancy_map']
            cur_occupancy_map = input_dict['cur']['his/observed_occupancy_map']
            nxt_occupancy_map = input_dict['nxt']['his/observed_occupancy_map']
            
            # get the ground truth
            gt_prv_occupancy_map = ground_truth_dict['prv']['pred/observed_occupancy_map']
            gt_nxt_occupancy_map = ground_truth_dict['nxt']['pred/observed_occupancy_map']
            gt_observed_occupancy_logits = ground_truth_dict['cur']['pred/observed_occupancy_map']
            gt_valid_mask = ground_truth_dict['cur']['pred/valid_mask']
            gt_occupancy_flow_map_mask = (torch.sum(gt_valid_mask, dim=-2) > 0)
            contextual_pred_observed_occupancy_logits = model_contextual.forward(prv_occupancy_map=prv_occupancy_map,
                                                       cur_occupancy_map=cur_occupancy_map,
                                                       nxt_occupancy_map=nxt_occupancy_map,
                                                       gt_prv_occupancy_map=gt_prv_occupancy_map,
                                                       gt_nxt__occupancy_map=gt_nxt_occupancy_map,
                                                       training=True)
            base_pred_observed_occupancy_logits = model_base.forward(cur_occupancy_map, gt_observed_occupancy_logits, training=True)

            contextual_pred_observed_occupancy_logits = torch.sigmoid(contextual_pred_observed_occupancy_logits)
            base_pred_observed_occupancy_logits = torch.sigmoid(base_pred_observed_occupancy_logits)
            base_occupancy_flow_map_metrics_dict = occupancy_flow_map_metrics.compute_occupancy_metrics(base_pred_observed_occupancy_logits, gt_observed_occupancy_logits, gt_occupancy_flow_map_mask, mean=False)
            contextual_occupancy_flow_map_metrics_dict = occupancy_flow_map_metrics.compute_occupancy_metrics(contextual_pred_observed_occupancy_logits, gt_observed_occupancy_logits, gt_occupancy_flow_map_mask, mean=False)
            # def get_k_step_mean(metrics, k):
            #     return {key: np.mean([metrics[key][step].cpu() for step in range(k)]) for key in metrics.keys()}
            print('base', torch.sum(base_pred_observed_occupancy_logits[..., 9, :]))
            print('context', torch.sum(contextual_pred_observed_occupancy_logits[..., 9, :]))
            print('gt', torch.sum(gt_observed_occupancy_logits[..., 9, :]))
            # for timestep in timestep2test:
                
            #     base_auc_dic[timestep].append(get_k_step_mean(base_occupancy_flow_map_metrics_dict, timestep)['vehicles_observed_occupancy_auc'])

            #     base_iou_dic[timestep].append(get_k_step_mean(base_occupancy_flow_map_metrics_dict, timestep)['vehicles_observed_occupancy_iou'])
            #     context_auc_dic[timestep].append(get_k_step_mean(contextual_occupancy_flow_map_metrics_dict, timestep)['vehicles_observed_occupancy_auc'])
            #     context_iou_dic[timestep].append(get_k_step_mean(contextual_occupancy_flow_map_metrics_dict, timestep)['vehicles_observed_occupancy_iou'])
            
            # visualize(gt_observed_occupancy_logits[0].cpu(), contextual_pred_observed_occupancy_logits[0].cpu(),base_pred_observed_occupancy_logits[0].cpu(), f'{batch_idx}')

                
        
        # print mean auc and iou
        for timestep in timestep2test:
            print(f'base_auc_{timestep}: {np.mean(base_auc_dic[timestep])}')
            print(f'base_iou_{timestep}: {np.mean(base_iou_dic[timestep])}')
            print(f'context_auc_{timestep}: {np.mean(context_auc_dic[timestep])}')
            print(f'context_iou_{timestep}: {np.mean(context_iou_dic[timestep])}')
                
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def visualize(ground_truth, prediction1, prediction2, id):
    H, W, T, _ = ground_truth.shape

    # Only keep left half
    ground_truth = ground_truth[:, :W // 2, :, :]
    prediction1 = prediction1[:, :W // 2, :, :]
    prediction2 = prediction2[:, :W // 2, :, :]
    W = W // 2

    x = np.arange(W)
    y = np.arange(H)
    selected_timesteps = [1, 2, 3, 10, 15]
    num_cols = len(selected_timesteps)

    # Color map
    low_focus_colormap = [
        [0.00, "white"],
        [0.10, "#fee8c8"],
        [0.30, "#fdbb84"],
        [0.50, "#fc8d59"],
        [0.70, "#ef6548"],
        [0.85, "#d7301f"],
        [1.00, "#990000"]
    ]

    # Shared color scale
    combined = np.concatenate([ground_truth, prediction1, prediction2], axis=-1)
    zmin, zmax = 0, 1

    # Create 3-row subplot: GT, Pred1, Pred2
    fig = make_subplots(
        rows=3, cols=num_cols,
        column_titles=[f"Timestep {t}" for t in selected_timesteps],
        row_titles=["", "", ""],
        vertical_spacing=0.05,
        horizontal_spacing=0.005
    )

    axis_off = dict(showticklabels=False, ticks="", title="", showgrid=False, zeroline=False)

    # === Add Ground Truth (Row 1) ===
    for col, t in enumerate(selected_timesteps):
        z = ground_truth[:, :, t, 0]
        fig.add_trace(
            go.Contour(
                z=z,
                x=x, y=y,
                zmin=zmin, zmax=zmax,
                colorscale=low_focus_colormap,
                contours=dict(showlines=False),
                colorbar=dict(
                    title="", tickvals=[], ticktext=[], len=0.5
                ) if col == num_cols - 1 else None,
            ),
            row=1, col=col + 1
        )

    # === Add Prediction 1 (Row 2) ===
    for col, t in enumerate(selected_timesteps):
        z = prediction1[:, :, t, 0]
        fig.add_trace(
            go.Contour(
                z=z,
                x=x, y=y,
                zmin=zmin, zmax=zmax,
                colorscale=low_focus_colormap,
                contours=dict(showlines=False),
                colorbar=dict(
                    title="", tickvals=[], ticktext=[], len=0.5
                ) if col == num_cols - 1 else None,
            ),
            row=2, col=col + 1
        )

    # === Add Prediction 2 (Row 3) ===
    for col, t in enumerate(selected_timesteps):
        z = prediction2[:, :, t, 0]
        fig.add_trace(
            go.Contour(
                z=z,
                x=x, y=y,
                zmin=zmin, zmax=zmax,
                colorscale=low_focus_colormap,
                contours=dict(showlines=False),
                colorbar=dict(
                    title="", tickvals=[], ticktext=[], len=0.5
                ) if col == num_cols - 1 else None,
            ),
            row=3, col=col + 1
        )

    # Draw black borders
    x0_data, x1_data = x[0], x[-1]
    y0_data, y1_data = y[0], y[-1]
    for r in range(3):
        for c in range(num_cols):
            subplot_idx = r * num_cols + c + 1
            fig.add_shape(
                type="rect",
                x0=x0_data, x1=x1_data,
                y0=y0_data, y1=y1_data,
                xref=f"x{subplot_idx}",
                yref=f"y{subplot_idx}",
                line=dict(color="black", width=1),
                layer="above"
            )

    # Layout
    fig.update_layout(
        height=H * 6,  # More height for 3 rows
        width=W * num_cols * 2.2,
        margin=dict(l=5, r=5, t=50, b=5),
        showlegend=False
    )

    for r in [1, 2, 3]:
        for c in range(1, num_cols + 1):
            fig.update_xaxes(axis_off, row=r, col=c)
            fig.update_yaxes(axis_off, row=r, col=c)

    fig.write_image(f"gt_vs_two_preds_contour_{id}.png", format="png")


if __name__ == "__main__":
    # ============= Parse Argument =============
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("--config", type=str, default="configs/model_configs/AROccFlowNetAutoRegressive.py", help="config file")
    args = parser.parse_args()
    # ============= Load Configuration =============
    config = load_config(args.config)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    world_size = torch.cuda.device_count()
    # mp.spawn(model_training, args=(world_size, config), nprocs=world_size)
    model_training(0, world_size, config)

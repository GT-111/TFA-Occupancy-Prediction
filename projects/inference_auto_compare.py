from multiprocessing import context
import os
from pickle import FALSE
from pyexpat import model
import torch
import warnings
import argparse
# import model
from models.AROccFlowNet.occupancy_flow_model_auto_regressive_three_scenes import AutoRegWrapperContext
from models.AROccFlowNet.occupancy_flow_model_auto_regressive import AutoRegWrapper
from models.OFMPNet.OFMPNet import OFMPNet
# import loss and metrics
from evaluation.metrics.occupancy_flow_map_metrics import OccupancyFlowMapMetrics
# import training utils
from tqdm import tqdm
from datasets.I24Motion.utils.dataset_utils import get_dataloader
from datasets.I24Motion.utils.training_utils import parse_data, parse_outputs_OFMPNet
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
    tfa_config = load_config('configs/model_configs/AROccFlowNetAutoRegressiveThreeScenes.py')
    ofmpnet_config = load_config('configs/model_configs/OFMPNet.py')
    base_checkpoint_dir = get_last_file_with_extension(base_config.train.checkpoint_dir, '.pth')
    tfa_checkpoint_dir = get_last_file_with_extension(tfa_config.train.checkpoint_dir, '.pth')
    ofmpnet_checkpoint_dir = get_last_file_with_extension(ofmpnet_config.train.checkpoint_dir, '.pth')
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
    model_tfa = AutoRegWrapperContext(tfa_config.models.auto_regressive_predictor).to(gpu_id)   
    model_ofmpnet = OFMPNet(ofmpnet_config.models.ofmpnet).to(gpu_id)

    model_base.load_state_dict(ddp2single(base_checkpoint_dir))
    model_tfa.load_state_dict(ddp2single(tfa_checkpoint_dir))
    model_ofmpnet.load_state_dict(ddp2single(ofmpnet_checkpoint_dir))

    return model_base, model_tfa, model_ofmpnet
def model_training(gpu_id, world_size, config, enable_ddp=True):
    
    dataloaders_config = config.dataloaders

    _, _, test_dataloader = get_dataloader(dataloaders_config)

    
    model_base, model_tfa, model_ofmpnet = setup_models(gpu_id)
    
    occupancy_flow_map_metrics = OccupancyFlowMapMetrics(gpu_id, no_warp=False)
   
    best_auc = 0

    timestep2test = [1, 3, 5, 10, 15]
    base_auc_dic = {timestep:[] for timestep in timestep2test}
    base_iou_dic = {timestep:[] for timestep in timestep2test}
    tfa_auc_dic = {timestep:[] for timestep in timestep2test}
    tfa_iou_dic = {timestep:[] for timestep in timestep2test}
    ofmpnet_auc_dic = {timestep:[] for timestep in timestep2test}
    ofmpnet_iou_dic = {timestep:[] for timestep in timestep2test}
    with torch.no_grad():
        loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        
        for batch_idx, data in loop:
            input_dict, ground_truth_dict = parse_data(data, gpu_id, config)
            # //[ ]Currently, only the current scene is being used
            prv_occupancy_map = input_dict['prv']['his/observed_occupancy_map']
            cur_occupancy_map = input_dict['cur']['his/observed_occupancy_map']
            nxt_occupancy_map = input_dict['nxt']['his/observed_occupancy_map']
            flow_origin_occupancy = input_dict['flow_origin_occupancy_map']
            cur_flow_map = input_dict['cur']['his/flow_map'][:, : ,:, -1, :]  # only use the last time step for flow map
            cur_observed_agent_features = input_dict['cur']['his/observed_agent_features']
            # get the ground truth
            gt_prv_occupancy_map = ground_truth_dict['prv']['pred/observed_occupancy_map']
            gt_nxt_occupancy_map = ground_truth_dict['nxt']['pred/observed_occupancy_map']
            gt_observed_occupancy_logits = ground_truth_dict['cur']['pred/observed_occupancy_map']
            gt_occluded_occupancy_logits = ground_truth_dict['cur']['pred/occluded_occupancy_map']
            gt_valid_mask = ground_truth_dict['cur']['pred/valid_mask']
            gt_occupancy_flow_map_mask = (torch.sum(gt_valid_mask, dim=-2) > 0)

            tfa_pred_observed_occupancy_logits = model_tfa.forward(prv_occupancy_map=prv_occupancy_map,
                                                       cur_occupancy_map=cur_occupancy_map,
                                                       nxt_occupancy_map=nxt_occupancy_map,
                                                       gt_prv_occupancy_map=gt_prv_occupancy_map,
                                                       gt_nxt__occupancy_map=gt_nxt_occupancy_map,
                                                       training=True)
            base_pred_observed_occupancy_logits = model_base.forward(cur_occupancy_map, gt_observed_occupancy_logits, training=True)
            ofmpnetpred_observed_occupancy_logits, _, _ = parse_outputs_OFMPNet(model_ofmpnet.forward(cur_occupancy_map, cur_flow_map, None, cur_observed_agent_features, cur_observed_agent_features))
            
            tfa_pred_observed_occupancy_logits = torch.sigmoid(tfa_pred_observed_occupancy_logits)
            base_pred_observed_occupancy_logits = torch.sigmoid(base_pred_observed_occupancy_logits)
            ofmpnet_pred_observed_occupancy_logits = torch.sigmoid(ofmpnetpred_observed_occupancy_logits)

            base_occupancy_flow_map_metrics_dict = occupancy_flow_map_metrics.compute_occupancy_metrics(base_pred_observed_occupancy_logits, gt_observed_occupancy_logits, gt_occupancy_flow_map_mask, mean=False)
            tfa_occupancy_flow_map_metrics_dict = occupancy_flow_map_metrics.compute_occupancy_metrics(tfa_pred_observed_occupancy_logits, gt_observed_occupancy_logits, gt_occupancy_flow_map_mask, mean=False)
            ofmpnet_occupancy_flow_map_metrics_dict = occupancy_flow_map_metrics.compute_occupancy_metrics(ofmpnet_pred_observed_occupancy_logits, gt_observed_occupancy_logits, gt_occupancy_flow_map_mask, mean=False)
            
            def get_k_step_mean(metrics, k):
                return {key: np.mean([metrics[key][step].cpu() for step in range(k)]) for key in metrics.keys()}
            for timestep in timestep2test:
                
                base_auc_dic[timestep].append(get_k_step_mean(base_occupancy_flow_map_metrics_dict, timestep)['vehicles_observed_occupancy_auc'])
                base_iou_dic[timestep].append(get_k_step_mean(base_occupancy_flow_map_metrics_dict, timestep)['vehicles_observed_occupancy_iou'])
                tfa_auc_dic[timestep].append(get_k_step_mean(tfa_occupancy_flow_map_metrics_dict, timestep)['vehicles_observed_occupancy_auc'])
                tfa_iou_dic[timestep].append(get_k_step_mean(tfa_occupancy_flow_map_metrics_dict, timestep)['vehicles_observed_occupancy_iou'])
                ofmpnet_auc_dic[timestep].append(get_k_step_mean(ofmpnet_occupancy_flow_map_metrics_dict, timestep)['vehicles_observed_occupancy_auc'])
                ofmpnet_iou_dic[timestep].append(get_k_step_mean(ofmpnet_occupancy_flow_map_metrics_dict, timestep)['vehicles_observed_occupancy_iou'])
            
            visualize(gt_observed_occupancy_logits[0].cpu(),
                      {'TFA':tfa_pred_observed_occupancy_logits[0].cpu(),
                       'Base':base_pred_observed_occupancy_logits[0].cpu(),
                       'OFMPNet':ofmpnet_pred_observed_occupancy_logits[0].cpu()
                       },
                      f'{batch_idx}')

                
        
        # print mean auc and iou
        for timestep in timestep2test:
            print(f'base_auc_{timestep}: {np.mean(base_auc_dic[timestep])}')
            print(f'base_iou_{timestep}: {np.mean(base_iou_dic[timestep])}')
            print(f'context_auc_{timestep}: {np.mean(tfa_auc_dic[timestep])}')
            print(f'context_iou_{timestep}: {np.mean(tfa_iou_dic[timestep])}')
            print(f'ofmpnet_auc_{timestep}: {np.mean(ofmpnet_auc_dic[timestep])}')
            print(f'ofmpnet_iou_{timestep}: {np.mean(ofmpnet_iou_dic[timestep])}')
                
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def visualize(ground_truth, predictions_dic, id):
    

    H, W, T, _ = ground_truth.shape

    # Only keep left half
    if True:
        ground_truth = ground_truth[:, :W // 2, :, :]
        predictions = [pred[:, :W // 2, :, :] for pred in predictions_dic.values()]
        W = W // 2
    else:
        ground_truth = ground_truth[:, :, :, :]
        predictions = [pred[:,:, :, :] for pred in predictions_dic.values()]
        

    x = np.arange(W)
    y = np.arange(H)
    selected_timesteps = [1, 2, 3, 10, 15]
    num_cols = len(selected_timesteps)
    num_rows = 1 + len(predictions)  # GT row + prediction rows

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
    combined = np.concatenate([ground_truth] + predictions, axis=-1)
    zmin, zmax = 0, 1

    # Create subplot
    fig = make_subplots(
        rows=num_rows, cols=num_cols,
        column_titles=[f"Timestep {t}" for t in selected_timesteps],
        row_titles=["GT"] + [k for k in predictions_dic.keys()],
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

    # === Add Predictions (Rows 2+) ===
    for row_idx, prediction in enumerate(predictions):
        for col, t in enumerate(selected_timesteps):
            z = prediction[:, :, t, 0]
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
                row=row_idx + 2, col=col + 1
            )

    # Draw black borders
    x0_data, x1_data = x[0], x[-1]
    y0_data, y1_data = y[0], y[-1]
    for r in range(num_rows):
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
        height=H * num_rows * 2,
        width=W * num_cols * 2.2,
        margin=dict(l=5, r=5, t=50, b=5),
        showlegend=False
    )

    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            fig.update_xaxes(axis_off, row=r, col=c)
            fig.update_yaxes(axis_off, row=r, col=c)

    fig.write_image(f"gt_vs_preds_contour_{id}.png", format="png")



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

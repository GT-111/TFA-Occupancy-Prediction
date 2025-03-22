import torch
import numpy as np
import matplotlib.pyplot as plt
from .visualize_utils import get_cmap
from utils.occupancy_flow_map_utils import sample

def validate_flow_wrap_occupancy(test_data, occupancy_flow_map_config, k):
    occupancy_flow_map_height = occupancy_flow_map_config.occupancy_flow_map_height
    occupancy_flow_map_width = occupancy_flow_map_config.occupancy_flow_map_width
    h = torch.arange(0, occupancy_flow_map_height, dtype=torch.float32)
    w = torch.arange(0, occupancy_flow_map_width, dtype=torch.float32)

    h_idx, w_idx = torch.meshgrid(h, w, indexing="xy")
    # These indices map each (x, y) location to (x, y).
    # [height, width, 2] but storing x, y coordinates.
    flow_origin_occupancy = torch.from_numpy(test_data['flow_origin_occupancy_map'])[...,k, :].squeeze()
    pred_occupancy_map = torch.from_numpy(test_data['pred/observed_occupancy_map'])[...,k, :].squeeze()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    colors_obs = [
        (1, 1, 1),  # White for 0
        (1, 0.8, 0.8),  # Light red for mid-range
        (1, 0, 1)  # Red for 1
    ]
    cmap_obs = get_cmap(colors_obs)
    axes[0, 0].imshow(flow_origin_occupancy, cmap=cmap_obs, interpolation='nearest')
    axes[0, 0].title.set_text('Flow Origin Occupancy')
    # add grid
    axes[0, 0].grid(True, which='both', color='black', linewidth=0.5)
    pred_flow = torch.from_numpy(test_data['pred/flow_map'])
    identity_indices = torch.stack(
        (
            w_idx.T,
            h_idx.T,
        ),dim=-1)
    warped_indices = identity_indices + pred_flow[...,k,:][None, ...]
    wp_origin = sample(
        image=flow_origin_occupancy[None, ..., None],
        warp=warped_indices,
        pixel_type=0,
    )
    axes[0, 1].imshow(np.array(pred_occupancy_map), cmap=cmap_obs, interpolation='nearest', )
    axes[0, 1].title.set_text('Predicted Occupancy')
    axes[0, 1].grid(True, which='both', color='black', linewidth=0.5)
    
    # axes[1, 0].quiver(pred_flow[...,k,:][..., 0], pred_flow[...,k,:][..., 1], color='black', alpha=0.5)
    axes[1, 0].imshow(np.array(wp_origin)[0,...,0], cmap=cmap_obs, interpolation='nearest', )
    axes[1, 0].title.set_text('Warped Occupancy')
    axes[1, 0].grid(True, which='both', color='black', linewidth=0.5)
    
    axes[1, 1].imshow(np.array(wp_origin)[0,...,0] * np.array(pred_occupancy_map), cmap=cmap_obs, interpolation='nearest', )
    axes[1, 1].title.set_text('Warped Occupancy * Predicted Occupancy')
    axes[1, 1].grid(True, which='both', color='black', linewidth=0.5)
    plt.savefig('test.png')
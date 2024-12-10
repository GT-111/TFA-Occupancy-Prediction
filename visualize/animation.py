from utils.file_utils import get_config
from utils.occ_flow_utils import GridMap
from dataset.I24Dataset import I24Dataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from utils.dataset_utils import get_road_map
import flow_vis
import numpy as np
import typing
def process_data_for_visualization(config, data_dic, ground_truth=True, pred_dict=None, valid_dict=None):
    grid_size_x = config.occupancy_flow_map.grid_size.x
    grid_size_y = config.occupancy_flow_map.grid_size.y
    history_data_dic = typing.DefaultDict(dict)
    future_data_dic = typing.DefaultDict(dict)
    # Flow map (T, W, H, 2)
    # Occupancy map (T, W, H)
    if 'prv' in valid_dict:
        history_data_dic['prv']['occupancy_map'] = data_dic['prv/state/his/observed_occupancy_map'].swapaxes(-2, -1).astype(np.float32)
        history_data_dic['prv']['flow_map'] = data_dic['prv/state/his/flow_map'].swapaxes(-2, -3).astype(np.float32)
        history_data_dic['prv']['flow_map_rendered'] = np.sum(history_data_dic['prv']['flow_map'], axis=0)
    
    if 'cur' in valid_dict:
        history_data_dic['cur']['occupancy_map'] = data_dic['cur/state/his/observed_occupancy_map'].swapaxes(-2, -1).astype(np.float32)
        history_data_dic['cur']['flow_map'] = data_dic['cur/state/his/flow_map'].swapaxes(-2, -3).astype(np.float32)
        history_data_dic['cur']['flow_map_rendered'] = np.sum(history_data_dic['cur']['flow_map'], axis=0)
        
    if 'nxt' in valid_dict:
        history_data_dic['nxt']['occupancy_map'] = data_dic['nxt/state/his/observed_occupancy_map'].swapaxes(-2, -1).astype(np.float32)
        history_data_dic['nxt']['flow_map'] = data_dic['nxt/state/his/flow_map'].swapaxes(-2, -3).astype(np.float32)
        history_data_dic['nxt']['flow_map_rendered'] = np.sum(history_data_dic['nxt']['flow_map'], axis=0)
    if ground_truth:
        if 'prv' in valid_dict:
            future_data_dic['prv']['occupancy_map'] = data_dic['prv/state/pred/observed_occupancy_map'].swapaxes(-2, -1).astype(np.float32)
            future_data_dic['prv']['flow_map'] = data_dic['prv/state/pred/flow_map'].swapaxes(-2, -3).astype(np.float32)
            future_data_dic['prv']['flow_map_rendered'] = np.sum(future_data_dic['prv']['flow_map'], axis=0)
        if 'cur' in valid_dict:
            future_data_dic['cur']['occupancy_map'] = data_dic['cur/state/pred/observed_occupancy_map'].swapaxes(-2, -1).astype(np.float32)
            future_data_dic['cur']['flow_map'] = data_dic['cur/state/pred/flow_map'].swapaxes(-2, -3).astype(np.float32)
            future_data_dic['cur']['flow_map_rendered'] = np.sum(future_data_dic['cur']['flow_map'], axis=0)
        if 'nxt' in valid_dict:
            future_data_dic['nxt']['occupancy_map'] = data_dic['nxt/state/pred/observed_occupancy_map'].swapaxes(-2, -1).astype(np.float32)
            future_data_dic['nxt']['flow_map'] = data_dic['nxt/state/pred/flow_map'].swapaxes(-2, -3).astype(np.float32)
            future_data_dic['nxt']['flow_map_rendered'] = np.sum(future_data_dic['nxt']['flow_map'], axis=0)
    else:
        if 'cur' in valid_dict:
            future_data_dic['cur']['occupancy_map'] = pred_dict['observed_occupancy_map'].swapaxes(-2, -1).astype(np.float32)
            future_data_dic['cur']['flow_map'] = pred_dict['flow_map'].swapaxes(-2, -3).astype(np.float32)
            future_data_dic['cur']['flow_map_rendered'] = np.sum(future_data_dic['cur']['flow_map'], axis=0)



    x = np.arange(0, grid_size_x)  # X-axis points (256)
    y = np.arange(0, grid_size_y)  # Y-axis points (128)
    X, Y = np.meshgrid(x, y)



    
    return X, Y, history_data_dic, future_data_dic


def set_white_to_transparent(image):
    """
    Converts white pixels in a normalized single-channel grayscale image (values between 0 and 1) 
    to transparent in RGBA format.

    Parameters:
    - image: NumPy array of shape (H, W) with pixel values in the range [0, 1].

    Returns:
    - rgba_image: NumPy array of shape (H, W, 4) with white pixels transparent.
    """
    # Ensure the input is a 2D grayscale image with normalized values
    if len(image.shape) != 2 or not np.all((image >= 0) & (image <= 1)):
        raise ValueError("Input must be a 2D single-channel grayscale image with values in the range [0, 1].")

    # Convert grayscale to RGB by stacking the grayscale values along 3 channels
    rgb_image = np.stack([image] * 3, axis=-1)

    # Convert to RGBA by adding an alpha channel initialized to 1 (fully opaque)
    rgba_image = np.dstack([rgb_image, np.ones_like(image)])

    # Identify white pixels (value == 1) and set their alpha to 0 (fully transparent)
    white_pixels = (image == 1)
    rgba_image[white_pixels, 3] = 0

    return rgba_image

def downsample(x, y, flow_map, down_sample_x=1, down_sample_y=1):
    """Downsample x, y, and flow_map with different rates for x and y axes."""
    x_down = x[::down_sample_y, ::down_sample_x]
    y_down = y[::down_sample_y, ::down_sample_x]
    flow_map_down = flow_map[::down_sample_y, ::down_sample_x, :]
    return x_down, y_down, flow_map_down

def initialize_quiver(ax, flow_map, x, y):
    """Initialize the quiver plot with downsampled data."""
    x, y, flow_map = downsample(x, y, flow_map)
    mask = (flow_map[..., 0] != 0) | (flow_map[..., 1] != 0)
    X_masked, Y_masked = x[mask], y[mask]
    U_masked, V_masked = flow_map[..., 0][mask], flow_map[..., 1][mask]
    return ax.quiver(X_masked, Y_masked, U_masked, V_masked, 
                     angles='xy', scale_units='xy', scale=0.25, width=0.005, color='black', alpha=0.5)


def get_cmap(colors):
    
    return LinearSegmentedColormap.from_list('while-colors', colors, N=256)

def visualize(config, data_dic, name, vis_occ=True, vis_flow=True, vis_optical_flow=True, valid_dict=None, pred_dict=None, ground_truth=True):
    X, Y, history_data_dic, future_data_dic = process_data_for_visualization(config, data_dic, valid_dict=valid_dict, pred_dict=pred_dict, ground_truth=ground_truth)

    road_map = get_road_map(config, batch_first=False).swapaxes(0,1)
    num_history_time_steps = history_data_dic['cur']['occupancy_map'].shape[0]
    num_future_time_steps = future_data_dic['cur']['occupancy_map'].shape[0]
    # Define the custom colormap: white from 0 to 0.5, then gradually red from 0.5 to 1
    colors_occ = [
        (1, 1, 1),  # White for 0
        (1, 0.8, 0.8),  # Light red for mid-range
        (0, 1, 0)  # Red for 1
    ]
    cmap_occ = get_cmap(colors_occ)
    
    colors_obs = [
        (1, 1, 1),  # White for 0
        (1, 0.8, 0.8),  # Light red for mid-range
        (1, 0, 1)  # Red for 1
    ]
    cmap_obs = get_cmap(colors_obs)
    

    # Initialize the plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    # # Create the meshgrid for the quiver plot
    # blue_square = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=15, label='Blue Square')
    # red_square = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=15, label='Red Square')
    # plt.legend(handles=[blue_square, red_square], loc='best')
    for ax in axes:
        ax.axis('off')

    # axes[0].set_title('Previous Scene')
    axes[1].set_title('Current Scene')
    # axes[2].set_title('Next Scene')
    if vis_optical_flow:
        if 'prv' in valid_dict:
            flow_rendered_prv = axes[0].imshow(flow_vis.flow_to_color(history_data_dic['prv']['flow_map_rendered']),interpolation='nearest', alpha=1)
        if 'cur' in valid_dict:  
            flow_rendered_cur = axes[1].imshow(flow_vis.flow_to_color(history_data_dic['cur']['flow_map_rendered']),interpolation='nearest', alpha=1)
        if 'nxt' in valid_dict:  
            flow_rendered_nex = axes[2].imshow(flow_vis.flow_to_color(history_data_dic['nxt']['flow_map_rendered']),interpolation='nearest', alpha=1)
    # Initialize the occupancy map with the first frame's data
    if vis_occ:
        
        # img_prv_occ_ogm = axes[0].imshow(prv_scene_occ_ogm[0], cmap=cmap_occ, interpolation='nearest', alpha=0.5)
        # img_cur_occ_ogm = axes[1].imshow(cur_scene_occ_ogm[0], cmap=cmap_occ, interpolation='nearest', alpha=0.5)
        # img_nxt_occ_ogm = axes[2].imshow(nxt_scene_occ_ogm[0], cmap=cmap_occ, interpolation='nearest', alpha=0.5)
        if 'prv' in valid_dict:  
            img_prv_obs_ogm = axes[0].imshow(history_data_dic['prv']['occupancy_map'][0], cmap=cmap_obs, interpolation='nearest', alpha=0.5)
        if 'cur' in valid_dict:  
            img_cur_obs_ogm = axes[1].imshow(history_data_dic['cur']['occupancy_map'][0], cmap=cmap_obs, interpolation='nearest', alpha=0.5)
        if 'nxt' in valid_dict:
            img_nxt_obs_ogm = axes[2].imshow(history_data_dic['nxt']['occupancy_map'][0], cmap=cmap_obs, interpolation='nearest', alpha=0.5)
    if vis_flow:
        if 'prv' in valid_dict:  
            quiver_prv = initialize_quiver(axes[0], history_data_dic['prv']['flow_map'][0], X, Y)
        if 'cur' in valid_dict:  
            quiver_cur = initialize_quiver(axes[1], history_data_dic['cur']['flow_map'][0], X, Y)
        if 'nxt' in valid_dict:  
            quiver_nxt = initialize_quiver(axes[2], history_data_dic['nxt']['flow_map'][0], X, Y)
    
    
    # Animation update function
    def update(frame):
        """Update both the quiver plot and the occupancy map for each frame."""
        # Update occupancy map
        
        for ax in axes:
            ax.clear()
            ax.axis('off')
        # axes[0].set_title('Previous Scene')
        axes[1].set_title('Current Scene')
        # axes[2].set_title('Next Scene')
        axes[0].scatter(X[::8, ::8], Y[::8, ::8], color='black', s=1, alpha=0.5)
        axes[1].scatter(X[::8, ::8], Y[::8, ::8], color='black', s=1, alpha=0.5)
        axes[2].scatter(X[::8, ::8], Y[::8, ::8], color='black', s=1, alpha=0.5)   
        if frame <= num_history_time_steps - 1:
            data_dic = history_data_dic
        else:
            data_dic = future_data_dic
            frame = frame - num_history_time_steps
            # set the title for the future scene
            # set the big title for the future scene
            fig.suptitle('Future Scene', fontsize=16)
            axes[0].set_title('Previous Scene')
            axes[1].set_title('Current Scene')
            axes[2].set_title('Next Scene')
        
        img_road_map = axes[1].imshow(road_map)
        
        
        if vis_optical_flow:
            if 'prv' in valid_dict: 
                flow_rendered_prv = axes[0].imshow(flow_vis.flow_to_color(data_dic['prv']['flow_map_rendered']),interpolation='nearest', alpha=1)
            if 'cur' in valid_dict: 
                flow_rendered_cur = axes[1].imshow(flow_vis.flow_to_color(data_dic['cur']['flow_map_rendered']),interpolation='nearest', alpha=1)
            if 'nxt' in valid_dict: 
                flow_rendered_nxt = axes[2].imshow(flow_vis.flow_to_color(data_dic['nxt']['flow_map_rendered']),interpolation='nearest', alpha=1)
            
        # Initialize the occupancy map with the first frame's data
        if vis_occ:
            
            # img_prv_occ_ogm = axes[0].imshow(prv_scene_occ_ogm[frame], cmap=cmap_occ, interpolation='nearest', alpha=0.5)
            # img_cur_occ_ogm = axes[1].imshow(cur_scene_occ_ogm[frame], cmap=cmap_occ, interpolation='nearest', alpha=0.5)
            # img_nxt_occ_ogm = axes[2].imshow(nxt_scene_occ_ogm[frame], cmap=cmap_occ, interpolation='nearest', alpha=0.5)
            if 'prv' in valid_dict: 
                img_prv_obs_ogm = axes[0].imshow(data_dic['prv']['occupancy_map'][frame], cmap=cmap_obs, interpolation='nearest', alpha=0.5)
            if 'cur' in valid_dict: 
                img_cur_obs_ogm = axes[1].imshow(data_dic['cur']['occupancy_map'][frame], cmap=cmap_obs, interpolation='nearest', alpha=0.5)
            if 'nxt' in valid_dict: 
                img_nxt_obs_ogm = axes[2].imshow(data_dic['nxt']['occupancy_map'][frame], cmap=cmap_obs, interpolation='nearest', alpha=0.5)
            

        if vis_flow:
            if 'prv' in valid_dict: 
                quiver_prv = initialize_quiver(axes[0], data_dic['prv']['flow_map'][frame], X, Y)
            if 'cur' in valid_dict: 
                quiver_cur = initialize_quiver(axes[1], data_dic['cur']['flow_map'][frame], X, Y)
            if 'nxt' in valid_dict: 
                quiver_nxt = initialize_quiver(axes[2], data_dic['nxt']['flow_map'][frame], X, Y)
        
        return_list = [img_road_map]
        if vis_flow:
            if 'prv' in valid_dict: 
                return_list.extend([quiver_prv])
            if 'cur' in valid_dict: 
                return_list.extend([quiver_cur])
            if 'nxt' in valid_dict: 
                return_list.extend([quiver_nxt])
        if vis_optical_flow:
            # return_list.extend([flow_rendered_prv, flow_rendered_cur, flow_rendered_nex])
            if 'prv' in valid_dict: 
                return_list.extend([flow_rendered_prv])
            if 'cur' in valid_dict:
                return_list.extend([flow_rendered_cur])
            if 'nxt' in valid_dict:
                return_list.extend([flow_rendered_nxt])
        if vis_occ:
            # return_list.extend([img_prv_obs_ogm, img_cur_obs_ogm, img_nxt_obs_ogm])
            if 'prv' in valid_dict: 
                return_list.extend([img_prv_obs_ogm])
            if 'cur' in valid_dict:
                return_list.extend([img_cur_obs_ogm])
            if 'nxt' in valid_dict:
                return_list.extend([img_nxt_obs_ogm])
        return return_list

    # Create the animation

    ani = animation.FuncAnimation(fig, update, frames=num_history_time_steps + num_future_time_steps, interval=200, blit=True)

    # Save the animation without padding or extra margins
    if vis_occ:
        name = name + '_occ'
    if vis_flow:
        name = name + '_flow'
    if vis_optical_flow:
        name = name + '_optical_flow'
    ani.save(name + '.mp4', writer='ffmpeg', fps=2, dpi=300)

from utils.metrics_utils import sample
import torch
if __name__ == '__main__':
    config = get_config('config_12.yaml')
    gridmap = GridMap(config)
    dataset = I24Dataset(config)
    k = 3
    name = "scene_1014"
    test_data = np.load(config.paths.processed_data + '/' + name + '.npy', allow_pickle=True).item()
    # visualize(config, test_data, name, vis_occ=True, vis_flow=True, vis_optical_flow=False, valid_dict={'cur': 1})
    h = torch.arange(0, config.occupancy_flow_map.grid_size.y, dtype=torch.float32)
    w = torch.arange(0, config.occupancy_flow_map.grid_size.x, dtype=torch.float32)

    h_idx, w_idx = torch.meshgrid(h, w, indexing="xy")
    # These indices map each (x, y) location to (x, y).
    # [height, width, 2] but storing x, y coordinates.
    flow_origin_occupancy = torch.from_numpy(test_data['observed_occupancy_map'])[...,-config.task_config.num_waypoints:][...,k-1] + torch.from_numpy(test_data['occluded_occupancy_map'])[...,-config.task_config.num_waypoints:][...,k-1]
    pred_occupancy_map = torch.from_numpy(test_data['observed_occupancy_map'])[...,-config.task_config.num_waypoints:][...,k]
    print(pred_occupancy_map.shape)
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    colors_obs = [
        (1, 1, 1),  # White for 0
        (1, 0.8, 0.8),  # Light red for mid-range
        (1, 0, 1)  # Red for 1
    ]
    cmap_obs = get_cmap(colors_obs)
    axes[0].imshow(flow_origin_occupancy, cmap=cmap_obs, interpolation='nearest')
    print(flow_origin_occupancy.shape)
    pred_flow = torch.from_numpy(test_data['flow_map'])[...,-config.task_config.num_waypoints:,:]
    print(pred_flow.shape)
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
    axes[1].imshow(np.array(pred_occupancy_map).astype(np.float32), cmap=cmap_obs, interpolation='nearest', )
    
    # axes[2].quiver(pred_flow[k][..., 1], pred_flow[k][..., 0], color='black', alpha=0.5)
    axes[2].imshow(np.array(wp_origin)[0,...,0].astype(np.float32), cmap=cmap_obs, interpolation='nearest', )
    plt.show()

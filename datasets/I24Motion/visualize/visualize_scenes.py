import matplotlib.pyplot as plt
import matplotlib.animation as animation
import flow_vis
import numpy as np
import typing
from .utils import get_cmap

def process_data_for_visualization(occupancy_flow_map_config, history_data_dic, future_data_dic):

    occupancy_flow_map_height = occupancy_flow_map_config.occupancy_flow_map_height
    occupancy_flow_map_width = occupancy_flow_map_config.occupancy_flow_map_width
    
    history_data_res_dic = typing.DefaultDict(dict)
    future_data_res_dic = typing.DefaultDict(dict)

    for scene in ['prv', 'cur', 'nxt']:
        history_data_res_dic[scene]['observed_occupancy_map'] = np.squeeze(history_data_dic[scene]['his/observed_occupancy_map']) # H,W,T,1
        # history_data_res_dic[scene]['occluded_occupancy_map'] = history_data_dic[scene]['his/occluded_occupancy_map'] # H,W,T,1
        history_data_res_dic[scene]['occluded_occupancy_map'] = np.squeeze(np.zeros_like(history_data_dic[scene]['his/observed_occupancy_map']))
        history_data_res_dic[scene]['flow_map'] = np.squeeze(history_data_dic[scene]['his/flow_map'])# H,W,T,2
        history_data_res_dic[scene]['flow_map_rendered'] = np.squeeze(np.sum(history_data_dic[scene]['his/flow_map'], axis=-2))
    
    for scene in ['prv', 'cur', 'nxt']:
        future_data_res_dic[scene]['observed_occupancy_map'] = np.squeeze(future_data_dic[scene]['pred/observed_occupancy_map'])
        future_data_res_dic[scene]['occluded_occupancy_map'] = np.squeeze(future_data_dic[scene]['pred/occluded_occupancy_map'])
        future_data_res_dic[scene]['flow_map'] = np.squeeze(future_data_dic[scene]['pred/flow_map'])
        future_data_res_dic[scene]['flow_map_rendered'] = np.squeeze(np.sum(future_data_dic[scene]['pred/flow_map'], axis=-2))
            
    x = np.arange(0, occupancy_flow_map_width)  # X-axis points (256)
    y = np.arange(0, occupancy_flow_map_height)  # Y-axis points (128)
    X, Y = np.meshgrid(x, y)
    
    return X, Y, history_data_res_dic, future_data_res_dic


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

def visualize(config, name, history_data_dic, future_data_dic, vis_occ=True, vis_flow=True, vis_optical_flow=True):
    occupancy_flow_map_config = config.dataset_config.occupancy_flow_map
    X, Y, history_data_dic, future_data_dic = process_data_for_visualization(occupancy_flow_map_config, history_data_dic, future_data_dic)

    # road_map = get_road_map(config, batch_first=False).swapaxes(0,1)
    num_history_time_steps = history_data_dic['cur']['observed_occupancy_map'].shape[-1]
    num_future_time_steps = future_data_dic['cur']['observed_occupancy_map'].shape[-1]
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
    axes[0].set_title('Previous Scene')
    axes[1].set_title('Current Scene')
    axes[2].set_title('Next Scene')
    if vis_optical_flow:
            flow_rendered_prv = axes[0].imshow(flow_vis.flow_to_color(history_data_dic['prv']['flow_map_rendered']),interpolation='nearest', alpha=1)
            flow_rendered_cur = axes[1].imshow(flow_vis.flow_to_color(history_data_dic['cur']['flow_map_rendered']),interpolation='nearest', alpha=1)
            flow_rendered_nex = axes[2].imshow(flow_vis.flow_to_color(history_data_dic['nxt']['flow_map_rendered']),interpolation='nearest', alpha=1)
    # Initialize the occupancy map with the first frame's data
    if vis_occ:
        
        img_prv_occ_ogm = axes[0].imshow(history_data_dic['prv']['occluded_occupancy_map'][:,:, 0], cmap=cmap_occ, interpolation='nearest', alpha=0.5)
        img_cur_occ_ogm = axes[1].imshow(history_data_dic['cur']['occluded_occupancy_map'][:,:, 0], cmap=cmap_occ, interpolation='nearest', alpha=0.5)
        img_nxt_occ_ogm = axes[2].imshow(history_data_dic['nxt']['occluded_occupancy_map'][:,:, 0], cmap=cmap_occ, interpolation='nearest', alpha=0.5)

        img_prv_obs_ogm = axes[0].imshow(history_data_dic['prv']['observed_occupancy_map'][:,:, 0], cmap=cmap_obs, interpolation='nearest', alpha=0.5)
        img_cur_obs_ogm = axes[1].imshow(history_data_dic['cur']['observed_occupancy_map'][:,:, 0], cmap=cmap_obs, interpolation='nearest', alpha=0.5)
        img_nxt_obs_ogm = axes[2].imshow(history_data_dic['nxt']['observed_occupancy_map'][:,:, 0], cmap=cmap_obs, interpolation='nearest', alpha=0.5)

    if vis_flow:
        
        quiver_prv = initialize_quiver(axes[0], history_data_dic['prv']['flow_map'][:, :, 0, :], X, Y)
        quiver_cur = initialize_quiver(axes[1], history_data_dic['cur']['flow_map'][:, :, 0, :], X, Y)
        quiver_nxt = initialize_quiver(axes[2], history_data_dic['nxt']['flow_map'][:, :, 0, :], X, Y)
    
    # Animation update function
    def update(frame):
        """Update both the quiver plot and the occupancy map for each frame."""
        # Update occupancy map
        
        for ax in axes:
            ax.clear()
            ax.axis('off')
        axes[0].set_title('Previous Scene')
        axes[1].set_title('Current Scene')
        axes[2].set_title('Next Scene')
        axes[0].scatter(X[::8, ::8], Y[::8, ::8], color='black', s=1, alpha=0.5)
        axes[1].scatter(X[::8, ::8], Y[::8, ::8], color='black', s=1, alpha=0.5)
        axes[2].scatter(X[::8, ::8], Y[::8, ::8], color='black', s=1, alpha=0.5)   
        if frame <= num_history_time_steps - 1:
            data_dic = history_data_dic
            flow_frame = frame - 2
            if flow_frame <= 0:
                flow_frame = 0
        else:
            data_dic = future_data_dic
            frame = frame - num_history_time_steps
            flow_frame = frame
            # set the title for the future scene
            # set the big title for the future scene
            fig.suptitle('Future Scene', fontsize=16)
            axes[0].set_title('Previous Scene')
            axes[1].set_title('Current Scene')
            axes[2].set_title('Next Scene')
        
        # img_road_map = axes[1].imshow(road_map)
        
        
        if vis_optical_flow:

            flow_rendered_prv = axes[0].imshow(flow_vis.flow_to_color(data_dic['prv']['flow_map_rendered']),interpolation='nearest', alpha=1)

            flow_rendered_cur = axes[1].imshow(flow_vis.flow_to_color(data_dic['cur']['flow_map_rendered']),interpolation='nearest', alpha=1)

            flow_rendered_nxt = axes[2].imshow(flow_vis.flow_to_color(data_dic['nxt']['flow_map_rendered']),interpolation='nearest', alpha=1)
            
        # Initialize the occupancy map with the first frame's data
        if vis_occ:
            
            img_prv_occ_ogm = axes[0].imshow(data_dic['prv']['occluded_occupancy_map'][:, :, frame], cmap=cmap_occ, interpolation='nearest', alpha=0.5)
            img_cur_occ_ogm = axes[1].imshow(data_dic['cur']['occluded_occupancy_map'][:, :, frame], cmap=cmap_occ, interpolation='nearest', alpha=0.5)
            img_nxt_occ_ogm = axes[2].imshow(data_dic['nxt']['occluded_occupancy_map'][:, :, frame], cmap=cmap_occ, interpolation='nearest', alpha=0.5)

            img_prv_obs_ogm = axes[0].imshow(data_dic['prv']['observed_occupancy_map'][:, :, frame], cmap=cmap_obs, interpolation='nearest', alpha=0.5)
            img_cur_obs_ogm = axes[1].imshow(data_dic['cur']['observed_occupancy_map'][:, :, frame], cmap=cmap_obs, interpolation='nearest', alpha=0.5)
            img_nxt_obs_ogm = axes[2].imshow(data_dic['nxt']['observed_occupancy_map'][:, :, frame], cmap=cmap_obs, interpolation='nearest', alpha=0.5)


        if vis_flow:

            quiver_prv = initialize_quiver(axes[0], data_dic['prv']['flow_map'][:, :, flow_frame, :], X, Y)
            quiver_cur = initialize_quiver(axes[1], data_dic['cur']['flow_map'][:, :, flow_frame, :], X, Y)
            quiver_nxt = initialize_quiver(axes[2], data_dic['nxt']['flow_map'][:, :, flow_frame, :], X, Y)
        return_list = []
        # return_list = [img_road_map]
        if vis_flow:
                return_list.extend([quiver_prv])
                return_list.extend([quiver_cur])
                return_list.extend([quiver_nxt])
        if vis_optical_flow:
                return_list.extend([flow_rendered_prv])
                return_list.extend([flow_rendered_cur])
                return_list.extend([flow_rendered_nxt])
        if vis_occ:
                return_list.extend([img_prv_obs_ogm])
                return_list.extend([img_cur_obs_ogm])
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








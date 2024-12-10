import numpy as np
class GridMap():
    def __init__(self, config):
        self.data_sample_frequency = 25
        self.map_size = (config.occupancy_flow_map.grid_size.x, config.occupancy_flow_map.grid_size.y)
        self.x_range = (0, config.preprocessing.spatial_window)
        self.y_range = (-80, 80)
        self.grid_size_x = (self.x_range[1] - self.x_range[0]) / self.map_size[0]
        self.grid_size_y = (self.y_range[1] - self.y_range[0]) / self.map_size[1]
        # number of agent points per side to describe the agent's occupancy
        self.agent_points_per_side_length = config.preprocessing.agent_points.per_side_length
        self.agent_points_per_side_width =config.preprocessing.agent_points.per_side_width
        self.history_length = config.task_config.history_length
        self.prediction_length = config.task_config.prediction_length
        self.num_his_points = config.task_config.num_his_points
        self.num_waypoints = config.task_config.num_waypoints
        self.config = config
        
    def get_agents_points(self, data):
        # start_pos = data['start_pos']
        # start_time = data['start_time']
    
        # # Shift the x and timestamp of data to the origin
        # data['x_position'] -= start_pos
        # data['timestamp'] -= start_time
    
        length = data['length'].reshape(-1, 1)
        width = data['width'].reshape(-1, 1)
        yaw_angle = data['yaw_angle']
        x_centers = data['x_position']
        y_centers = data['y_position'] + self.y_range[1]  # Shift y to the positive range
        
        centers = np.stack([x_centers, y_centers], axis=-1)  # Shape: (num_agents, T, 2)
        num_of_agents = x_centers.shape[0]
        num_time_steps = x_centers.shape[1]
    
        # Create the base grid
        x = np.linspace(-1, 1, self.agent_points_per_side_length).reshape(-1, 1)
        y = np.linspace(-1, 1, self.agent_points_per_side_width).reshape(1, -1)
        base_grid = np.stack(np.meshgrid(x, y, indexing='ij'), axis=-1)  # Shape: (48, 16, 2)
    
        # Rescale the grid based on each vehicle's length and width
        lengths_rescaled = length[:, None, None, 0] / 2  # Shape: (num_agents, 1, 1)
        widths_rescaled = width[:, None, None, 0] / 2  # Shape: (num_agents, 1, 1)
    
        grid_scaled = np.zeros((num_of_agents, self.agent_points_per_side_length, self.agent_points_per_side_width, 2))
        grid_scaled[..., 0] = base_grid[..., 0] * lengths_rescaled  # Scale x-axis
        grid_scaled[..., 1] = base_grid[..., 1] * widths_rescaled  # Scale y-axis
    
        # Flatten the scaled grid for rotation
        grid_flat = grid_scaled.reshape((num_of_agents, -1, 2))  # Shape: (num_agents, 768, 2)
    
        # Create rotation matrices
        cos_yaw = np.cos(yaw_angle).reshape(num_of_agents, num_time_steps, 1, 1)
        sin_yaw = np.sin(yaw_angle).reshape(num_of_agents, num_time_steps, 1, 1)
    
        rotation_matrix = np.concatenate([
            np.concatenate([cos_yaw, -sin_yaw], axis=-1),  # First row: [cos, -sin]
            np.concatenate([sin_yaw, cos_yaw], axis=-1)  # Second row: [sin, cos]
        ], axis=-2)  # Shape: (num_agents, T, 2, 2)
        
        # Apply the rotation using batch matrix multiplication
        # Ensure grid_flat matches the shape for each time step
        # Ensure grid_flat matches the shape for each time step (num_agents, T, 768, 2)
        grid_flat = np.repeat(grid_flat[:, None, :, :], num_time_steps, axis=1)
        
        # Apply the batch matrix multiplication
        transformed_points = np.einsum('ntij,ntkj->ntki', rotation_matrix, grid_flat)  # Shape: (num_agents, T, 768, 2)
        
        # Reshape back to (num_agents, T, 48, 16, 2)
        transformed_points = transformed_points.reshape(
            num_of_agents, num_time_steps, self.agent_points_per_side_length, self.agent_points_per_side_width, 2
        )
        # Add the center coordinates to shift the points accordingly
        vehicle_points = transformed_points + centers[:, :, None, None, :]  # Shape: (num_agents, T, 48, 16, 2)
        
        return vehicle_points

    def get_map(self, num_time_steps, timestamp, vehicle_grids):
        
        num_points_per_vehicle = vehicle_grids.shape[2]
        # timestamp (num_of_agents, num_time_steps)
        timestamp_repeated = np.repeat(timestamp[..., None], num_points_per_vehicle, axis=2)
        timestamp_valid = (timestamp_repeated > 0)
        vehicle_grids_valid = (vehicle_grids[...,0] < self.config.occupancy_flow_map.grid_size.x) & (vehicle_grids[...,0] >= 0) & (vehicle_grids[...,1] < self.config.occupancy_flow_map.grid_size.y) & (vehicle_grids[...,1] >= 0)
        valid_agent_indices, valid_time_indices, valid_agent_points_indices = (np.where(timestamp_valid & vehicle_grids_valid))
        valid_agent_indices = valid_agent_indices.astype(np.int32)[..., np.newaxis]
        valid_time_indices = valid_time_indices.astype(np.int32)[..., np.newaxis]
        valid_agent_points_indices = valid_agent_points_indices.astype(np.int32)[..., np.newaxis]
        
        x_img_coord = vehicle_grids[valid_agent_indices, valid_time_indices, valid_agent_points_indices, 0]
        y_img_coord = vehicle_grids[valid_agent_indices, valid_time_indices, valid_agent_points_indices, 1]

        # [num_points_to_render, 3]
        xy_img_coord = np.concatenate(
          [
              y_img_coord.astype(np.int32),
              x_img_coord.astype(np.int32),
              valid_time_indices,
          ],
          axis=1,
        )
        # [num_points_to_render]
        gt_values = np.squeeze(np.ones_like(x_img_coord, dtype=np.float32), axis=-1)
        topdown_shape = [self.config.occupancy_flow_map.grid_size.y, self.config.occupancy_flow_map.grid_size.x, num_time_steps]
        # [batch_size, grid_height_cells, grid_width_cells, num_steps]
        xy_img_coord_t = tuple(xy_img_coord.T) 
        occupancy_map = np.zeros(topdown_shape, dtype=np.float32)
        np.add.at(occupancy_map, xy_img_coord_t, gt_values.squeeze())
        # scatter_nd() accumulates values if there are repeated indices.  Since
        # we sample densely, this happens all the time.  Clip the final values.
        return np.clip(occupancy_map, 0.0, 1.0)

    
    def get_flow(self, timestamp, vehicle_points, vehicle_grids):
        # timestamp (num_of_agents, num_time_steps)
        # timestamp_repeated (num_of_agents, num_time_steps, num_points_per_vehicle)
        num_flow_steps = self.config.task_config.num_waypoints + self.config.task_config.num_his_points - 1
        num_points_per_vehicle = vehicle_grids.shape[2]
        timestamp_repeated= timestamp[...,None].repeat(num_points_per_vehicle, axis = 2)
        timestamp = timestamp_repeated[:, :-1,...]
        timestamp_shifted = timestamp_repeated[:, 1:,...]
        timestamp_valid = (timestamp > 0) & (timestamp_shifted > 0)
        # timestamp_repeated (num_of_agents, num_time_steps, num_points_per_vehicle)
        vehicle_grids = vehicle_grids[:, 1:,...]
        vehicle_grids_valid = (vehicle_grids[...,0] < self.config.occupancy_flow_map.grid_size.x) & (vehicle_grids[...,0] >= 0) & (vehicle_grids[...,1] < self.config.occupancy_flow_map.grid_size.y) & (vehicle_grids[...,1] >= 0)
        valid_agent_indices, valid_time_indices, valid_agent_points_indices = (np.where(timestamp_valid & vehicle_grids_valid))
        # cast the indices to int32
        valid_agent_indices = valid_agent_indices.astype(np.int32)[..., np.newaxis]
        valid_time_indices = valid_time_indices.astype(np.int32)[..., np.newaxis]
        valid_agent_points_indices = valid_agent_points_indices.astype(np.int32)[..., np.newaxis]
        
        x_img_coord = vehicle_grids[valid_agent_indices, valid_time_indices, valid_agent_points_indices, 0]
        y_img_coord = vehicle_grids[valid_agent_indices, valid_time_indices, valid_agent_points_indices, 1]
        xy_img_coord = np.concatenate(
          [
              y_img_coord.astype(np.int32),
              x_img_coord.astype(np.int32),
              valid_time_indices,
          ],
          axis=1,
        )
        vehicle_points_dxy = vehicle_points[:, :-1,...] - vehicle_points[:, 1:,...]
        vehicle_points_dx = vehicle_points_dxy[..., 0]
        vehicle_points_dy = vehicle_points_dxy[..., 1]
        
        topdown_shape = [self.config.occupancy_flow_map.grid_size.y, self.config.occupancy_flow_map.grid_size.x, num_flow_steps]
        gt_values_dx = vehicle_points_dx[valid_agent_indices, valid_time_indices, valid_agent_points_indices]
        gt_values_dy = vehicle_points_dy[valid_agent_indices, valid_time_indices, valid_agent_points_indices]

        # tf.scatter_nd() accumulates values when there are repeated indices.
        # Keep track of number of indices writing to the same pixel so we can
        # account for accumulated values.
        # [num_points_to_render]
        gt_values = np.squeeze(np.ones_like(x_img_coord, dtype=np.float32), axis=-1)

        # [batch_size, grid_height_cells, grid_width_cells, num_flow_steps]
        flow_x = np.zeros(topdown_shape, dtype=np.float32)
        flow_y = np.zeros(topdown_shape, dtype=np.float32)
        num_values_per_pixel = np.zeros(topdown_shape, dtype=np.float32)
        xy_img_coord_t = tuple(xy_img_coord.T) 
        np.add.at(flow_x, xy_img_coord_t, gt_values_dx.squeeze())
        np.add.at(flow_y, xy_img_coord_t, gt_values_dy.squeeze())
        np.add.at(num_values_per_pixel, xy_img_coord_t, gt_values)
        flow_x = np.divide(flow_x, num_values_per_pixel, out=np.zeros_like(flow_x), where=(num_values_per_pixel != 0))
        flow_y = np.divide(flow_y, num_values_per_pixel, out=np.zeros_like(flow_y), where=(num_values_per_pixel != 0))
        flow = np.stack([flow_x, flow_y], axis=-1)
        return flow
        
  
    def get_map_flow(self, data):
        num_of_agents = data['x_position'].shape[0]
        num_time_steps = self.num_his_points + self.num_waypoints
        if num_of_agents == 0:
            return  np.zeros([num_time_steps, *self.map_size]), np.zeros([num_time_steps, *self.map_size]), np.zeros([num_time_steps, *self.map_size, 2])
        timestamp = data['timestamp']
        
        vehicle_points = self.get_agents_points(data)
        num_of_agents, num_time_steps = vehicle_points.shape[0], vehicle_points.shape[1]
        vehicle_points = vehicle_points.reshape(num_of_agents, num_time_steps, self.agent_points_per_side_length * self.agent_points_per_side_width, 2)
        if np.isnan(vehicle_points).any() or np.isinf(vehicle_points).any():
        # Handle invalid values
            vehicle_points = np.nan_to_num(vehicle_points, nan=0.0, posinf=0.0, neginf=0.0)

        vehicle_grids = np.floor(vehicle_points / np.array((self.grid_size_x, self.grid_size_y))).astype(int)
        
        # Calculate the occupancy map
        occluded_idx = np.argmax((timestamp > 0), axis=1) > self.num_his_points
        occluded_occupancy_map = self.get_map(num_time_steps, timestamp[occluded_idx], vehicle_grids[occluded_idx])
        observed_occupancy_map = self.get_map(num_time_steps, timestamp[~occluded_idx], vehicle_grids[~occluded_idx])
        # Calculate the flow map
        flow_map = self.get_flow(timestamp, vehicle_points, vehicle_grids)
        
        # return occluded_occupancy_map, observed_occupancy_map, flow_map
        return  occluded_occupancy_map, observed_occupancy_map, flow_map

if __name__ == '__main__':
    from utils.file_utils import get_config
    config = get_config('./config_12.yaml')
    grid_map = GridMap(config)
    test_data = np.load(config.paths.processed_data + 'scene_1014.npy', allow_pickle=True).item()
    occ_map, obs_map ,flow_map = grid_map.get_map_flow(test_data)
    print(occ_map.shape, obs_map.shape, flow_map.shape)
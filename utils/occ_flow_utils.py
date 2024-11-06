import numpy as np

class GridMap():
    def __init__(self, config):
        self.data_sample_frequency = 25
        self.map_size = (config.occ_flow_map.grid_size_x, config.occ_flow_map.grid_size_y)
        self.x_range = (0, config.preprocess.spatial_window)
        self.y_range = (-80, 80)
        self.grid_size_x = (self.x_range[1] - self.x_range[0]) / self.map_size[0]
        self.grid_size_y = (self.y_range[1] - self.y_range[0]) / self.map_size[1]
        # number of agent points per side to describe the agent's occupancy
        self.agent_points_per_side_length = config.preprocess.agent_points_per_side_length
        self.agent_points_per_side_width =config.preprocess.agent_points_per_side_width
        self.his_len = config.task.his_len
        
    def get_agents_points(self, data):
        # start_pos = data['start_pos']
        # print(f'start_pos {start_pos}')
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

    def get_map(self, num_of_agents, num_time_steps, timestamp, vehicle_grids, occluded_idx):
        
        occupancy_map = np.zeros([num_of_agents, num_time_steps, *self.map_size], dtype=np.float16)
        # map = np.repeat(map[None, ...], num_of_agents, axis=0)
       # Extract valid indices from the valid_mask
        valid_agent_indices, valid_time_indices = np.where(timestamp > 0)  # Shape: (N_valid,), (N_valid,)
        # Get the corresponding vehicle grid points for valid entries
        vehicle_grids_masked = vehicle_grids[valid_agent_indices, valid_time_indices]  # Shape: (N_valid, 2560, 2)
        # Extract x and y coordinates
        grid_x = vehicle_grids_masked[:, :, 0].ravel()  # Shape: (N_valid * 2560,)
        grid_y = vehicle_grids_masked[:, :, 1].ravel()  # Shape: (N_valid * 2560,)
        # Repeat agent and time indices to match grid points
        
        agent_indices_repeated = np.repeat(valid_agent_indices, vehicle_grids_masked.shape[1])  # Shape: (N_valid * 2560,)
        time_indices_repeated = np.repeat(valid_time_indices, vehicle_grids_masked.shape[1])  # Shape: (N_valid * 2560,)
        valid_mask = (grid_x >= 0) & (grid_x < self.map_size[0]) & (grid_y >= 0) & (grid_y < self.map_size[1])

        # Apply the mask to keep only valid indices
        grid_x = grid_x[valid_mask]
        grid_y = grid_y[valid_mask]
        agent_indices_repeated = agent_indices_repeated[valid_mask]
        time_indices_repeated = time_indices_repeated[valid_mask]
        # Assign 1s to the map using advanced indexing
        occupancy_map[agent_indices_repeated, time_indices_repeated, grid_x, grid_y] = 1
        occluded_occupancy_map = np.sum(occupancy_map[occluded_idx], axis=0)
        observed_occupancy_map = np.sum(occupancy_map[~occluded_idx], axis=0)
        return np.clip(occluded_occupancy_map, 0, 1), np.clip(observed_occupancy_map, 0, 1)
    
    def get_flow(self, num_of_agents, num_time_steps, timestamp, vehicle_points, vehicle_grids):
        
        flow_map = np.zeros([num_of_agents, num_time_steps, *self.map_size, 2])
        timestamp_shifted = np.roll(timestamp, shift=1, axis=1)
        valid_mask = (timestamp > 0) & (timestamp_shifted > 0)
        # get the first valid time index
        first_true_idx = np.argmax(valid_mask, axis=1)
        valid_mask[:, first_true_idx] = False  # First time step has no flow
        valid_agent_indices, valid_time_indices = np.where(valid_mask)  # Shape: (N_valid,), (N_valid,)
        
        # Get the corresponding vehicle grid points for valid entries
   
        vehicle_grids_masked = vehicle_grids[valid_agent_indices, valid_time_indices]  # Shape: (N_valid, 2560, 2)
        vehicle_points_masked = (np.roll(vehicle_points, shift=1, axis=1) - vehicle_points)[valid_agent_indices, valid_time_indices]  # Shape: (N_valid, 2560, 2)
        # set the first valid time index to 0
        
        
        # Repeat agent and time indices to match grid points
        agent_indices_repeated = np.repeat(valid_agent_indices, vehicle_grids_masked.shape[1])  # Shape: (N_valid * 2560,)
        time_indices_repeated = np.repeat(valid_time_indices, vehicle_grids_masked.shape[1])  # Shape: (N_valid * 2560,)
        
        grid_x = vehicle_grids_masked[:, :, 0].ravel() # Shape: (N_valid * 2560,)
        grid_y = vehicle_grids_masked[:, :, 1].ravel()  # Shape: (N_valid * 2560,)
        valid_mask = (grid_x >= 0) & (grid_x < self.map_size[0]) & (grid_y >= 0) & (grid_y < self.map_size[1])

        # Apply the mask to keep only valid indices
      
        agent_indices_repeated = agent_indices_repeated[valid_mask]
        time_indices_repeated = time_indices_repeated[valid_mask]
        flow_map[agent_indices_repeated, time_indices_repeated, grid_x[valid_mask], grid_y[valid_mask]] = vehicle_points_masked.reshape(-1, 2)[valid_mask]
 
        return np.sum(flow_map, axis=0)
    
    def get_map_flow(self, data):
        num_of_agents = data['x_position'].shape[0]
        num_time_steps = data['x_position'].shape[1]
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
        occluded_idx = np.argmax((timestamp > 0), axis=1) > self.his_len
        num_of_agents_occluded = np.sum(occluded_idx)
        num_of_agents_observed = num_of_agents - num_of_agents_occluded
        
        occluded_occupancy_map, observed_occupancy_map= self.get_map(num_of_agents, num_time_steps, timestamp, vehicle_grids, occluded_idx)
        # observed_occupancy_map = self.get_map(num_of_agents_observed, num_time_steps, timestamp[~occluded_idx], vehicle_grids[~occluded_idx])
        # Calculate the flow map
        flow_map = self.get_flow(num_of_agents, num_time_steps, timestamp, vehicle_points, vehicle_grids)
        
        
                    
        # return occluded_occupancy_map, observed_occupancy_map, flow_map
        return  occluded_occupancy_map, observed_occupancy_map, flow_map
    
    
# Test
if __name__ == '__main__':
    from utils.file_utils import get_config
    config = get_config()
    grid_map = GridMap(config)
    test_data = np.load('/hdd/HetianGuo/I24/processed_data/scene_20.npy', allow_pickle=True).item()
    occ_map, obs_map ,flow_map = grid_map.get_map_flow(test_data['cur'])
    
    
    
    
    
import torch

class GridMap_gpu():
    def __init__(self, config):
        self.data_sample_frequency = 25
        self.map_size = (config.occ_flow_map.grid_size_x, config.occ_flow_map.grid_size_y)
        self.x_range = (0, config.preprocess.spatial_window * 3)
        self.y_range = (-80, 80)
        self.grid_size_x = (self.x_range[1] - self.x_range[0]) / self.map_size[0]
        self.grid_size_y = (self.y_range[1] - self.y_range[0]) / self.map_size[1]
        self.agent_points_per_side_length = config.preprocess.agent_points_per_side_length
        self.agent_points_per_side_width = config.preprocess.agent_points_per_side_width
        self.his_len = config.task.his_len

    def get_agents_points(self, data):
        device = data['length'].device  # Get device from input data
        length = data['length'].view(-1, 1).to(device)
        width = data['width'].view(-1, 1).to(device)
        yaw_angle = data['yaw_angle'].to(device)
        x_centers = data['x_position'].to(device)
        y_centers = data['y_position'].to(device) + self.y_range[1]

        centers = torch.stack([x_centers, y_centers], dim=-1)
        num_of_agents = x_centers.size(0)
        num_time_steps = x_centers.size(1)

        # Base grid creation on GPU
        x = torch.linspace(-1, 1, self.agent_points_per_side_length, device=device).view(-1, 1)
        y = torch.linspace(-1, 1, self.agent_points_per_side_width, device=device).view(1, -1)
        base_grid = torch.stack(torch.meshgrid(x.squeeze(), y.squeeze(), indexing='ij'), dim=-1)

        # Scale based on vehicle dimensions
        lengths_rescaled = length[:, None, None] / 2
        widths_rescaled = width[:, None, None] / 2
        grid_scaled = torch.zeros((num_of_agents, self.agent_points_per_side_length, self.agent_points_per_side_width, 2), device=device)
        grid_scaled[..., 0] = (base_grid[..., 0] * lengths_rescaled).squeeze()
        grid_scaled[..., 1] = (base_grid[..., 1] * widths_rescaled).squeeze()
        grid_flat = grid_scaled.view(num_of_agents, -1, 2)

        # Rotation matrices
        cos_yaw = torch.cos(yaw_angle).view(num_of_agents, num_time_steps, 1, 1)
        sin_yaw = torch.sin(yaw_angle).view(num_of_agents, num_time_steps, 1, 1)
        rotation_matrix = torch.cat([torch.cat([cos_yaw, -sin_yaw], dim=-1), torch.cat([sin_yaw, cos_yaw], dim=-1)], dim=-2)

        # Batch matrix multiplication
        grid_flat = grid_flat[:, None, :, :].expand(-1, num_time_steps, -1, -1)
        transformed_points = torch.einsum('ntij,ntkj->ntki', rotation_matrix.to(dtype=torch.float32), grid_flat)

        transformed_points = transformed_points.view(num_of_agents, num_time_steps, self.agent_points_per_side_length, self.agent_points_per_side_width, 2)
        vehicle_points = transformed_points + centers[:, :, None, None, :]

        return vehicle_points

    def get_map(self, num_of_agents, num_time_steps, timestamp, vehicle_grids, occluded_idx):
        device = vehicle_grids.device
        occupancy_map = torch.zeros((num_of_agents, num_time_steps, *self.map_size), dtype=torch.float16, device=device)

        valid_agent_indices, valid_time_indices = torch.where(timestamp > 0)
        vehicle_grids_masked = vehicle_grids[valid_agent_indices, valid_time_indices]
        grid_x, grid_y = vehicle_grids_masked[..., 0].flatten(), vehicle_grids_masked[..., 1].flatten()

        agent_indices_repeated = valid_agent_indices.repeat_interleave(vehicle_grids_masked.size(1))
        time_indices_repeated = valid_time_indices.repeat_interleave(vehicle_grids_masked.size(1))
        valid_mask = (grid_x >= 0) & (grid_x < self.map_size[0]) & (grid_y >= 0) & (grid_y < self.map_size[1])

        grid_x, grid_y = grid_x[valid_mask], grid_y[valid_mask]
        agent_indices_repeated = agent_indices_repeated[valid_mask]
        time_indices_repeated = time_indices_repeated[valid_mask]
        occupancy_map[agent_indices_repeated, time_indices_repeated, grid_x, grid_y] = 1

        return occupancy_map[occluded_idx].sum(dim=0), occupancy_map[~occluded_idx].sum(dim=0)

    def get_flow(self, num_of_agents, num_time_steps, timestamp, vehicle_points, vehicle_grids):
        device = vehicle_points.device
        flow_map = torch.zeros((num_of_agents, num_time_steps, *self.map_size, 2), device=device)
        timestamp_shifted = torch.roll(timestamp, shifts=1, dims=1)
        valid_mask = (timestamp > 0) & (timestamp_shifted > 0)

        first_true_idx = torch.argmax(valid_mask.int(), dim=1)
        valid_mask[torch.arange(valid_mask.size(0)), first_true_idx] = False
        valid_agent_indices, valid_time_indices = torch.where(valid_mask)

        vehicle_grids_masked = vehicle_grids[valid_agent_indices, valid_time_indices]
        vehicle_points_masked = (torch.roll(vehicle_points, shifts=1, dims=1) - vehicle_points)[valid_agent_indices, valid_time_indices]

        agent_indices_repeated = valid_agent_indices.repeat_interleave(vehicle_grids_masked.size(1))
        time_indices_repeated = valid_time_indices.repeat_interleave(vehicle_grids_masked.size(1))
        grid_x, grid_y = vehicle_grids_masked[..., 0].flatten(), vehicle_grids_masked[..., 1].flatten()
        valid_mask = (grid_x >= 0) & (grid_x < self.map_size[0]) & (grid_y >= 0) & (grid_y < self.map_size[1])

        agent_indices_repeated = agent_indices_repeated[valid_mask]
        time_indices_repeated = time_indices_repeated[valid_mask]
        flow_map[agent_indices_repeated, time_indices_repeated, grid_x[valid_mask], grid_y[valid_mask]] = vehicle_points_masked.view(-1, 2)[valid_mask]

        return flow_map.sum(dim=0)

    def get_map_flow(self, data):
        device = data['x_position'].device
        num_of_agents = data['x_position'].size(0)
        num_time_steps = data['x_position'].size(1)

        if num_of_agents == 0:
            return (
                torch.zeros((num_time_steps, *self.map_size), device=device),
                torch.zeros((num_time_steps, *self.map_size), device=device),
                torch.zeros((num_time_steps, *self.map_size, 2), device=device)
            )

        timestamp = data['timestamp']
        vehicle_points = self.get_agents_points(data)
        vehicle_points = vehicle_points.view(num_of_agents, num_time_steps, self.agent_points_per_side_length * self.agent_points_per_side_width, 2)

        if torch.isnan(vehicle_points).any() or torch.isinf(vehicle_points).any():
            vehicle_points = torch.nan_to_num(vehicle_points, nan=0.0, posinf=0.0, neginf=0.0)

        vehicle_grids = torch.floor(vehicle_points / torch.tensor([self.grid_size_x, self.grid_size_y], device=device)).int()
        occluded_idx = torch.argmax((timestamp > 0).int(), dim=1) > self.his_len

        occluded_occupancy_map, observed_occupancy_map = self.get_map(num_of_agents, num_time_steps, timestamp, vehicle_grids, occluded_idx)
        flow_map = self.get_flow(num_of_agents, num_time_steps, timestamp, vehicle_points, vehicle_grids)

        return occluded_occupancy_map, observed_occupancy_map, flow_map

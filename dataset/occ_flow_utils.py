import numpy as np

class GridMap():
    def __init__(self, config):
        self.data_sample_frequency = 25
        self.map_size = (config.dataset.grid_size_x, config.dataset.grid_size_y)
        self.x_range = (0, config.dataset.spatial_window)
        self.y_range = (-80, 80)
        # self.map = np.zeros([config.dataset.temporal_window, *self.map_size])
        # self.flow = np.zeros([config.dataset.temporal_window, *self.map_size, 2])
        self.grid_size_x = (self.x_range[1] - self.x_range[0]) / self.map_size[0]
        self.grid_size_y = (self.y_range[1] - self.y_range[0]) / self.map_size[1]
        # number of agent points per side to describe the agent's occupancy
        self.agent_points_per_side_length = config.preprocess.agent_points_per_side_length
        self.agent_points_per_side_width =config.preprocess.agent_points_per_side_width
        
    @ DeprecationWarning
    def load_data(self, data):
        
        self.data = data
        self.shift_data()
        self.generate_valid_mask()
    @ DeprecationWarning
    def generate_valid_mask(self):
        # generate validity mask
        valid_idx = np.where(self.data['timestamp'] != 0)
        valid_mask = np.zeros(self.data['timestamp'].shape)
        valid_mask[valid_idx] = 1
        self.valid_mask = valid_mask
        self.valid_idx = valid_idx
    @ DeprecationWarning
    def shift_data(self):
        start_pos = self.data['start_pos']
        start_time = self.data['start_time']
        # shift the x_pos of data to the origin
        self.data['x_position'] = self.data['x_position'] - start_pos
        # shift the timestamp of data to the origin
        self.data['timestamp'] = self.data['timestamp'] - start_time
    @ DeprecationWarning
    def derive_bboxes(self):
        vehicle_width = self.data['width']
        vehicle_length = self.data['length']
        
        x_position = self.data['x_position']
        y_position = self.data['y_position']
        yaw_angle = self.data['yaw_angle']
        timestamp = self.data['timestamp']
        vehicle_num, time_span = timestamp.shape
        # print(vehicle_num, time_span)
        vehicle_length = np.repeat(vehicle_length[:, None], time_span, axis=1)
        vehicle_width = np.repeat(vehicle_width[:, None], time_span, axis=1)
        
        bbox_corners = np.zeros([vehicle_num, time_span, 4, 2])
        bbox_corners[:, :, 0, 0] = -vehicle_length / 2
        bbox_corners[:, :, 0, 1] = -vehicle_width / 2
        bbox_corners[:, :, 1, 0] = vehicle_length / 2
        bbox_corners[:, :, 1, 1] = -vehicle_width / 2
        bbox_corners[:, :, 2, 0] = vehicle_length / 2
        bbox_corners[:, :, 2, 1] = vehicle_width / 2
        bbox_corners[:, :, 3, 0] = -vehicle_length / 2
        bbox_corners[:, :, 3, 1] = vehicle_width / 2
        # get the rotation matrix
        rotation_matrix = np.zeros([vehicle_num, time_span, 2, 2])
        rotation_matrix[:, :, 0, 0] = np.cos(yaw_angle)
        rotation_matrix[:, :, 0, 1] = -np.sin(yaw_angle)
        rotation_matrix[:, :, 1, 0] = np.sin(yaw_angle)
        rotation_matrix[:, :, 1, 1] = np.cos(yaw_angle)
        # rotate the bounding box
        rotated_bbox = np.zeros(bbox_corners.shape)
        for i in range(vehicle_num):
            # print(rotation_matrix[i].shape)
            # print(np.swapaxes(bbox_corners[i], axis1=-1, axis2=-2).shape)
            rotated_bbox[i] = np.swapaxes(np.matmul(rotation_matrix[i], np.swapaxes(bbox_corners[i], axis1=-1, axis2=-2)), axis1=-1, axis2=-2)
        # translate the bounding box
        translated_bbox = np.zeros(rotated_bbox.shape)
        for i in range(vehicle_num):
            translated_bbox[i] = rotated_bbox[i] + np.swapaxes(np.array([x_position[i], y_position[i]]), axis1=-1, axis2=-2)[:, None, ...]
        self.data['bboxes'] = translated_bbox
    @ DeprecationWarning
    def bboxes2occupancy(self):
        # calculate the grids within the bounding box
        bboxes = self.data['bboxes']
        x = self.data['x_position']
        y = self.data['y_position']
        timestamp = self.data['timestamp']
        # print(bboxes.shape)
        for vehicle_idx in range(bboxes.shape[0]):
            for time_idx in range(bboxes.shape[1]):
                if timestamp[vehicle_idx, time_idx] <= 0:
                    continue
                bbox = bboxes[vehicle_idx, time_idx]
                occupied_grids = self.bbox2grids(bbox).astype(int)
                # print(occupied_grids)
                # set the occupancy of the grids to 1 using array slicing
                # print(f'map shape{self.map.shape}')
                self.map[time_idx, occupied_grids[:, 0], occupied_grids[:, 1]] = 1
    @ DeprecationWarning        
    def bbox2grids(self, bbox):
        min_x, min_y = np.floor(np.min(bbox, axis=0))
        max_x, max_y = np.floor(np.max(bbox, axis=0))
        # print(min_x, min_y, max_x, max_y)
        
        #Step 2: Create a meshgrid of all possible grid cells in the bounding box area
        x_indices = np.arange(min_x, max_x + 1)
        y_indices = np.arange(min_y, max_y + 1)
        
        grid_x, grid_y = np.meshgrid(x_indices, y_indices)
        grid_centers = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        # print(grid_centers.shape)
        inside_mask = self.is_point_in_polygon_vectorized(grid_centers, bbox)

        # Step 6: Convert the filtered grid centers back to grid indices
        occupied_grids = np.column_stack((grid_y.ravel()[inside_mask], grid_x.ravel()[inside_mask]))
        # filter out the grids that are out of the map
        valid_idx = np.where((occupied_grids[:, 0] >= 0) & (occupied_grids[:, 0] < self.map_size[0]) & (occupied_grids[:, 1] >= 0) & (occupied_grids[:, 1] < self.map_size[1]))
        occupied_grids = occupied_grids[valid_idx]
        return occupied_grids
    @ DeprecationWarning
    def is_point_in_polygon_vectorized(self, points, polygon):
        """ Vectorized point-in-polygon check using cross products. """
        n = polygon.shape[0]
        inside = np.ones(points.shape[0], dtype=bool)

        # Compute the cross-product sign for each polygon edge
        for i in range(n):
            j = (i + 1) % n
            edge = polygon[j] - polygon[i]  # Vector for each polygon edge
            edge_perp = np.array([-edge[1], edge[0]])  # Perpendicular vector to the edge
            vec_to_points = points - polygon[i]  # Vector from edge to points
            cross_products = np.dot(vec_to_points, edge_perp)  # Calculate cross product
            inside &= (cross_products >= 0)

        return inside
    
    
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
        y_centers = data['y_position']
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


    
    def get_map_flow(self, data):
        num_of_agents = data['x_position'].shape[0]
        num_time_steps = data['x_position'].shape[1]
        if num_of_agents == 0:
            return  np.zeros([num_time_steps, *self.map_size]), np.zeros([num_time_steps, *self.map_size, 2])
        timestamp = data['timestamp']
        
        vehicle_points = self.get_agents_points(data)
        num_of_agents, num_time_steps = vehicle_points.shape[0], vehicle_points.shape[1]
        vehicle_points = vehicle_points.reshape(num_of_agents, num_time_steps, self.agent_points_per_side_length * self.agent_points_per_side_width, 2)
        # map the vehicles points to the grid map
        grid_size = np.array((self.grid_size_x, self.grid_size_y))
        if np.isnan(vehicle_points).any() or np.isinf(vehicle_points).any():
        # Handle invalid values
            vehicle_points = np.nan_to_num(vehicle_points, nan=0.0, posinf=0.0, neginf=0.0)

        vehicle_grids = np.floor(vehicle_points / grid_size).astype(int)
        # Calculate the occupancy map
        map = np.zeros([num_of_agents, num_time_steps, *self.map_size], dtype=np.float16)
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
        map[agent_indices_repeated, time_indices_repeated, grid_x, grid_y] = 1
        # Sum the map across agents
        map = np.sum(map, axis=0)
        
        flow = np.zeros([num_of_agents, num_time_steps, *self.map_size, 2])
        # flow = np.repeat(flow[None, ...], num_of_agents, axis=0)
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
        flow[agent_indices_repeated, time_indices_repeated, grid_x[valid_mask], grid_y[valid_mask]] = vehicle_points_masked.reshape(-1, 2)[valid_mask]
        flow = np.sum(flow, axis=0)
                    
        return map, flow
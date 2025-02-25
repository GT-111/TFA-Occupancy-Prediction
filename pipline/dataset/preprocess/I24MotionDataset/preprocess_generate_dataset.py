import os
import typing
import concurrent.futures
import pandas as pd
import numpy as np
from tqdm import tqdm
from pipline.dataset.dataset_utils.I24Motion_utils.occupancy_flow_map_utils import GridMap
from configs.utils.config import load_config
from pipline.utils.file_utils import get_files_with_extension

if __name__ == '__main__':
    dataset_config = load_config("configs/dataset_configs/I24Motion_config.py")


class I24MotionDatasetFile():
    def __init__(self, data_file_path, dataset_config):
        self.data_file_path = data_file_path
        # assert the data format is parquet and the data file exists
        assert data_file_path.endswith('.parquet') and os.path.exists(data_file_path)
        # load the data file
        self.data_df = pd.read_parquet(data_file_path)
        # ============= Load Paths =================
        self.paths = dataset_config.paths
        self.generated_data_path = self.paths.generated_data_path
        # ============= Load Occpancy Flow Map Config =================
        self.occupancy_flow_map_config = dataset_config.occupancy_flow_map
        # ============= Load Trajectory COnfig =================
        self.trajectory_config = dataset_config.trajectory
        self.node_features_list = self.trajectory_config.node_features_list
        self.vector_features_list = self.trajectory_config.vector_features_list
        # ============= Load Task Config =================
        self.task_config = dataset_config.task
        self.grid_map_helper = GridMap(occupancy_flow_map_config=self.occupancy_flow_map_config, task_config=self.task_config)
        self.start_pos = dataset_config.start_position * 5280 # mile to feet
        self.end_pos = dataset_config.data_attributes.end_position * 5280  # mile to feet
        # ============= Prepare the Data Attributes =================
        self.x_min = max(self.data_df['x_position'].min(), self.start_pos)
        self.x_max = self.data_df['x_position'].max()
        self.time_min = max(self.data_df['timestamp'].min(), 0)
        self.time_max = self.data_df['timestamp'].max()
        self.data_df.set_index('timestamp', inplace=True)
        self.data_df.sort_index(inplace=True)
        # ============= Prepare the Occpancy Flow Map Attributes =================
        self.spatial_stride = self.occupancy_flow_map_config.spatial_stride
        self.temporal_stride = self.occupancy_flow_map_config.temporal_stride
        self.spatial_window = self.occupancy_flow_map_config.spatial_window
        self.temporal_window = self.occupancy_flow_map_config.temporal_window
        self.spatial_length = int(np.floor((self.x_max - self.x_min - self.spatial_window) / self.spatial_stride))
        self.temporal_length = int(np.floor((self.time_max - self.time_min - self.temporal_window) / self.temporal_stride))
        self.max_idx = self.spatial_length * self.temporal_length
        # ============= Prepare the Task Attributes =================
        self.history_length = self.task_config.history_length
        self.prediction_length = self.task_config.prediction_length
        self.num_his_points = self.task_config.num_his_points
        self.num_waypoints = self.task_config.num_waypoints

    def get_scene_data(self, spatial_start, spatial_end, temporal_start ,temporal_end):
        
        # time stamp is the index, use the index to slice the data
        scene_data = self.data_df.loc[temporal_start:temporal_end].copy()
        scene_data.loc[:, 'timestamp']= scene_data.index
        scene_data.reset_index(drop=True, inplace=True)
        scene_data = scene_data[(scene_data['x_position'].between(spatial_start, spatial_end))]
        return scene_data

    def idx2range(self, idx):
        
        spatial_idx = idx % self.spatial_length
        temporal_idx = idx // self.spatial_length
        spatial_start = self.x_min + spatial_idx * self.spatial_stride
        spatial_end = spatial_start + self.spatial_window - 1
        temporal_start = self.time_min + temporal_idx * self.temporal_stride
        temporal_end = temporal_start + self.temporal_window - 1

        return spatial_start, spatial_end, temporal_start, temporal_end

    def get_feature_dics(self, idx):
        spatial_start, spatial_end, temporal_start ,temporal_end= self.idx2range(idx)
        
        # Determine prv spatial range
        prv_spatial_start = spatial_start - self.spatial_window if spatial_start - self.spatial_window >= self.x_min else spatial_start
        prv_spatial_end   = spatial_end - self.spatial_window if spatial_start - self.spatial_window >= self.x_min else spatial_start

        # Determine next spatial range
        nxt_spatial_start = spatial_start + self.spatial_window if spatial_start + self.spatial_window <= self.x_max else spatial_end
        nxt_spatial_end   = spatial_end + self.spatial_window if spatial_start + self.spatial_window <= self.x_max else spatial_end
            
        prv_feature_dic = self.get_feature_dic(prv_spatial_start, prv_spatial_end, temporal_start ,temporal_end)
        cur_feature_dic = self.get_feature_dic(spatial_start, spatial_end, temporal_start ,temporal_end)
        nxt_feature_dic = self.get_feature_dic(nxt_spatial_start, nxt_spatial_end, temporal_start ,temporal_end)

        return prv_feature_dic, cur_feature_dic, nxt_feature_dic
    
    def convert(self, coordinates):

        # coordinates (Number of Agents, timestamps)
        coordinates = np.concatenate(
                        [coordinates[:,:self.history_length][:, ::(self.history_length//self.num_his_points)], 
                        coordinates[:,self.history_length: self.history_length + self.prediction_length][:,::(self.prediction_length // self.num_waypoints)]],
                        axis=1)

        return coordinates
    
    def get_feature_dic(self, spatial_start, spatial_end, temporal_start, temporal_end, threshold_to_keep_percent=0.4):
        # Step 1: Get the scene data
        scene_data = self.get_scene_data(spatial_start, spatial_end, temporal_start ,temporal_end)
        # Compute timestamp indices relative to temporal_start for the entire DataFrame
        scene_data['timestamp_idx'] = (scene_data['timestamp'] - temporal_start).astype(np.int32)

        # Aggregate counts of valid history and prediction timestamps per agent (_id)
        group_counts = scene_data.groupby('_id')['timestamp_idx'].agg(
            #count_his=lambda x: (x < self.history_length).sum(),
            count_pred=lambda x: ((x >= self.history_length) & (x < self.history_length + self.prediction_length)).sum()
        )

        # Calculate the valid percentage for history and prediction for each agent
        #group_counts['valid_his_percent'] = group_counts['count_his'] / self.history_length
        group_counts['valid_pred_percent'] = group_counts['count_pred'] / self.prediction_length

        # Identify agents that meet the threshold criteria for both history and prediction
        valid_ids = group_counts[
            #(group_counts['valid_his_percent'] >= threshold_to_keep_percent) & 
            (group_counts['valid_pred_percent'] >= threshold_to_keep_percent)
        ].index

        # Filter the original scene_data to keep only rows with valid agent IDs
        #scene_data = scene_data[scene_data['_id'].isin(valid_ids)]

        vehicles_num = scene_data['_id'].nunique()
        vehicles_id = scene_data['_id'].unique()
        vehicles_id_dic = {_id: i - 1 for i, _id in enumerate(vehicles_id)}
        vehicles_length = np.zeros(vehicles_num, dtype=np.float32)
        vehicles_width = np.zeros(vehicles_num, dtype=np.float32)
        vehicles_height = np.zeros(vehicles_num)
        vehicles_direction = np.zeros(vehicles_num, dtype=np.float32)
        vehicles_class = np.zeros(vehicles_num)
        shape = (vehicles_num, self.temporal_window)
        vehicles_x = np.zeros(shape, dtype=np.float32)
        vehicles_y = np.zeros(shape, dtype=np.float32)
        vehicles_timestamp = np.zeros(shape, dtype=np.int32)
        vehicles_x_velocity = np.zeros(shape, dtype=np.float32)
        vehicles_y_velocity = np.zeros(shape, dtype=np.float32)
        vehicles_yaw_angle = np.zeros(shape, dtype=np.float32)
        for _id, group in scene_data.groupby('_id'):
            
            idx = vehicles_id_dic[_id]
            vehicle_length_cur = group['length'].iloc[0]
            vehicle_width_cur = group['width'].iloc[0]
            vehicle_height_cur = group['height'].iloc[0]
            vehicle_direction_cur = group['direction'].iloc[0]
            # skip the vehicle if the length or width is missing
            if np.isnan(vehicle_length_cur) or np.isnan(vehicle_width_cur):
                continue
            timestamp = group['timestamp'].values
            timestamp_idx = (timestamp - temporal_start).astype(np.int32)
            vehicle_x_cur = group['x_position'].values - spatial_start
            vehicle_y_cur = group['y_position'].values
            vehicle_x_velocity_cur = (group['x_position'].values - group['x_position'].shift(1).values) / (timestamp - np.roll(timestamp, shift=1))
            vehicle_y_velocity_cur = (group['y_position'].values - group['y_position'].shift(1).values) / (timestamp - np.roll(timestamp, shift=1))
            # fill in the values if its null
            vehicle_x_cur = np.nan_to_num(vehicle_x_cur)
            vehicle_y_cur = np.nan_to_num(vehicle_y_cur)
            vehicle_x_velocity_cur = np.nan_to_num(vehicle_x_velocity_cur)
            vehicle_y_velocity_cur = np.nan_to_num(vehicle_y_velocity_cur)
            # calculate yaw angle from x_velocity and y_velocity
            vehicle_yaw_angle_cur = np.arctan2(vehicle_y_velocity_cur, vehicle_x_velocity_cur)       
            vehicle_yaw_angle_cur = np.nan_to_num(vehicle_yaw_angle_cur)
            # skip the vehicle if the yaw angle is abnormal (yaw angle should be with in +- 30 degree)
            if np.abs(np.arctan(vehicle_y_velocity_cur, vehicle_x_velocity_cur)).max() > np.pi / 4:
                continue
            
            vehicles_length[idx] = vehicle_length_cur
            vehicles_width[idx] = vehicle_width_cur
            vehicles_height[idx] = vehicle_height_cur
            vehicles_direction[idx] = vehicle_direction_cur
            vehicles_timestamp[idx, timestamp_idx] = timestamp
            vehicles_x[idx, timestamp_idx] = vehicle_x_cur
            vehicles_y[idx, timestamp_idx] = vehicle_y_cur
            timestamp_shifted = np.roll(vehicles_timestamp[idx], shift=1, axis=0)
            valid_mask = (vehicles_timestamp[idx] > 0) & (timestamp_shifted > 0)
            vehicles_x_velocity[idx, timestamp_idx] = vehicle_x_velocity_cur
            vehicles_y_velocity[idx, timestamp_idx] = vehicle_y_velocity_cur
            vehicles_x_velocity[idx, ~valid_mask] = 0
            vehicles_y_velocity[idx, ~valid_mask] = 0
            # calculate yaw angle from x_velocity and y_velocity
            vehicles_yaw_angle[idx, timestamp_idx] = vehicle_yaw_angle_cur

        # fill the values if its null
        vehicles_length = np.nan_to_num(vehicles_length)
        vehicles_width = np.nan_to_num(vehicles_width)
        vehicles_direction = np.nan_to_num(vehicles_direction)
        vehicles_x = np.nan_to_num(vehicles_x)
        vehicles_y = np.nan_to_num(vehicles_y)
        vehicles_timestamp = np.nan_to_num(vehicles_timestamp)
        vehicles_x_velocity = np.nan_to_num(vehicles_x_velocity)
        vehicles_y_velocity = np.nan_to_num(vehicles_y_velocity)
        vehicles_yaw_angle = np.nan_to_num(vehicles_yaw_angle)


        # Step 4: Prepare the result dictionary
        result_dict = {
            # 'scene_id': idx,
            # '_id': vehicles_id,
            'num_vehicles': vehicles_num,
            'timestamp': self.convert(vehicles_timestamp),
            'length': vehicles_length,
            'width': vehicles_width,
            # 'height': vehicles_height,
            'direction': vehicles_direction,
            'class':vehicles_class,
            'x_position': self.convert(vehicles_x),
            'y_position': self.convert(vehicles_y),
            'x_velocity': self.convert(vehicles_x_velocity),
            'y_velocity': self.convert(vehicles_y_velocity),
            'yaw_angle': self.convert(vehicles_yaw_angle),
            # 'start_pos': spatial_start,
            # 'start_time': temporal_start,
        }

        return result_dict
    
    def get_trajectories(self, feature_dic):
        # Get the observed and occluded trajectories
        # observed_trajectories: (Nobs, T, D)
        # occluded_trajectories: (Nocc, T, D)
        
        num_time_steps = self.num_his_points + self.num_waypoints
        observed_idx = (feature_dic['timestamp'][...,self.num_his_points - 1] > 0)
        
        vector_features_list =  self.vector_features_list
        node_features_list = self.node_features_list
    
        vector_features_observed = np.stack([feature_dic[feature][observed_idx][:, None].repeat(num_time_steps, axis = 1) for feature in vector_features_list], axis = -1)
        vector_features_occluded = np.stack([feature_dic[feature][~observed_idx][:, None].repeat(num_time_steps, axis = 1) for feature in vector_features_list], axis = -1)
        node_features_observed = np.stack([feature_dic[feature][observed_idx] for feature in node_features_list], axis = -1)
        node_features_occluded = np.stack([feature_dic[feature][~observed_idx] for feature in node_features_list], axis = -1)
        observed_trajectories = np.concatenate([node_features_observed, vector_features_observed], axis = -1)
        occluded_trajectories = np.concatenate([node_features_occluded, vector_features_occluded], axis = -1)

        observed_types = feature_dic['class'][observed_idx][:, None].repeat(num_time_steps, axis = 1)
        occluded_types = feature_dic['class'][~observed_idx][:, None].repeat(num_time_steps, axis = 1)
        # the vehicle class is concatenated to the end of the trajectory
    
        return observed_trajectories, occluded_trajectories, observed_types, occluded_types
    

    def get_output_dic(self, feature_dic):

        occluded_occupancy_map, observed_occupancy_map, flow_map = self.grid_map_helper.get_map_flow(feature_dic)
        observed_trajectories, occluded_trajectories, observed_types, occluded_types = self.get_trajectories(feature_dic)
        result_dic = {
            'occluded_occupancy_map': occluded_occupancy_map, # H,W,T,1
            'observed_occupancy_map': observed_occupancy_map, # H,W,T,1
            'flow_map': flow_map, # H,W,(T-1),2
            'observed_trajectories': observed_trajectories, # (Nobs, T, D)
            'occluded_trajectories': occluded_trajectories, # (Nocc, T, D)
        }
        output_dic = typing.DefaultDict(dict)
        output_dic['his/occluded_occupancy_map'] = result_dic['occluded_occupancy_map'][..., :self.num_his_points]
        output_dic['pred/occluded_occupancy_map'] = result_dic['occluded_occupancy_map'][..., self.num_his_points: self.num_his_points + self.num_waypoints]
        output_dic['his/observed_occupancy_map'] = result_dic['observed_occupancy_map'][..., :self.num_his_points]
        output_dic['pred/observed_occupancy_map'] = result_dic['observed_occupancy_map'][..., self.num_his_points: self.num_his_points + self.num_waypoints]
        output_dic['his/flow_map'] = result_dic['flow_map'][..., :-self.num_waypoints, :]
        output_dic['pred/flow_map'] = result_dic['flow_map'][..., -self.num_waypoints:, :]
        output_dic['his/observed_trajectories'] = result_dic['observed_trajectories'][..., :self.num_his_points, :]
        output_dic['pred/observed_trajectories'] = result_dic['observed_trajectories'][..., self.num_his_points: self.num_his_points + self.num_waypoints, :]
        output_dic['his/occluded_trajectories'] = result_dic['occluded_trajectories'][..., :self.num_his_points, :]
        output_dic['pred/occluded_trajectories'] = result_dic['occluded_trajectories'][..., self.num_his_points: self.num_his_points + self.num_waypoints, :]
        all_occupancy_map = result_dic['occluded_occupancy_map'] + result_dic['observed_occupancy_map']
        all_occupancy_map = np.clip(all_occupancy_map, 0, 1)
        output_dic['flow_origin_occupancy_map'] = all_occupancy_map[...,self.num_his_points - 1: self.num_his_points -1 + self.num_waypoints]
        
        output_dic['observed_types'] = observed_types
        output_dic['occluded_types'] = occluded_types
        
        return output_dic
    
    def process_idx(self, idx, cur_threshold_to_keep_num=10, adj_threshold_to_keep_num=10):
        

        # Get feature dictionaries and add occupancy and flow maps
        prv_feature_dic, cur_feature_dic, nxt_feature_dic = self.get_feature_dics(idx)
        if cur_feature_dic['num_vehicles'] <= adj_threshold_to_keep_num or prv_feature_dic['num_vehicles'] <= cur_threshold_to_keep_num or nxt_feature_dic['num_vehicles'] <= adj_threshold_to_keep_num:
            #print(f'Skipping scene {idx} due to low vehicle count.')
            #print(f'cur: {cur_feature_dic["num_vehicles"]}, prv: {prv_feature_dic["num_vehicles"]}, nxt: {nxt_feature_dic["num_vehicles"]}')
            return
        points_per_vehicle_history = (np.sum(cur_feature_dic['timestamp'][:, :self.num_his_points]!=0) / cur_feature_dic['num_vehicles'])/self.num_his_points
        #points_per_vehicle_future = (np.sum(cur_feature_dic['timestamp'][:, self.num_his_points: self.num_his_points + self.num_waypoints]!=0) / cur_feature_dic['num_vehicles'])/self.num_waypoints
        # if the number of valid points per vehicle is less than threshold_to_keep, skip the scene
        if points_per_vehicle_history < 0.3:
            #print(f'Skipping scene {idx} due to low points per vehicle.')
            #print(f'points_per_vehicle_history: {points_per_vehicle_history}')
            return
        output_dic = {
            'prv': self.get_output_dic(prv_feature_dic),
            'cur': self.get_output_dic(cur_feature_dic),
            'nxt': self.get_output_dic(nxt_feature_dic),
        }
        print(f'Saving scene {idx}...')
        np.save(f'{self.generated_data_path}/scene_{idx}', output_dic)

class I24MotionDatasetPreprocessor():
    
    def __init__(self, dataset_config):
        self.dataset_config = dataset_config
        self.num_scenes_to_process_per_file = dataset_config.num_scenes_to_process_per_file
        self.num_threads = 30
         # ============= Load Paths =================
        self.paths = dataset_config.paths
        self.processed_data_path = self.paths.processed_data_path

    def process_file(self, data_file_path):
        data_file = I24MotionDatasetFile(data_file_path, dataset_config)
        num_scenes_to_process = min(data_file.max_idx, self.num_scenes_to_process_per_file)
        for idx in range(1000):
            data_file.process_idx(idx)
        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
        #     # Wrap tqdm around the executor to show progress
        #     list(tqdm(executor.map(data_file.process_idx, range(self.num_scenes_to_process_per_file)), total=num_scenes_to_process))

    def process_files(self,):
        data_files = get_files_with_extension(self.processed_data_path, '.parquet')
        for data_file in data_files:
            self.process_file(data_file)






if __name__ == '__main__':
    dataset_config = load_config("configs/dataset_configs/I24Motion_config.py")
    # process_raw_json2csv(dataset_config)
    preprocessor = I24MotionDatasetPreprocessor(dataset_config)
    preprocessor.process_files()









import pandas as pd
import numpy as np
from utils.file_utils import get_config
from utils.occ_flow_utils import GridMap
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
config = get_config('configs/config_12.yaml')

grid_map = GridMap(config)
start_pos = config.data_attributes.start_position * 5280
end_pos = config.data_attributes.end_position * 5280  # mile to feet
data_df = pd.read_parquet(config.paths.raw_data + '/63858a2cfb3ff533c12df166.parquet')
x_min = max(data_df['x_position'].min(), start_pos)
x_max = data_df['x_position'].max()
time_min = max(data_df['timestamp'].min(), 0)
time_max = data_df['timestamp'].max()
data_df.set_index('timestamp', inplace=True)
data_df.sort_index(inplace=True)
print('loaded data')
spatial_stride = config.preprocessing.spatial_stride
temporal_stride = config.preprocessing.temporal_stride
spatial_window = config.preprocessing.spatial_window
temporal_window = config.preprocessing.temporal_window
spatial_length = int(np.floor((x_max - x_min - spatial_window) / spatial_stride))
temporal_length = int(np.floor((time_max - time_min - temporal_window) / temporal_stride))

history_length = config.task_config.history_length
prediction_length = config.task_config.prediction_length
num_waypoints = config.task_config.num_waypoints
num_his_points = config.task_config.num_his_points

def get_scene_data(idx_list):
    prv_idx, cur_idx, nxt_idx = idx_list
    prv_spatial_start, _, prv_temporal_start, prv_temporal_end = idx2range(prv_idx)
    _, nxt_spatial_end, _, _ = idx2range(nxt_idx)
    
    # time stamp is the index, use the index to slice the data
    scene_data = data_df.loc[prv_temporal_start:prv_temporal_end].copy()
    scene_data.loc[:, 'timestamp']= scene_data.index
    scene_data.reset_index(drop=True, inplace=True)
    scene_data = scene_data[(scene_data['x_position'].between(prv_spatial_start, nxt_spatial_end))]
    return scene_data

def idx2range(idx):
    spatial_idx = idx % spatial_length
    temporal_idx = idx // spatial_length
    spatial_start = x_min + spatial_idx * spatial_stride
    spatial_end = spatial_start + spatial_window - 1
    temporal_start = time_min + temporal_idx * temporal_stride
    temporal_end = temporal_start + temporal_window - 1
    return spatial_start, spatial_end, temporal_start, temporal_end


def get_feature_dics(idx_list):
    prv_idx, cur_idx, nxt_idx = idx_list
    # prv_time = time.time()
    scene_data = get_scene_data(idx_list)
    # print(f'get_scene_data time: {time.time() - prv_time}')
    prv_feature_dic = get_feature_dic(scene_data, prv_idx)
    cur_feature_dic = get_feature_dic(scene_data, cur_idx)
    nxt_feature_dic = get_feature_dic(scene_data, nxt_idx)
    return prv_feature_dic, cur_feature_dic, nxt_feature_dic

@DeprecationWarning
def add_occ_flow(feature_dic):
    occluded_occupancy_map, observed_occupancy_map, flow_map = grid_map.get_map_flow(feature_dic)
    feature_dic['occluded_occupancy_map'] = occluded_occupancy_map.astype(np.float32)
    feature_dic['observed_occupancy_map'] = observed_occupancy_map.astype(np.float32)
    feature_dic['flow_map'] = flow_map.astype(np.float32)
    return feature_dic 

def convert(coordinates):
    # coordinates (Na, timestamps)
    coordinates = np.concatenate([coordinates[:,:history_length][:, ::(history_length//num_his_points)], 
              coordinates[:,history_length: history_length + prediction_length][:,::(prediction_length // num_waypoints)]]
             ,axis=1
             )
    
    return coordinates

def get_feature_dic(scene_data, scene_idx):
    spatial_start, spatial_end, temporal_start, temporal_end = idx2range(scene_idx)
    scene_data = scene_data[(scene_data['x_position'].between(spatial_start, spatial_end))]
    vehicles_num = scene_data['_id'].nunique()
    vehicles_id = scene_data['_id'].unique()
    vehicles_id_dic = {_id: i - 1 for i, _id in enumerate(vehicles_id)}
    vehicles_length = np.zeros(vehicles_num, dtype=np.float32)
    vehicles_width = np.zeros(vehicles_num, dtype=np.float32)
    vehicles_height = np.zeros(vehicles_num)
    vehicles_direction = np.zeros(vehicles_num, dtype=np.float32)
    vehicles_class = np.zeros(vehicles_num)
    shape = (vehicles_num, temporal_window)
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
        'timestamp': convert(vehicles_timestamp),
        'length': vehicles_length,
        'width': vehicles_width,
        # 'height': vehicles_height,
        'direction': vehicles_direction,
        'class':vehicles_class,
        'x_position': convert(vehicles_x),
        'y_position': convert(vehicles_y),
        'x_velocity': convert(vehicles_x_velocity),
        'y_velocity': convert(vehicles_y_velocity),
        'yaw_angle': convert(vehicles_yaw_angle),
        # 'start_pos': spatial_start,
        # 'start_time': temporal_start,
    }
    
    return result_dict

def get_trajectories(data, config):
    
    num_time_steps = config.task_config.num_his_points + config.task_config.num_waypoints
    observed_idx = (data['timestamp'][...,config.task_config.num_his_points - 1] > 0)
    
    # 'cur/meta/length', 'cur/meta/width', 'cur/meta/class', 'cur/meta/direction'
    vector_features_list = ['length', 'width', 'class', 'direction']
    node_features_list = ['timestamp', 'x_position', 'y_position', 'x_velocity', 'y_velocity', 'yaw_angle']

    vector_features_observed = np.stack([data[feature][observed_idx][:, None].repeat(num_time_steps, axis = 1) for feature in vector_features_list], axis = -1)
    vector_features_occluded = np.stack([data[feature][~observed_idx][:, None].repeat(num_time_steps, axis = 1) for feature in vector_features_list], axis = -1)
    node_features_observed = np.stack([data[feature][observed_idx] for feature in node_features_list], axis = -1)
    node_features_occluded = np.stack([data[feature][~observed_idx] for feature in node_features_list], axis = -1)
    observed_trajectories = np.concatenate([node_features_observed, vector_features_observed], axis = -1)
    occluded_trajectories = np.concatenate([node_features_occluded, vector_features_occluded], axis = -1)
    return observed_trajectories, occluded_trajectories

import typing
import concurrent.futures
def process_idx(idx):
    spatial_idx = idx % spatial_length
    if spatial_idx == 0:
        prv_idx = idx
    else:
        prv_idx = idx - 1
    if spatial_idx == spatial_length - 1:
        nxt_idx = idx
    else:
        nxt_idx = idx + 1
    idx_list = [prv_idx, idx, nxt_idx]
    
    # Get feature dictionaries and add occupancy and flow maps
    prv_feature_dic, cur_feature_dic, nxt_feature_dic = get_feature_dics(idx_list)
    if cur_feature_dic['num_vehicles'] <= 2:
        return
    points_per_vehicle_history = (np.sum(cur_feature_dic['timestamp'][:, :num_his_points]!=0) / cur_feature_dic['num_vehicles'])/num_his_points
    points_per_vehicle_future = (np.sum(cur_feature_dic['timestamp'][:, num_his_points: num_his_points + num_waypoints]!=0) / cur_feature_dic['num_vehicles'])/num_waypoints
    # if the number of valid points per vehicle is less than 0.4, skip the scene
    if points_per_vehicle_history < 0.4 or points_per_vehicle_future < 0.4:
        return

    occluded_occupancy_map, observed_occupancy_map, flow_map = grid_map.get_map_flow(cur_feature_dic)
    observed_trajectories, occluded_trajectories = get_trajectories(cur_feature_dic, config)
    result_dic = {
        'occluded_occupancy_map': occluded_occupancy_map, # H,W,T,1
        'observed_occupancy_map': observed_occupancy_map, # H,W,T,1
        'flow_map': flow_map, # H,W,(T-1),2
        'observed_trajectories': observed_trajectories, # (Nobs, T, D)
        'occluded_trajectories': occluded_trajectories, # (Nocc, T, D)
    }
    
    feature_dic = typing.DefaultDict(dict)
    feature_dic['his/occluded_occupancy_map'] = result_dic['occluded_occupancy_map'][..., :num_his_points]
    feature_dic['pred/occluded_occupancy_map'] = result_dic['occluded_occupancy_map'][..., num_his_points: num_his_points + num_waypoints]
    feature_dic['his/observed_occupancy_map'] = result_dic['observed_occupancy_map'][..., :num_his_points]
    feature_dic['pred/observed_occupancy_map'] = result_dic['observed_occupancy_map'][..., num_his_points: num_his_points + num_waypoints]
    feature_dic['his/flow_map'] = result_dic['flow_map'][..., :-num_waypoints, :]
    feature_dic['pred/flow_map'] = result_dic['flow_map'][..., -num_waypoints:, :]
    feature_dic['his/observed_trajectories'] = result_dic['observed_trajectories'][..., :num_his_points, :]
    feature_dic['pred/observed_trajectories'] = result_dic['observed_trajectories'][..., num_his_points: num_his_points + num_waypoints, :]
    feature_dic['his/occluded_trajectories'] = result_dic['occluded_trajectories'][..., :num_his_points, :]
    feature_dic['pred/occluded_trajectories'] = result_dic['occluded_trajectories'][..., num_his_points: num_his_points + num_waypoints, :]
    all_occupancy_map = result_dic['occluded_occupancy_map'] + result_dic['observed_occupancy_map']
    all_occupancy_map = np.clip(all_occupancy_map, 0, 1)
    feature_dic['flow_origin_occupancy_map'] = all_occupancy_map[...,num_his_points - 1: num_his_points -1 + num_waypoints]
    np.save(f'{config.paths.processed_data}/scene_{idx}', feature_dic)

# # # Number of threads to use
num_threads = 30
total_len = spatial_length * temporal_length
print(total_len)
total_len = 1500000
# Create ThreadPoolExecutor to parallelize the processing of idx
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Wrap tqdm around the executor to show progress
    list(tqdm(executor.map(process_idx, range(total_len)), total=total_len))

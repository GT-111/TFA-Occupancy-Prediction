import pandas as pd
import numpy as np
from utils.file_utils import get_config
from dataset.occ_flow_utils import GridMap
from tqdm import tqdm

config = get_config()

grid_map = GridMap(config)
start_pos = config.preprocess.start_pos * 5280
end_pos = config.preprocess.end_pos * 5280  # mile to feet
data_df = pd.read_parquet(config.dataset.processed_data)
print('loaded data')
x_min = max(data_df['x_position'].min(), start_pos)
x_max = data_df['x_position'].max()
time_min = max(data_df['timestamp'].min(), 0)
time_max = data_df['timestamp'].max()
spatial_stride = config.preprocess.spatial_stride
temporal_stride = config.preprocess.temporal_stride
spatial_window = config.preprocess.spatial_window
temporal_window = config.preprocess.temporal_window
spatial_length = int(np.floor((x_max - x_min - spatial_window) / spatial_stride))
temporal_length = int(np.floor((time_max - time_min - temporal_window) / temporal_stride))
total_len = spatial_length * temporal_length
his_len = config.dataset.his_len
pred_len = config.dataset.pred_len
def get_scene_data(idx_list):
    prv_idx, cur_idx, nxt_idx = idx_list
    
    
    prv_spatial_start, _, prv_temporal_start, prv_temporal_end = idx2range(prv_idx)
    _, nxt_spatial_end, _, _ = idx2range(nxt_idx)
    
    scene_data = data_df[(data_df['timestamp'].between(prv_temporal_start, prv_temporal_end))]
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

def add_occ_flow(feature_dic):
    occ_map, flow_map = grid_map.get_map_flow(feature_dic)
    feature_dic['occupancy_map'] = occ_map
    feature_dic['flow_map'] = flow_map
    return feature_dic

def get_feature_dics(idx_list):
    prv_idx, cur_idx, nxt_idx = idx_list
    scene_data = get_scene_data(idx_list)
    
    prv_feature_dic, cur_feature_dic, nxt_feature_dic = get_feature_dic(scene_data, prv_idx), get_feature_dic(scene_data, cur_idx), get_feature_dic(scene_data, nxt_idx)
    return prv_feature_dic, cur_feature_dic, nxt_feature_dic
    
def get_feature_dic(scene_data, idx):
    spatial_start, spatial_end, temporal_start, temporal_end = idx2range(idx)
    scene_data = scene_data[(scene_data['x_position'].between(spatial_start, spatial_end))]
    vehicles_num = scene_data['_id'].nunique()
    vehicles_id = scene_data['_id'].unique()
    vehicles_id_dic = {_id: i - 1 for i, _id in enumerate(vehicles_id)}
    vehicles_length = np.zeros(vehicles_num, dtype=np.float16)
    vehicles_width = np.zeros(vehicles_num, dtype=np.float16)
    vehicles_height = np.zeros(vehicles_num)
    vehicles_direction = np.zeros(vehicles_num, dtype=np.float16)
    vehicles_class = np.zeros(vehicles_num)
    shape = (vehicles_num, temporal_window)
    vehicles_x = np.zeros(shape, dtype=np.float16)
    vehicles_y = np.zeros(shape, dtype=np.float16)
    vehicles_timestamp = np.zeros(shape, dtype=np.int16)
    vehicles_x_velocity = np.zeros(shape, dtype=np.float16)
    vehicles_y_velocity = np.zeros(shape, dtype=np.float16)
    vehicles_yaw_angle = np.zeros(shape, dtype=np.float16)
    for _id, group in scene_data.groupby('_id'):
        idx = vehicles_id_dic[_id]
        vehicles_length[idx] = group['length'].iloc[0]
        vehicles_width[idx] = group['width'].iloc[0]
        vehicles_height[idx] = group['height'].iloc[0]
        vehicles_direction[idx] = group['direction'].iloc[0]
        timestamp = group['timestamp'].values
        timestamp_idx = (timestamp - temporal_start).astype(np.int32)
        vehicles_timestamp[idx, timestamp_idx] = timestamp
        vehicles_x[idx, timestamp_idx] = group['x_position'].values - spatial_start
        vehicles_y[idx, timestamp_idx] = group['y_position'].values
        timestamp_shifted = np.roll(vehicles_timestamp[idx], shift=1, axis=0)
        valid_mask = (vehicles_timestamp[idx] > 0) & (timestamp_shifted > 0)
        vehicles_x_velocity[idx, timestamp_idx] = (group['x_position'].values - group['x_position'].shift(1).values) / (timestamp - timestamp_shifted[timestamp_idx])
        vehicles_y_velocity[idx, timestamp_idx] = (group['y_position'].values - group['y_position'].shift(1).values) / (timestamp - timestamp_shifted[timestamp_idx])
        vehicles_x_velocity[idx, ~valid_mask] = 0
        vehicles_y_velocity[idx, ~valid_mask] = 0
        # calculate yaw angle from x_velocity and y_velocity
        vehicles_yaw_angle[idx, timestamp_idx] = np.arctan2(vehicles_y_velocity[idx, timestamp_idx], vehicles_x_velocity[idx, timestamp_idx])
        
    # Step 4: Prepare the result dictionary
    result_dict = {
        # 'scene_id': idx,
        # '_id': vehicles_id,
        'num_vehicles': vehicles_num,
        'timestamp': vehicles_timestamp,
        'length': vehicles_length,
        'width': vehicles_width,
        # 'height': vehicles_height,
        'direction': vehicles_direction,
        'class':vehicles_class,
        'x_position': vehicles_x,
        'y_position': vehicles_y,
        'x_velocity': vehicles_x_velocity,
        'y_velocity': vehicles_y_velocity,
        'yaw_angle': vehicles_yaw_angle,
        'start_pos': spatial_start,
        'start_time': temporal_start,
    }
    return result_dict

import concurrent.futures
import typing

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
    # prv_feature_dic = add_occ_flow(prv_feature_dic)
    # cur_feature_dic = add_occ_flow(cur_feature_dic)
    # nxt_feature_dic = add_occ_flow(nxt_feature_dic)
    result_dic = {
        'prv': prv_feature_dic,
        'cur': cur_feature_dic,
        'nxt': nxt_feature_dic
    }
    # Create the feature dictionary to save
    # feature_dic = typing.DefaultDict(dict)
    # for dic_k, dic in {'prv':prv_feature_dic, 'cur': cur_feature_dic, 'nxt': nxt_feature_dic}.items():
    #     for k, v in dic.items():
    #         if k in ['timestamp', 'x_position', 'y_position', 'x_velocity', 'y_velocity', 'yaw_angle']:
    #             # (Num of Agents, Timestamp)
                
    #             feature_dic[dic_k + '/state/his/' + k] = v[:, :his_len]
    #             feature_dic[dic_k + '/state/pred/' + k] = v[: his_len: his_len + pred_len]
    #         # elif k in ['occupancy_map', 'flow_map']:
    #         #     # (Timestamp, H, W) -> Occ (Timestamp, H, W, 2) -> Flow
    #         #     feature_dic[dic_k + '/state/his/' + k] = v[:his_len,...]
    #         #     feature_dic[dic_k + '/state/pred/' + k] = v[his_len: his_len + pred_len,...]
    #             # pred_v = v[his_len: his_len + pred_len,...]
    #             # pred_v = pred_v.reshape(-1, pred_len//10, *pred_v.shape[1:]).sum(axis=0)
    #             # print(f'{k}, pred {pred_v.shape}')
                
    #         else:
    #             feature_dic[dic_k + '/meta/' + k] = v
    
    # Save the feature dictionary
    np.save(f'{config.paths.processed_data}/scene_{idx}', result_dic)

# # Number of threads to use
num_threads = 16

# Create ThreadPoolExecutor to parallelize the processing of idx
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Wrap tqdm around the executor to show progress
    list(tqdm(executor.map(process_idx, range(total_len)), total=total_len))


    
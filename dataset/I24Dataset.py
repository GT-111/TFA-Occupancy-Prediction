from utils.file_utils import get_npy_files
from utils.occ_flow_utils import GridMap
import typing
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
class I24Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.grid_map = GridMap(config)
        self.data_files = get_npy_files(config.paths.processed_data)[:config.dataloader_config.total_samples]
        
    def add_occ_flow(self, feature_dic):
        occluded_occupancy_map, observed_occupancy_map, flow_map = self.grid_map.get_map_flow(feature_dic)
        feature_dic['occluded_occupancy_map'] = occluded_occupancy_map
        feature_dic['observed_occupancy_map'] = observed_occupancy_map
        feature_dic['flow_map'] = flow_map
        return feature_dic 
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data_dic = np.load(self.data_files[idx], allow_pickle=True).item()
        
        # Create the feature dictionary to save
        feature_dic = data_dic

        return feature_dic
    
    
    class I24Dataset_df(Dataset):
        
        def __init__(self, config):
            self.config = config
            self.grid_map = GridMap(config)
            self.data_df = pd.read_parquet('63858a2cfb3ff533c12df166.parquet')
            self.start_pos = self.config.data_attributes.start_position * 5280
            self.end_pos = self.config.data_attributes.end_position * 5280  # mile to feet
            self.x_min = max(self.data_df['x_position'].min(), self.start_pos)
            self.x_max = self.data_df['x_position'].max()
            self.time_min = max(self.data_df['timestamp'].min(), 0)
            self.time_max = self.data_df['timestamp'].max()
            self.data_df.set_index('timestamp', inplace=True)
            self.data_df.sort_index(inplace=True)
            self.spatial_stride = config.preprocessing.spatial_stride
            self.temporal_stride = config.preprocessing.temporal_stride
            self.spatial_window = config.preprocessing.spatial_window
            self.temporal_window = config.preprocessing.temporal_window
            self.spatial_length = int(np.floor((self.x_max - self.x_min - self.spatial_window) / self.spatial_stride))
            self.temporal_length = int(np.floor((self.time_max - self.time_min - self.temporal_window) / self.temporal_stride))

            self.his_len = self.config.task_config.history_length
            self.pred_len = self.config.task_config.prediction_length
            
        def add_occ_flow(self, feature_dic):
            occluded_occupancy_map, observed_occupancy_map, flow_map = self.grid_map.get_map_flow(feature_dic)
            feature_dic['occluded_occupancy_map'] = occluded_occupancy_map
            feature_dic['observed_occupancy_map'] = observed_occupancy_map
            feature_dic['flow_map'] = flow_map
            return feature_dic 

        def get_scene_data(self, idx_list):
            prv_idx, cur_idx, nxt_idx = idx_list


            prv_spatial_start, _, prv_temporal_start, prv_temporal_end = self.idx2range(prv_idx)
            _, nxt_spatial_end, _, _ = self.idx2range(nxt_idx)

            scene_data = self.data_df[(self.data_df['timestamp'].between(prv_temporal_start, prv_temporal_end))]
            scene_data = scene_data[(scene_data['x_position'].between(prv_spatial_start, nxt_spatial_end))]

            scene_data = self.data_df.loc[prv_temporal_start:prv_temporal_end].copy()
            scene_data.loc[:, 'timestamp']= scene_data.index
            scene_data.reset_index(drop=True, inplace=True)
            scene_data = scene_data[(scene_data['x_position'].between(prv_spatial_start, nxt_spatial_end))]
            return scene_data

        def idx2range(self, idx):
            spatial_idx = idx % self.spatial_length
            temporal_idx = idx // self.spatial_length
            spatial_start = self.x_min + spatial_idx * self.spatial_stride
            spatial_end = spatial_start + self.spatial_window - 1
            temporal_start = self.time_min + temporal_idx * self.temporal_stride
            temporal_end = temporal_start + self.temporal_window - 1
            return spatial_start, spatial_end, temporal_start, temporal_end

        def add_occ_flow(self, feature_dic):
            occ_map, flow_map = self.grid_map.get_map_flow(feature_dic)
            feature_dic['occupancy_map'] = occ_map
            feature_dic['flow_map'] = flow_map
            return feature_dic

        def get_feature_dics(self, idx_list):
            prv_idx, cur_idx, nxt_idx = idx_list
            scene_data = self.get_scene_data(idx_list)
            prv_feature_dic, cur_feature_dic, nxt_feature_dic = self.get_feature_dic(scene_data, prv_idx), self.get_feature_dic(scene_data, cur_idx), self.get_feature_dic(scene_data, nxt_idx)
            return prv_feature_dic, cur_feature_dic, nxt_feature_dic
            
        def get_feature_dic(self, scene_data, idx):
            
            spatial_start, spatial_end, temporal_start, temporal_end = self.idx2range(idx)
            scene_data = scene_data[(scene_data['x_position'].between(spatial_start, spatial_end))]
            vehicles_num = scene_data['_id'].nunique()
            vehicles_id = scene_data['_id'].unique()
            vehicles_id_dic = {_id: i - 1 for i, _id in enumerate(vehicles_id)}
            vehicles_length = np.zeros(vehicles_num, dtype=np.float16)
            vehicles_width = np.zeros(vehicles_num, dtype=np.float16)
            vehicles_height = np.zeros(vehicles_num)
            vehicles_direction = np.zeros(vehicles_num, dtype=np.float16)
            vehicles_class = np.zeros(vehicles_num)
            shape = (vehicles_num, self.temporal_window)
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
        
        
        def __len__(self):
            
            return self.spatial_length * self.temporal_length

        def __getitem__(self, idx):
            spatial_idx = idx % self.spatial_length
            if spatial_idx == 0:
                prv_idx = idx
            else:
                prv_idx = idx - 1
            if spatial_idx == self.spatial_length - 1:
                nxt_idx = idx
            else:
                nxt_idx = idx + 1
            idx_list = [prv_idx, idx, nxt_idx]
            prv_feature_dic, cur_feature_dic, nxt_feature_dic = self.get_feature_dics(idx_list)

            result_dic = {
                'prv': prv_feature_dic,
                'cur': cur_feature_dic,
                'nxt': nxt_feature_dic
            }

            # Create the feature dictionary to save
            feature_dic = typing.DefaultDict(dict)
            for dic_k, dic in result_dic.items():
                dic = self.add_occ_flow(dic)
                for k, v in dic.items():
                    if k in ['timestamp', 'x_position', 'y_position', 'x_velocity', 'y_velocity', 'yaw_angle']:
                        # (Num of Agents, Timestamp)

                        feature_dic[dic_k + '/state/his/' + k] = v[:, :self.his_len]
                        feature_dic[dic_k + '/state/pred/' + k] = v[: self.his_len: self.his_len + self.pred_len]
                    elif k in ['occluded_occupancy_map', 'observed_occupancy_map', 'flow_map']:
                        # (Timestamp, H, W) -> Occ (Timestamp, H, W, 2) -> Flow
                        feature_dic[dic_k + '/state/his/' + k] = v[:self.his_len,...]
                        feature_dic[dic_k + '/state/pred/' + k] = v[self.his_len: self.his_len + self.pred_len,...]
                        # pred_v = v[his_len: his_len + pred_len,...]
                        # pred_v = pred_v.reshape(-1, pred_len//10, *pred_v.shape[1:]).sum(axis=0)
                        # print(f'{k}, pred {pred_v.shape}')
                    else:
                        feature_dic[dic_k + '/meta/' + k] = v

            return feature_dic


import numpy as np
import os
import ijson
import pandas as pd
import easydict as edict
import json
import pytz
from tqdm import tqdm
import logging
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import collections
import ijson

def get_json_files(raw_data_path = './raw_data'):

    files = []
    for root, _, file in os.walk(raw_data_path):
        for f in file:
            if f.endswith('.json'):
                files.append(os.path.join(root, f))

    return files

def get_config (config_file = './config.json'):
    with open(config_file) as f:
        config = edict.EasyDict(json.load(f))
    return config


config = get_config()

class MileMarker():
    def __init__(self, mile_marker_data: pd.Series):
        self.MM = mile_marker_data['MM']
        self.X_WGS84 = mile_marker_data['X_WGS84']
        self.Y_WGS84 = mile_marker_data['Y_WGS84']
        self.X_TN = mile_marker_data['X_TN']
        self.Y_TN = mile_marker_data['Y_TN']


class Scene():
    def __init__(self, scene_id, start_pos, spatial_window, start_time, temporal_window):
        self.scene_id = scene_id
        self.record_cnt = 0
        # the sensor herz is 25 hz, so the temporal dimension is temporal_window * 25
        self.temporal_window = temporal_window
        self.spatial_window = spatial_window
        self.start_pos = start_pos
        self.start_time = start_time
        self.result_dic = collections.defaultdict(list)
    def add_record(self, record, idx):
        # print(f'this is the {self.record_cnt} record, of {self.scene_id} scene with start_pos {self.start_pos} and start_time {self.start_time}')
        
        # Calculate local time indices for the given record indices
        local_time_idx = np.trunc((np.array(record['timestamp'])[idx] - (self.start_time)) * 25).astype(int)
        
        # Preallocate arrays based on temporal window size
        timestamp = np.zeros(self.temporal_window)
        x_position = np.zeros(self.temporal_window)
        y_position = np.zeros(self.temporal_window)
        valid = np.zeros(self.temporal_window)

        # Fill preallocated arrays with actual data
        timestamp[local_time_idx] = np.array(record['timestamp'])[idx]
        x_position[local_time_idx] = np.array(record['x_position'])[idx]
        y_position[local_time_idx] = np.array(record['y_position'])[idx]
        valid[local_time_idx] = 1

        # Append the relevant data fields to result dictionary
        for field in ['_id', 'length', 'width', 'height', 'coarse_vehicle_class', 'direction']:
            self.result_dic[field].append(record[field])

        # Add array data to the result dictionary
        self.result_dic['timestamp'].append(timestamp)
        self.result_dic['x_position'].append(x_position)
        self.result_dic['y_position'].append(y_position)
        self.result_dic['valid'].append(valid)

        self.record_cnt += 1

    def to_numpy(self):
        # savee the scene to numpy
        for key in self.result_dic.keys():
            self.result_dic[key] = np.array(self.result_dic[key]) if key != 'record_cnt' else self.record_cnt
        return self.result_dic





class I24Motion():

    def __init__(self, config: edict.EasyDict):
        self.config = config
        self.raw_data_files = get_json_files(self.config.paths.raw_data)
        self.mile_markers_data = pd.read_csv(os.path.join(self.config.paths.auxiliary_data, 'mile_marker_layer.csv'))
        self.grade_elevation = pd.read_csv(os.path.join(self.config.paths.auxiliary_data, 'grade_elevation.csv'))
        self.poles_data = pd.read_csv(os.path.join(self.config.paths.auxiliary_data, 'pole_layer.csv'))
        self.ramp_landmarks_data = pd.read_csv(os.path.join(self.config.paths.auxiliary_data, 'ramp_and_landmark_layer.csv'))
        self.duration_dict = np.load(os.path.join(self.config.paths.auxiliary_data, 'duration_dict.npy'), allow_pickle=True).item()
        self.initialize_mile_markers_dick()
        self.scene_cnt = 0
        self.uid_dic = {_ : _.split('/')[-1].split('__')[0] for _ in self.raw_data_files}
        

  
    
   
    
    def preprocess_all(self):
        for raw_data_file in self.raw_data_files:
            uid = self.uid_dic[raw_data_file]
            scenes = self.preprocess_file(raw_data_file)
            for pos_offset in self.scenes.keys():
                    for time_offset in self.scenes[pos_offset].keys():
                        scene = self.scenes[pos_offset][time_offset]
                        if scene.record_cnt == 0:
                            continue
                        scene_data = scene.to_numpy()
                        save_path = self.config.paths.processed_data
                        np.save(os.path.join(save_path, f'{uid}_{pos_offset}_{time_offset}.npy'), scene_data)
                        
                        print(f'{uid}_{pos_offset}_{time_offset}.npy saved!')
            break



    def preprocess_file(self, raw_data_file):
        start_pos = self.config.preprocess.start_pos * 5280 # mile to feet
        end_pos = self.config.preprocess.end_pos * 5280 # mile to feet
        scenes = collections.defaultdict(dict)
        scene_cnt = 0
        uid = self.uid_dic[raw_data_file]
        start_timestamp = self.duration_dict[uid]['start_timestamp']

        with open(raw_data_file, 'r') as input_file:
            parser = ijson.items(input_file, 'item', use_float=True)
            # for record in parser:
            #     timestamp_shifted = ((np.array(record['timestamp']) - start_timestamp))
            #     # if the record is before the start_pos, skip
            #     if np.min(np.array(record['x_position'])) < start_pos:
            #         continue
                
            #     pos_offset_array = (np.array(record['x_position']) - start_pos) // self.config.preprocess.spatial_stride
            #     time_offset_array = timestamp_shifted * 25 // self.config.preprocess.temporal_stride
            #     # print(timestamp_shifted * 25)
            #     for pos_offset in np.unique(pos_offset_array):
            #         pos_idx = np.where(pos_offset_array == pos_offset) 
            #         if pos_offset not in scenes:
            #             scenes[pos_offset] = {}
            #         # print(pos_offset * self.config.preprocess.spatial_stride + start_pos)
            #         # print(pos_offset * self.config.preprocess.spatial_stride + start_pos + self.config.preprocess.spatial_window)
            #         for time_offset in np.unique(time_offset_array):
            #             if time_offset not in scenes[pos_offset]:
            #                 scenes[pos_offset][time_offset] = Scene(self.scene_cnt, 
            #                                                         pos_offset * self.config.preprocess.spatial_stride + start_pos, 
            #                                                         self.config.preprocess.spatial_window, 
            #                                                         time_offset * self.config.preprocess.temporal_stride / 25 + start_timestamp, 
            #                                                         self.config.preprocess.temporal_window)
            #             time_idx = np.where(time_offset_array == time_offset)
            #             # get the data for this scene
            #             idx = np.intersect1d(pos_idx, time_idx)
            #             # if no data in this scene, skip
            #             if len(idx) == 0:
            #                 continue 
                        
            #             scenes[pos_offset][time_offset].add_record(record, idx)
            #             self.scene_cnt = self.scene_cnt + 1
            for record in tqdm(parser):
                timestamp_shifted = ((np.array(record['timestamp']) - start_timestamp))
                pos_offset_array = (np.array(record['x_position']) - start_pos) // self.config.preprocess.spatial_stride
                time_offset_array = (timestamp_shifted * 25) // self.config.preprocess.temporal_stride
                unique_pairs, pair_indices = np.unique(np.column_stack((pos_offset_array, time_offset_array)), axis=0, return_inverse=True)
                # print(unique_pairs)
                for i, (pos_offset, time_offset) in enumerate(unique_pairs):
                    # Get the indices for the current unique pair
                    pair_idx = np.where(pair_indices == i)[0]
                    # Initialize the scene if not already present
                    if time_offset not in scenes[pos_offset]:
                        scenes[pos_offset][time_offset] = Scene(scene_cnt, 
                                                                pos_offset * self.config.preprocess.spatial_stride + start_pos, 
                                                                self.config.preprocess.spatial_window, 
                                                                time_offset * self.config.preprocess.temporal_stride / 25 + start_timestamp, 
                                                                self.config.preprocess.temporal_window)
                    # Add record to the corresponding scene using vectorized indexing
                    # print(pair_idx)
                    scenes[pos_offset][time_offset].add_record(record, pair_idx)
                    scene_cnt += 1
                gc.collect()
        print(f'{uid} finished!')
        return scenes
    
  
# test 
dataset = I24Motion(config)
dataset.preprocess_all()

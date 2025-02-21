import os

from sympy import root
dataset = 'I24Motion'

# ============= Path ===================
dataset_root_dir = '/home/hg25079/Projects/MotionPrediction/Datasets/I24Motion/'
raw_data_path = 'raw_data/'
auxilary_data_path = 'auxiliary_data/'
processed_data_path = 'processed_data/'

# ============= Data Parameters =================
sample_frequency = 25 # the frequency of the input data
start_position = 58.5 # the start position of the input data in miles
end_position = 63.5 # the end position of the input data in miles
keys_to_use = ['_id', 'timestamp', 'x_position', 'y_position', 'length', 'width', 'height', 'direction', 'coarse_vehicle_class', 
                # 'road_segment_ids', # 'flags', # 'merged_ids', # 'fragment_ids', # 'fine_vehicle_class', # 'compute_node_id', # 'local_fragment_id', # 'starting_x', # 'first_timestamp', # 'configuration_id', # 'ending_x', # 'last_timestamp', # 'x_score', # 'y_score'
            ]
# ============= Preprocessing =================
spatial_stride: 70
spatial_window: 280
temporal_stride: 29
temporal_window: 116

# ============= Task General Parameters =================
history_length = 20
num_his_points = 10
prediction_length = 400
num_waypoints = 20

# ============= Task Parameters =================
occupancy_flow_map_height = 256
occupancy_flow_map_width = 256
vehicle_per_side_length: 48 # the number of points in the vehicle length direction
vehicle_per_side_width: 16 # the number of points in the vehicle width direction

# ============= Config ===================
config = dict(
    keys_to_use = keys_to_use,
    paths=dict(
        dataset_root_dir=dataset_root_dir,
        raw_data_path=os.path.join(dataset_root_dir, raw_data_path),
        auxilary_data_path=os.path.join(dataset_root_dir, auxilary_data_path),
        processed_data_path=os.path.join(dataset_root_dir, processed_data_path),
    ),
    attributes=dict(
        sample_frequency=sample_frequency,
    )
)
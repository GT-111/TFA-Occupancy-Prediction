import os

from sympy import root
dataset = 'I24Motion'

# ============= Path ===================
dataset_root_dir = '/home/hg25079/Projects/MotionPrediction/Datasets/I24Motion'
raw_data_path = 'raw_data/'
auxilary_data_path = 'auxiliary_data/'
processed_data_path = 'processed_data/'
generated_data_path = 'generated_data/'
num_scenes_to_process_per_file = 1000000
# ============= Data Parameters =================
sample_frequency = 25 # the frequency of the input data
start_position = 58.5 # the start position of the input data in miles
end_position = 63.5 # the end position of the input data in miles
keys_to_use = ['_id', 'timestamp', 'x_position', 'y_position', 'length', 'width', 'height', 'direction', 'coarse_vehicle_class', 
                # 'road_segment_ids', # 'flags', # 'merged_ids', # 'fragment_ids', # 'fine_vehicle_class', # 'compute_node_id', # 'local_fragment_id', # 'starting_x', # 'first_timestamp', # 'configuration_id', # 'ending_x', # 'last_timestamp', # 'x_score', # 'y_score'
            ]

# ============= Task General Parameters =================
history_length = 100
num_his_points = 10
prediction_length = 400
num_waypoints = 20

# ============= Occupancy Flow Map Parameters =================
occupancy_flow_map_height = 512
occupancy_flow_map_width = 96
vehicle_points_per_side_length = 36 # the number of points in the vehicle length direction
vehicle_points_per_side_width = 12 # the number of points in the vehicle width direction

spatial_window = 1400 # the size of the spatial window in feet
spatial_stride = spatial_window // 4
temporal_window = history_length + prediction_length # the size of the temporal window in seconds
temporal_stride = temporal_window // 4
# ============= Config ===================
config = dict(
    keys_to_use = keys_to_use,
    start_position=start_position,
    end_position=end_position,
    num_scenes_to_process_per_file=num_scenes_to_process_per_file,
    paths=dict(
        dataset_root_dir=dataset_root_dir,
        raw_data_path=os.path.join(dataset_root_dir, raw_data_path),
        auxilary_data_path=os.path.join(dataset_root_dir, auxilary_data_path),
        processed_data_path=os.path.join(dataset_root_dir, processed_data_path),
        generated_data_path=os.path.join(dataset_root_dir, generated_data_path),
    ),
    data_attributes=dict(
        sample_frequency=sample_frequency,
        start_position=start_position,
        end_position=end_position,
    ),
    occupancy_flow_map=dict(
        spatial_stride=spatial_stride,
        spatial_window=spatial_window,
        temporal_stride=temporal_stride,
        temporal_window=temporal_window,
        occupancy_flow_map_height=occupancy_flow_map_height,
        occupancy_flow_map_width=occupancy_flow_map_width,
        vehicle_points_per_side_length=vehicle_points_per_side_length,
        vehicle_points_per_side_width=vehicle_points_per_side_width,
    ),
    task=dict(
        history_length=history_length,
        num_his_points=num_his_points,
        prediction_length=prediction_length,
        num_waypoints=num_waypoints,
    ),

)
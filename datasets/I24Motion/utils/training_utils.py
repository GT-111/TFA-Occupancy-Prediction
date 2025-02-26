import torch
from typing import DefaultDict
def parse_data(data_dic, gpu_id, config):
    """
    Parse data from dataloader
    """
    
    input_dict = DefaultDict(dict)
    ground_truth_dict = DefaultDict(dict)
    input_keys = [
        # 'his/occluded_occupancy_map', 
        'his/observed_occupancy_map', 
        'his/flow_map', 
        'flow_origin_occupancy_map',
        'his/observed_agent_features', 
        'his/valid_mask',
        'agent_types', 
    ]
    groundtruth_keys = [
        'pred/occluded_occupancy_map', 
        'pred/observed_occupancy_map', 
        'pred/flow_map', 
        'pred/trajectories',
        'pred/valid_mask',
    ]
    for scene_key in ['prv', 'cur', 'nxt']:
    
        for key in input_keys:
            data = data_dic[key].to(gpu_id, dtype=torch.float32)
            if torch.isnan(data).any():
                data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
            input_dict[scene_key][key] = data

        for key in groundtruth_keys:
            data = data_dic[key].to(gpu_id, dtype=torch.float32)
            if torch.isnan(data).any():
                data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
            ground_truth_dict[scene_key][key] = data

        
    return input_dict, ground_truth_dict



def parse_outputs(outputs, num_waypoints):
    """
    Parse model outputs
    """
    raise NotImplementedError
    


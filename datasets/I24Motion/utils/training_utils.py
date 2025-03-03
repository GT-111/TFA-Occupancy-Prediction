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
            data = data_dic[scene_key][key].to(gpu_id, dtype=torch.float32)
            if torch.isnan(data).any():
                data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
            input_dict[scene_key][key] = data

        for key in groundtruth_keys:
            data = data_dic[scene_key][key].to(gpu_id, dtype=torch.float32)
            if torch.isnan(data).any():
                data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
            ground_truth_dict[scene_key][key] = data

        
    return input_dict, ground_truth_dict

def parse_data_vis(data_dic):
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
            data = data_dic[scene_key][key]
            if torch.isnan(data).any():
                data = torch.where(torch.isnan(data), torch.zeros_like(data), data)

            input_dict[scene_key][key] = data.numpy()
        for key in groundtruth_keys:
            data = data_dic[scene_key][key]
            if torch.isnan(data).any():
                data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
            ground_truth_dict[scene_key][key] = data.numpy()

        
    return input_dict, ground_truth_dict

def parse_outputs(outputs, train):
    """
    Parse model outputs
    """
    pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits, predicted_trajectories, predicted_trajectories_scores = outputs
    if train:
        return pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits, predicted_trajectories, predicted_trajectories_scores
    
    else:
        pred_observed_occupancy_logits = torch.sigmoid(pred_observed_occupancy_logits)
        pred_occluded_occupancy_logits = torch.sigmoid(pred_occluded_occupancy_logits)

        return pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits, predicted_trajectories, predicted_trajectories_scores
    
    
    


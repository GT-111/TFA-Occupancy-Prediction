import torch

def parse_data(data, gpu_id, config):
    """
    Parse data from dataloader
    """
    input_dict = {}
    ground_truth_dict = {}
    state_keys = [
            'his/occluded_occupancy_map', 
            'his/observed_occupancy_map', 
            'his/flow_map', 
            'his/observed_trajectories', 
            'his/occluded_trajectories', 
            'flow_origin_occupancy_map'
            ]
    state_to_predict_keys = [
        'pred/occluded_occupancy_map', 
        'pred/observed_occupancy_map', 
        'pred/flow_map', 
    ]
    # # for scene_key in ['prv', 'cur', 'nxt']:
    # for scene_key in ['cur']:
    #     for key in meta_array:
    #         meta_keys.append(f'{scene_key}/meta/{key}')
    #     for key in state:
    #         state_keys.append(f'{scene_key}/state/his/{key}')
    #         if key in state_to_predict:
    #             state_to_predict_keys.append(f'{scene_key}/state/pred/{key}')
    for key in state_keys:
        input_dict[key] = data[key].to(gpu_id, dtype=torch.float32)

    for key in state_to_predict_keys:
        ground_truth_dict[key] = data[key].to(gpu_id, dtype=torch.float32)
            
    for key, val in input_dict.items():
        if torch.isnan(val).any():
            input_dict[key] = torch.where(torch.isnan(val), torch.zeros_like(val), val)
    for key, val in ground_truth_dict.items():
        if torch.isnan(val).any():
            ground_truth_dict[key] = torch.where(torch.isnan(val), torch.zeros_like(val), val)
    
    return input_dict, ground_truth_dict

def save_checkpoint(model, optimizer, scheduler, epoch, path, global_step):
    """
    Save model checkpoint
    """
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path + f'/epoch_{epoch+1}.pth')
    
def parse_outputs(outputs, num_waypoints):
    """
    Parse model outputs
    """
    B, T, W, H, C = outputs.shape
    pred_observed_occupancy_logits = outputs[:, :, :, :, :1].reshape(B, num_waypoints, W, H, 1)
    pred_occluded_occupancy_logits = outputs[:, :, :, :, 1:2].reshape(B, num_waypoints, W, H, 1)
    pred_flow_logits = outputs[:, :, :, :, 2:].reshape(B, num_waypoints, W, H, 2)
    
    return pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits

def get_scene_vehicle_nums(data):
    """
    Get number of vehicles in each scene
    """
    scene_vehicle_nums = {}
    for scene_key in ['prv', 'cur', 'nxt']:
        scene_vehicle_nums[scene_key] = (data[f'{scene_key}/meta/num_vehicles'].shape[0])
    return scene_vehicle_nums
import torch
def parse_data(data, gpu_id):
    """
    Parse data from dataloader
    """
    input_dict = {}
    ground_truth_dict = {}
    state = ['timestamp', 'x_position', 'y_position', 'x_velocity', 'y_velocity', 'yaw_angle', 'occluded_occupancy_map', 'observed_occupancy_map', 'flow_map']
    state_to_predict = ['occluded_occupancy_map', 'observed_occupancy_map', 'flow_map']
    state_keys = []
    state_to_predict_keys = []
    for scene_key in ['prv', 'cur', 'nxt']:
        for key in state:
            state_keys.append(f'{scene_key}/state/his/{key}')
            if key in state_to_predict:
                state_to_predict_keys.append(f'{scene_key}/state/pred/{key}')
    for key in state_keys:
        input_dict[key] = data[key].to(gpu_id)
    for key in state_to_predict_keys:
        ground_truth_dict[key] = data[key].to(gpu_id)
        
    return input_dict, ground_truth_dict

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """
    Save model checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path + f'/epoch_{epoch+1}.pt')

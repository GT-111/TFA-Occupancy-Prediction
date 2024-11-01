import torch
def parse_data(data, gpu_id, config):
    """
    Parse data from dataloader
    """
    input_dict = {}
    ground_truth_dict = {}
    num_waypoints = config.task.num_waypoints
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
        if data[key].dim() == 4:
            # OCC
            batch_size, timestamps, height, width = data[key].shape
            ground_truth_dict[key] = data[key].reshape(batch_size, timestamps // num_waypoints, num_waypoints, height, width).sum(axis=1).to(gpu_id)
        elif data[key].dim() == 5:
            # FLOW
            batch_size, timestamps, height, width, num_channels= data[key].shape
            ground_truth_dict[key] = data[key].reshape(batch_size, timestamps // num_waypoints, num_waypoints, height, width, num_channels).sum(axis=1).to(gpu_id)
        
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
    
# from utils.file_utils import  get_config
# import torch
# gpu_id = torch.cuda.current_device()
# config = get_config()
# data = torch.load('data.pth')
# input_dict, ground_truth_dict = parse_data(data, gpu_id, config)
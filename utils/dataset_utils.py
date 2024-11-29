import torch
import numpy as np
from dataset.I24Dataset import I24Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def merge_batch_by_padding_2nd_dim(tensor_list, return_pad_mask=False):
    # Determine the dimensions of the tensors
    tensor_shape_len = len(tensor_list[0].shape)
    if tensor_shape_len not in [2]:
        return torch.stack(tensor_list, dim=0)

    # Flag for determining if we need to adjust the dimensions back
    only_2d_tensor = tensor_shape_len == 2

    # If the input tensors are 2D, add an extra dimension to make it compatible with 3D tensors
    if only_2d_tensor:
        tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list]  # Convert to (batch, length, 1, 1)
        
    
    # Calculate the maximum length along the second dimension
    maxt_feat0 = max([x.shape[0] for x in tensor_list])
    
    # Retrieve the shape attributes after adjustment
    _, num_feat1, num_feat2 = tensor_list[0].shape

    # Initialize lists to store padded tensors and mask tensors
    ret_tensor_list = []
    ret_mask_list = []

    # Pad tensors and generate masks
    for k in range(len(tensor_list)):
        cur_tensor = tensor_list[k]

        # Create a new tensor with the maximum shape and copy the current tensor into it
        new_tensor = cur_tensor.new_zeros(maxt_feat0, num_feat1, num_feat2)
        new_tensor[:cur_tensor.shape[0], :, :] = cur_tensor
        # new_tensor = new_tensor[None,]
        ret_tensor_list.append(new_tensor)
        # Create a mask tensor indicating the padded regions
        new_mask_tensor = cur_tensor.new_zeros(maxt_feat0)
        new_mask_tensor[:cur_tensor.shape[0]] = 1
        ret_mask_list.append(new_mask_tensor.bool())
        
    # Concatenate the padded tensors and masks along the batch dimension
    ret_tensor = torch.stack(ret_tensor_list, dim=0)  # (num_stacked_samples, maxt_feat0, num_feat1, num_feat2)
    ret_mask = torch.stack(ret_mask_list, dim=0)
    # print(ret_tensor.shape)
    # If the input was originally a 2D tensor, squeeze back to the original number of dimensions
    if only_2d_tensor:
        
        ret_tensor = ret_tensor.squeeze(dim=-1).squeeze(dim=-1)  # Remove last two dimensions

    if return_pad_mask:
        return ret_tensor, ret_mask
    else:
        return ret_tensor


def process_batch(batch_list):
    key_to_list = {}
    batch_size = len(batch_list)
    meta_scalar = [
                    'num_vehicles',
                    # '_id', 
                #    'scene_id', 
                   'start_pos', 
                   'start_time'
                   ]
    meta_array = ['length', 
                  'width', 
                #   'height', 
                  'class', 
                  'direction']
    state = ['timestamp', 'x_position', 'y_position', 'x_velocity', 'y_velocity', 'yaw_angle', 'occluded_occupancy_map', 'observed_occupancy_map', 'flow_map']
    state_keys = []
    meta_scalar_keys = []
    meta_array_keys = []
    # for scene_key in ['prv', 'cur', 'nxt']:
    for scene_key in ['cur']:
        for key in meta_scalar:
            key = f'{scene_key}/meta/{key}'
            meta_scalar_keys.append(key)
        for key in meta_array:
            key = f'{scene_key}/meta/{key}'
            meta_array_keys.append(key)
        for key in state:
            his_key = f'{scene_key}/state/his/{key}'
            pred_key = f'{scene_key}/state/pred/{key}'
            state_keys.extend([his_key, pred_key])
            
    for key in meta_scalar_keys:
        key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]
    for key in meta_array_keys:
        key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]
    for key in state_keys:
        key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]
    
    
    input_dict = {}
    for key, val_list in key_to_list.items():
        if key in state_keys:
            
            val_list = [torch.from_numpy(x) for x in val_list]
            input_dict[key] = merge_batch_by_padding_2nd_dim(val_list)
            
        elif key in meta_scalar_keys:
            # chec if value list are zero-dimensional arrays
            if not isinstance(val_list[0], np.ndarray):
                
                input_dict[key] = torch.tensor(val_list)
            else:   
                input_dict[key] = np.concatenate(val_list, axis=0)
        else:
            
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = torch.cat(val_list, dim=0)
    return input_dict

def collate_fn(batch_list):
    
    return process_batch([_ for _ in batch_list])


def get_dataset(config):
    dataset = I24Dataset(config)
    return dataset

def get_train_val_test_dataset(dataset, config):
    
    train_ratio = config.dataset_splits.train_ratio
    val_ratio = config.dataset_splits.validation_ratio
    test_ratio = config.dataset_splits.test_ratio
    
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    return train_dataset, val_dataset, test_dataset

def get_dataloader_ddp(config):
    
    dataset = get_dataset(config)
    train_dataset, val_dataset, test_dataset = get_train_val_test_dataset(dataset, config)
    
    train_dataloader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset, shuffle=config.dataloader_config.shuffle, drop_last=True), batch_size=config.dataloader_config.batch_size, collate_fn=collate_fn, num_workers=config.dataloader_config.num_workers)
    val_dataloader = DataLoader(val_dataset, sampler=DistributedSampler(val_dataset, shuffle=config.dataloader_config.shuffle, drop_last=True), batch_size=config.dataloader_config.batch_size, collate_fn=collate_fn, num_workers=config.dataloader_config.num_workers)
    test_dataloader = DataLoader(test_dataset, sampler=DistributedSampler(test_dataset, shuffle=config.dataloader_config.shuffle, drop_last=True), batch_size=config.dataloader_config.batch_size, collate_fn=collate_fn, num_workers=config.dataloader_config.num_workers)
    
    return train_dataloader, val_dataloader, test_dataloader

def get_dataloader(config):
    
    dataset = get_dataset(config)
    train_dataset, val_dataset, test_dataset = get_train_val_test_dataset(dataset, config)
    
    train_dataloader = DataLoader(train_dataset, shuffle=config.dataloader_config.shuffle, batch_size=config.dataloader_config.batch_size, collate_fn=collate_fn, num_workers=config.dataloader_config.num_workers)
    val_dataloader = DataLoader(val_dataset, shuffle=config.dataloader_config.shuffle, batch_size=config.dataloader_config.batch_size, collate_fn=collate_fn, num_workers=config.dataloader_config.num_workers)
    test_dataloader = DataLoader(test_dataset, shuffle=config.dataloader_config.shuffle, batch_size=config.dataloader_config.batch_size, collate_fn=collate_fn, num_workers=config.dataloader_config.num_workers)
    
    return train_dataloader, val_dataloader, test_dataloader


def get_trajs(input_dict, config):
    num_vehicles = input_dict['cur/meta/num_vehicles']
    vector_features_list = ['cur/meta/length', 'cur/meta/width', 'cur/meta/class', 'cur/meta/direction']
    node_features_list = ['cur/state/his/timestamp', 'cur/state/his/x_position', 'cur/state/his/y_position', 'cur/state/his/x_velocity', 'cur/state/his/y_velocity', 'cur/state/his/yaw_angle',]
    obs_idx = (input_dict['cur/state/his/timestamp'] > 0)[...,-1]
    vector_feature = torch.cat([torch.unsqueeze(input_dict[feature], dim=1) for feature in vector_features_list], dim=1)
    node_feature = torch.cat([torch.unsqueeze(input_dict[feature], dim=3) for feature in node_features_list], dim=3)
    B, N, T, _ = node_feature.shape
    vector_feature_processed = torch.zeros([B, N, T, 4]).to(node_feature.device)
    prv_vehicle_idx = 0
    for batch_idx in range(B):
        vector_feature_processed[batch_idx, :int(num_vehicles[batch_idx]) - 1, :, :] = torch.repeat_interleave(torch.unsqueeze(vector_feature[int(prv_vehicle_idx):int(prv_vehicle_idx + num_vehicles[batch_idx] - 1), :], dim = 1), repeats=T, dim=1)
        prv_vehicle_idx += num_vehicles[batch_idx]
    trajs = torch.cat([node_feature, vector_feature_processed], dim=3)
    obs_trajs = torch.zeros_like(trajs)
    occ_trajs = torch.zeros_like(trajs)
    obs_trajs[obs_idx] = trajs[obs_idx]
    occ_trajs[~obs_idx] = trajs[~obs_idx]
    return obs_trajs, occ_trajs


def get_road_map(config, batch_first=True):
    batch_size = config.dataloader_config.batch_size
    grid_size_x = config.occupancy_flow_map.grid_size.x
    grid_size_y = config.occupancy_flow_map.grid_size.y
    map_size = (grid_size_x, grid_size_y)
    base_img = np.ones((*map_size,3))
    road_lines_south = [-60, -48, -36, -24, -12]
    road_lines_north = [12, 24, 36, 48, 60]
    road_boundaries_north = [12, 60]
    road_boundaries_south = [-60, -12]
    def to_map_coord(y):
        return int(y * grid_size_y / 160 + grid_size_y / 2)
    center_lines_north = [to_map_coord((road_lines_north[i] + road_lines_north[i + 1]) / 2) for i in range(0, len(road_lines_north) - 1)]
    
    center_lines_south = [to_map_coord((road_lines_south[i] + road_lines_south[i + 1]) / 2) for i in range(0, len(road_lines_south) - 1)]
    # set the road area
    road_boundaries_north = [to_map_coord(x) for x in road_boundaries_north]
    road_boundaries_south = [to_map_coord(x) for x in road_boundaries_south]
    base_img[:, road_boundaries_north[0]:road_boundaries_north[1], :] = 0.75
    base_img[:, road_boundaries_south[0]:road_boundaries_south[1], :] = 0.75
    # set the road center lines
    base_img[:, center_lines_north, :] = 0.5
    base_img[:, center_lines_south, :] = 0.5
    
    # set the road lines
    road_lines_south = [to_map_coord(x) for x in road_lines_south]
    road_lines_north = [to_map_coord(x) for x in road_lines_north]
    base_img[:, road_lines_south, :] = 0.25
    base_img[:, road_lines_north, :] = 0.25
    # set the road boundaries
    base_img[:, road_boundaries_north, :] = 0
    base_img[:, road_boundaries_south, :] = 0
    if batch_first:
        base_img = base_img[None, :, :, :]
        base_img = np.repeat(base_img, batch_size, axis=0)
    
    return base_img





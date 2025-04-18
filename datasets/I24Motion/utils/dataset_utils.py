import torch
import numpy as np
from datasets.I24Motion.I24Motion_dataset import I24MotionDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import DefaultDict


import torch

def merge_batch_by_padding_0nd_dim_ndarray(tensor_list, return_pad_mask=False):
    # Determine the dimensions of the tensors
    tensor_shape_len = len(tensor_list[0].shape)
    if tensor_shape_len not in [2, 3]:
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


def merge_batch_by_padding_0nd_dim_array(tensor_list, return_pad_mask=False):
    # Determine the dimensions of the tensors
    tensor_shape_len = len(tensor_list[0].shape)
    assert tensor_shape_len in [1]

    # Calculate the maximum length along the second dimension
    maxt_feat0 = max([x.shape[0] for x in tensor_list])
    

    # Initialize lists to store padded tensors and mask tensors
    ret_tensor_list = []
    ret_mask_list = []

    # Pad tensors and generate masks
    for k in range(len(tensor_list)):
        cur_tensor = tensor_list[k]

        # Create a new tensor with the maximum shape and copy the current tensor into it
        new_tensor = cur_tensor.new_zeros(maxt_feat0)
        new_tensor[:cur_tensor.shape[0], ...] = cur_tensor
        # new_tensor = new_tensor[None,]
        ret_tensor_list.append(new_tensor)
        # Create a mask tensor indicating the padded regions
        new_mask_tensor = cur_tensor.new_zeros(maxt_feat0)
        new_mask_tensor[:cur_tensor.shape[0]] = 1
        ret_mask_list.append(new_mask_tensor.bool())
        
    # Concatenate the padded tensors and masks along the batch dimension
    ret_tensor = torch.stack(ret_tensor_list, dim=0)  # (num_stacked_samples, maxt_feat0, num_feat1, num_feat2)
    ret_mask = torch.stack(ret_mask_list, dim=0)
    

    if return_pad_mask:
        return ret_tensor, ret_mask
    else:
        return ret_tensor


def process_batch(batch_list):
    key_to_list_all = DefaultDict(dict)
    batch_size = len(batch_list)
    occupancy_flow_map_keys = [
        # 'his/occluded_occupancy_map', 
        'pred/occluded_occupancy_map', 
        'his/observed_occupancy_map', 
        'pred/observed_occupancy_map', 
        'his/flow_map', 
        'pred/flow_map', 
        'flow_origin_occupancy_map'
    ]
    trajectory_keys = [            
        'his/observed_agent_features', 
        'pred/observed_agent_features', 
        # 'his/occluded_agent_features', 
        # 'pred/occluded_agent_features', 
        # 'his/trajectories',
        'pred/trajectories',
        'his/valid_mask',
        'pred/valid_mask',
    ]
    meta_scalar_keys = ['agent_types']

    for scene_key in ['prv', 'cur', 'nxt']:
        for key in occupancy_flow_map_keys:
            key_to_list_all[scene_key][key] = [batch_list[bs_idx][scene_key][key] for bs_idx in range(batch_size)]

        for key in trajectory_keys:
            key_to_list_all[scene_key][key] = [batch_list[bs_idx][scene_key][key] for bs_idx in range(batch_size)]

        for key in meta_scalar_keys:
            key_to_list_all[scene_key][key] = [batch_list[bs_idx][scene_key][key] for bs_idx in range(batch_size)]
            
    input_dict = DefaultDict(dict)
    for scene_key in ['prv', 'cur', 'nxt']:
        key_to_list = key_to_list_all[scene_key]
        for key, val_list in key_to_list.items():
            if key in occupancy_flow_map_keys:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[scene_key][key] = merge_batch_by_padding_0nd_dim_ndarray(val_list)

            elif key in trajectory_keys:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[scene_key][key] = merge_batch_by_padding_0nd_dim_ndarray(val_list)
            
            elif key in meta_scalar_keys:
                
                # check if value list are zero-dimensional arrays
                if not isinstance(val_list[0], np.ndarray):
                    input_dict[scene_key][key] = torch.tensor(val_list)
                else:   
                    val_list = [torch.from_numpy(x) for x in val_list]
                    input_dict[scene_key][key] = merge_batch_by_padding_0nd_dim_array(val_list)
            
    return input_dict

def collate_fn(batch_list):
    
    return process_batch([batch_data for batch_data in batch_list])


def get_dataset(config):
    dataset = I24MotionDataset(config)
    return dataset

def get_train_val_test_dataset(dataset, datasets_config):
    
    train_ratio = datasets_config.train_ratio
    val_ratio = datasets_config.validation_ratio
    test_ratio = datasets_config.test_ratio
    
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    return train_dataset, val_dataset, test_dataset

def get_dataloader_ddp(dataloaders_config):
    datasets_config = dataloaders_config.datasets
    train_loader_config = dataloaders_config.train
    val_loader_config = dataloaders_config.val
    test_loader_config = dataloaders_config.test
    dataset = get_dataset(datasets_config)
    train_dataset, val_dataset, test_dataset = get_train_val_test_dataset(dataset, datasets_config)
    

    train_dataloader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset, shuffle=train_loader_config.shuffle, drop_last=True), batch_size=train_loader_config.batch_size, collate_fn=collate_fn, num_workers=train_loader_config.num_workers)
    val_dataloader = DataLoader(val_dataset, sampler=DistributedSampler(val_dataset, shuffle=val_loader_config.shuffle, drop_last=True), batch_size=val_loader_config.batch_size, collate_fn=collate_fn, num_workers=val_loader_config.num_workers)
    test_dataloader = DataLoader(test_dataset, sampler=DistributedSampler(test_dataset, shuffle=test_loader_config.shuffle, drop_last=True), batch_size=test_loader_config.batch_size, collate_fn=collate_fn, num_workers=test_loader_config.num_workers)
    
    return train_dataloader, val_dataloader, test_dataloader

def get_dataloader(dataloaders_config):
    
    datasets_config = dataloaders_config.datasets
    train_loader_config = dataloaders_config.train
    val_loader_config = dataloaders_config.val
    test_loader_config = dataloaders_config.test
    dataset = get_dataset(datasets_config)
    train_dataset, val_dataset, test_dataset = get_train_val_test_dataset(dataset, datasets_config)
    
    train_dataloader = DataLoader(train_dataset, shuffle=train_loader_config.shuffle, batch_size=train_loader_config.batch_size, collate_fn=collate_fn, num_workers=train_loader_config.num_workers)
    val_dataloader = DataLoader(val_dataset, shuffle=val_loader_config.shuffle, batch_size=val_loader_config.batch_size, collate_fn=collate_fn, num_workers=val_loader_config.num_workers)
    test_dataloader = DataLoader(test_dataset, shuffle=test_loader_config.shuffle, batch_size=test_loader_config.batch_size, collate_fn=collate_fn, num_workers=test_loader_config.num_workers)
    
    return train_dataloader, val_dataloader, test_dataloader


@ DeprecationWarning
def get_road_map(config, batch_size = None, batch_first=True):

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





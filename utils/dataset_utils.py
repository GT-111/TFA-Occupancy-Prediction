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
    for scene_key in ['prv', 'cur', 'nxt']:
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
    
    train_ratio = config.dataset.train_ratio
    val_ratio = config.dataset.val_ratio
    test_ratio = config.dataset.test_ratio
    
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    return train_dataset, val_dataset, test_dataset

def get_dataloader(config):
    
    dataset = get_dataset(config)
    train_dataset, val_dataset, test_dataset = get_train_val_test_dataset(dataset, config)
    
    train_dataloader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset, shuffle=config.dataloader.shuffle, drop_last=True), batch_size=config.dataloader.batch_size, collate_fn=collate_fn, num_workers=config.dataloader.num_workers)
    val_dataloader = DataLoader(val_dataset, sampler=DistributedSampler(val_dataset, shuffle=config.dataloader.shuffle, drop_last=True), batch_size=config.dataloader.batch_size, collate_fn=collate_fn, num_workers=config.dataloader.num_workers)
    test_dataloader = DataLoader(test_dataset, sampler=DistributedSampler(test_dataset, shuffle=config.dataloader.shuffle, drop_last=True), batch_size=config.dataloader.batch_size, collate_fn=collate_fn, num_workers=config.dataloader.num_workers)
    
    return train_dataloader, val_dataloader, test_dataloader



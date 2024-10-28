from utils.file_utils import get_npy_files
from dataset.occ_flow_utils import GridMap
import typing
import numpy as np
from torch.utils.data import Dataset

class I24Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.grid_map = GridMap(config)
        self.data_files = get_npy_files(config.dataset.processed_data)
        
    def add_occ_flow(self, feature_dic):
        occluded_occupancy_map, observed_occupancy_map, flow_map = self.grid_map.get_map_flow(feature_dic)
        feature_dic['occluded_occupancy_map'] = occluded_occupancy_map
        feature_dic['observed_occupancy_map'] = observed_occupancy_map
        feature_dic['flow_map'] = flow_map
        return feature_dic 
    
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_dic = np.load(self.data_files[idx], allow_pickle=True).item()
        
        # Create the feature dictionary to save
        his_len = self.config.dataset.his_len
        pred_len = self.config.dataset.pred_len
        feature_dic = typing.DefaultDict(dict)
        for dic_k, dic in data_dic.items():
            dic = self.add_occ_flow(dic)
            for k, v in dic.items():
                if k in ['timestamp', 'x_position', 'y_position', 'x_velocity', 'y_velocity', 'yaw_angle']:
                    # (Num of Agents, Timestamp)

                    feature_dic[dic_k + '/state/his/' + k] = v[:, :his_len]
                    feature_dic[dic_k + '/state/pred/' + k] = v[: his_len: his_len + pred_len]
                elif k in ['occluded_occupancy_map', 'observed_occupancy_map', 'flow_map']:
                    # (Timestamp, H, W) -> Occ (Timestamp, H, W, 2) -> Flow
                    feature_dic[dic_k + '/state/his/' + k] = v[:his_len,...]
                    feature_dic[dic_k + '/state/pred/' + k] = v[his_len: his_len + pred_len,...]
                    # pred_v = v[his_len: his_len + pred_len,...]
                    # pred_v = pred_v.reshape(-1, pred_len//10, *pred_v.shape[1:]).sum(axis=0)
                    # print(f'{k}, pred {pred_v.shape}')
                else:
                    feature_dic[dic_k + '/meta/' + k] = v
            
        return feature_dic
    
    
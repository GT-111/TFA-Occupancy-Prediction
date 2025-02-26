import numpy as np
from torch.utils.data import Dataset
from utils.file_utils import get_files_with_extension
class I24MotionDataset(Dataset):
    def __init__(self, datasets_config):
        self.datasets_config = datasets_config
        self.data_path = self.datasets_config.data_path
        self.total_data_samples = self.datasets_config.total_data_samples
        self.data_files = get_files_with_extension(self.data_path, '.npy')[:self.total_data_samples]
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data_dic = np.load(self.data_files[idx], allow_pickle=True).item()
        
        # Create the feature dictionary to save
        feature_dic = data_dic

        return feature_dic
    
    


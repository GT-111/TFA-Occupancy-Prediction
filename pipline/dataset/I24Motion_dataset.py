import numpy as np
from torch.utils.data import Dataset
from pipline.utils.file_utils import get_files_with_extension
class I24MotionDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_files = get_files_with_extension(config.paths.processed_data, '.npy')[:config.dataloader_config.total_samples]
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data_dic = np.load(self.data_files[idx], allow_pickle=True).item()
        
        # Create the feature dictionary to save
        feature_dic = data_dic

        return feature_dic
    
    


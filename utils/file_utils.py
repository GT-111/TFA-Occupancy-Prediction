import os
import easydict as edict
import json
import yaml
def get_json_files(raw_data_path = './raw_data'):

    files = []
    for root, _, file in os.walk(raw_data_path):
        for f in file:
            if f.endswith('.json'):
                files.append(os.path.join(root, f))

    return files

def get_npy_files(processed_data_path = './processed_data'):
    files = []
    for root, _, file in os.walk(processed_data_path):
        for f in file:
            if f.endswith('.npy'):
                files.append(os.path.join(root, f))

    return files

def get_config (config_file = './config.json'):
    # Load config file
    # if config is end with .json
    if config_file.endswith('.json'):
        with open(config_file) as f:
            config = edict.EasyDict(json.load(f))
    elif config_file.endswith('.yaml'):
        # if config is end with .yaml
        with open(config_file) as f:
            config = edict.EasyDict(yaml.load(f, Loader=yaml.FullLoader))
            
    return config





def get_last_checkpoint(checkpoint_path = './checkpoints/'):
    files = []
    
    for root, _, file in os.walk(checkpoint_path):
        for f in file:
            if f.endswith('.pt'):
                files.append(os.path.join(root, f))
    files.sort()
    return files[-1] if len(files) > 0 else None

get_last_checkpoint()
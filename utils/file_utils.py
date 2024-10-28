import os
import easydict as edict
import json

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
    with open(config_file) as f:
        config = edict.EasyDict(json.load(f))
    return config

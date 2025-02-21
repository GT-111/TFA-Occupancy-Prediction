from utils.config import load_config
from preprocess.preprocess_stage_preparation import process_raw_json2csv
if __name__ == '__main__':
    dataset_config = load_config("configs/dataset_configs/I24Motion_config.py")
    process_raw_json2csv(dataset_config)

from utils.config import load_config
from preprocess.I24MotionDataset.preprocess_preparation import process_raw_json2csv
from preprocess.I24MotionDataset.preprocess_generate_dataset import I24MotionDatasetPreprocessor
if __name__ == '__main__':
    dataset_config = load_config("configs/dataset_configs/I24Motion_config.py")
    process_raw_json2csv(dataset_config)
    # preprocessor = I24MotionDatasetPreprocessor(dataset_config)
    # preprocessor.process_files()
    
import attr
import ijson
from utils.file_utils import get_json_files, get_config
import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.config import load_config



def process_file(raw_data_file, keys_to_use, dataset_metadata, sample_frequency, buffer_size=500, max_workers=4):
    
    df = pd.DataFrame(columns=keys_to_use)
    buffer = []  # Temporary buffer to hold intermediate results

    file_name = os.path.basename(raw_data_file).split('_')[0]
    print(f'Processing {file_name}...')
    start_time_stamp = dataset_metadata[dataset_metadata['Unique Identifier'] == file_name]['timestamp'].values[0]
    with open(raw_data_file, 'r') as input_file:
        parser = ijson.items(input_file, 'item', use_float=True)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for record in tqdm(parser):
                futures.append(
                    executor.submit(process_record, record=record, keys_to_use=keys_to_use, start_time_stamp=start_time_stamp, sample_frequency=sample_frequency)
                )

                if len(futures) >= buffer_size:
                    df = flush_buffer(futures, df, buffer)
                    futures.clear()  

            if futures:
                df = flush_buffer(futures, df, buffer)

    return df, file_name

def flush_buffer(futures, df, buffer):
    """Flush the buffer by collecting results and concatenating with the DataFrame."""
    for future in as_completed(futures):
        result = future.result()
        buffer.append(result)

    df = pd.concat([df, pd.concat(buffer, axis=0)], ignore_index=True)
    buffer.clear()  # Clear the buffer after flushing
    return df

def process_record(record, keys_to_use, start_time_stamp, sample_frequency):
    record_length = len(record['timestamp'])
    record_dic = {}
    for k in keys_to_use:
        if k in ['timestamp', 'x_position', 'y_position']:
            record_dic[k] = np.array(record[k])
        elif k in ['_id', 'merged_ids', 'fragment_ids']:
            record_dic[k] = np.repeat(record[k]['$oid'], record_length)
        else:
            record_dic[k] = np.repeat(record[k], record_length)
    
    record_dic['timestamp'] = (record_dic['timestamp'] - start_time_stamp)
    record_dic['timestamp'] = np.round((record_dic['timestamp'] - record_dic['timestamp'][0] % (1 / sample_frequency)) * sample_frequency)
    record_df = pd.DataFrame(record_dic, columns=keys_to_use).set_index('timestamp')
    record_df = record_df.reindex(pd.RangeIndex(start=record_df.index.min(), stop=record_df.index.max() + 1))
    
    # Step 3: Interpolate numeric columns
    record_df[['x_position', 'y_position']] = record_df[['x_position', 'y_position']].interpolate(method='spline', order=3)

    # Step 4: Forward fill non-numeric columns
    record_df[['_id', 'direction', 'length', 'width', 'height' ,'coarse_vehicle_class']] = record_df[['_id', 'direction', 'length', 'width', 'height', 'coarse_vehicle_class']].ffill()

    record_df['timestamp'] = record_df.index
    record_df = record_df.reset_index(drop=True)
    record_df = record_df.astype({'timestamp': 'int32', 'x_position': 'float32', 'y_position': 'float32', 'length': 'float32', 'width': 'float32', 'height': 'float32', 'coarse_vehicle_class': 'int32', 'direction': 'int32'})
    return record_df
            

def read_meta_data(meta_data_path):
    
    dataset_metadata = pd.read_csv(meta_data_path)
    date = dataset_metadata['Date']
    start_time_cst = dataset_metadata['Start time (CST)']
    # convert to Unix timestamp
    start_time_cst = pd.to_datetime(date + ' ' + start_time_cst)
    dataset_metadata['timestamp'] = start_time_cst.apply(lambda x: x.timestamp()) 

    return dataset_metadata

def process_raw_json2csv(config):
    # ============= Load Paths =================
    paths = config.paths
    keys_to_use = config.keys_to_use
    raw_data_path = paths.raw_data_path
    processed_data_path = paths.processed_data_path
    auxilary_data_path = paths.auxilary_data_path

    # ============= Load Metadata =================
    meta_data_path = os.path.join(auxilary_data_path, 'I24MotionMetadata.csv')
    dataset_metadata = read_meta_data(meta_data_path)

    # ============= Load Attributes =================
    attributes = config.attributes
    sample_frequency = attributes.sample_frequency
    # ============= Process Files =================
    raw_data_files = get_json_files(raw_data_path)
    # print(raw_data_files)
    for raw_data_file_idx, raw_data_file in enumerate(raw_data_files):

        # Process the raw data file
        result_df, file_name = process_file(raw_data_file=raw_data_file, keys_to_use=keys_to_use, sample_frequency=sample_frequency, dataset_metadata=dataset_metadata)
        # Save the processed data file
        result_df.to_parquet(os.path.join(processed_data_path, file_name) + '.parquet')
        
        print(f'Processed {raw_data_file_idx + 1}/{len(raw_data_files)} files')

if __name__ == '__main__':
    dataset_config = load_config("configs/dataset_configs/I24Motion_config.py")
    process_raw_json2csv(dataset_config)
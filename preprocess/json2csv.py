import ijson
from utils.file_utils import get_json_files, get_config
import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def process_file(raw_data_file, keys_to_use, duration_dict, buffer_size=500):
    
    df = pd.DataFrame(columns=keys_to_use)
    buffer = []  # Temporary buffer to hold intermediate results

    file_name = os.path.basename(raw_data_file).split('_')[0]
    start_time_stamp = duration_dict[file_name]['start_timestamp']

    with open(raw_data_file, 'r') as input_file:
        parser = ijson.items(input_file, 'item', use_float=True)

        with ThreadPoolExecutor() as executor:
            futures = []

            for record in tqdm(parser):
                futures.append(
                    executor.submit(process_record, record, start_time_stamp, keys_to_use)
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

def process_record(record, start_time_stamp, keys_to_use):
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
    
    record_dic['timestamp'] = np.round((record_dic['timestamp'] - record_dic['timestamp'][0] % (1 / config.data_attributes.sample_frequency)) * config.data_attributes.sample_frequency)
    record_df = pd.DataFrame(record_dic, columns=keys_to_use).set_index('timestamp')
    
    record_df = record_df.reindex(pd.RangeIndex(start=record_df.index.min(), stop=record_df.index.max() + 1))

    # Step 3: Interpolate numeric columns
    record_df_numeric = record_df.select_dtypes(include=[np.number])
    record_df[record_df_numeric.columns] = record_df_numeric.interpolate(method='spline', order=3)

    # Step 4: Forward fill non-numeric columns
    record_df['_id'].ffill(inplace=True)
    record_df['coarse_vehicle_class'].ffill(inplace=True)
    record_df['direction'].ffill(inplace=True)
    record_df['timestamp'] = record_df.index
    record_df = record_df.reset_index(drop=True)[keys_to_use]
    record_df = record_df.astype({'timestamp': 'int32', 'x_position': 'float32', 'y_position': 'float32', 'length': 'float32', 'width': 'float32', 'height': 'float32', 'coarse_vehicle_class': 'int32', 'direction': 'int32'})
    return record_df
            

def json2csv(config, keys_to_use, duration_dict):
    
    raw_data_files = get_json_files(config.paths.raw_data)
    # print(raw_data_files)
    for raw_data_file in raw_data_files:
        
        result_df, file_name = process_file(raw_data_file, keys_to_use, duration_dict)
        path = config.paths.processed_data + file_name + '.parquet'
        
        result_df.to_parquet(path)
        
        
        print(f'{file_name} processed')
        break
def json2csv(config, keys_to_use, duration_dict, file_path, output_path):
    
    # print(raw_data_files)
    
        
    result_df, file_name = process_file(file_path, keys_to_use, duration_dict)
    path = output_path + file_name + '.parquet'
    
    result_df.to_parquet(path)
    
    
    print(f'{file_name} processed')
    

if __name__ == '__main__':
    config = get_config()

    duration_dict = np.load('auxiliary_data/duration_dict.npy', allow_pickle=True).item()
    keys_to_use = ['_id', 
                    'timestamp', 
                    'x_position', 
                    'y_position', 
                    # 'road_segment_ids', 
                    # 'flags', 
                    'length', 
                    'width', 
                    'height', 
                    # 'merged_ids', 
                    # 'fragment_ids', 
                    'coarse_vehicle_class', 
                    # 'fine_vehicle_class', 
                    'direction', 
                    # 'compute_node_id', 
                    # 'local_fragment_id', 
                    # 'starting_x', 
                    # 'first_timestamp', 
                    # 'configuration_id', 
                    # 'ending_x', 
                    # 'last_timestamp', 
                    # 'x_score', 
                    # 'y_score'
                    ]
    # json2csv(config, keys_to_use, duration_dict)
    json2csv(config, keys_to_use, duration_dict, '/media/thing1/T9/HetianGuo/I24Motion/raw_data/637b023440527bf2daa5932f__post1.json', './raw_data/')
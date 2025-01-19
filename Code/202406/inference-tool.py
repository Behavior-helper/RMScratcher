import os
import time
import csv
import logging
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from yolov5_dependency import Yolov5Service
import yolov5_context
from calculate_frame_diff_GPU import calculate_frame_differences
#from calculate_frame_differences import calculate_frame_differences
from behavior_identification import Behaviors_identification_by_videos

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

processed_mkv_files = set()
processed_csv_files = set()

def process_video_file(file_path, output_folder):
    try:
        cuda.init()
        dev = cuda.Device(0)  
        context = dev.make_context()
        base_name = os.path.basename(file_path)
        output_csv = os.path.join(output_folder, base_name.replace(".mkv", ".csv"))
        calculate_frame_differences(file_path, output_csv)
        logger.info(f'Finished processing {file_path}, results saved to {output_csv}')

        context.pop() 
    except Exception as e:
        logger.error(f'Error processing {file_path}: {e}', exc_info=True)
    finally:
        try:
            cuda.Context.pop()  
        except Exception:
            pass


def process_csv_file(csv_path, yolov5_service, Result_folder, Identify_scraching_region=False):
    if csv_path in processed_csv_files:
        return  

    processed_csv_files.add(csv_path) 

    try:
        
        start_time = time.time()
        time_5min = Behaviors_identification_by_videos(csv_path, Identify_scraching_region)
        end_time = time.time()
        
        logger.info(f"Processed {csv_path} - Time cost: {end_time - start_time} s")

        
        with open(csv_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            rows = list(csv_reader)
            if len(rows) < 4:
                logger.warning(f"CSV file {csv_path} does not have enough columns.")
                return
            fourth_column = [row[3] for row in rows if len(row) > 3]

        
        for i in range(time_5min.shape[0]):
            start_frame = int(time_5min[i, 0])
            end_frame = int(time_5min[i, 1])
            if start_frame < len(fourth_column) and end_frame < len(fourth_column):
                time_5min[i, 0] = fourth_column[start_frame]
                time_5min[i, 1] = fourth_column[end_frame]

        result_csv_path = f"{os.path.splitext(csv_path)[0]}_Result{os.path.splitext(csv_path)[1]}"
        filename = os.path.basename(result_csv_path)
        new_path = os.path.join(Result_folder, filename)
        np.savetxt(new_path, time_5min, fmt='%s', delimiter=',')
        logger.info(f"Result saved to {new_path}")
    except Exception as e:
        logger.error(f'Error processing CSV file {csv_path}: {e}', exc_info=True)


def process_all_csv_files(folder_path, yolov5_service, Result_folder):
    
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        
        futures = [executor.submit(process_csv_file, csv_file, yolov5_service, Result_folder) for csv_file in csv_files]
        
        
        for future in futures:
            try:
                future.result() 
            except Exception as e:
                logger.error(f'Error in processing CSV file: {e}', exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference Tool')
    parser.add_argument('-i', '--input_dir', required=True, help='Path to the input directory')
    parser.add_argument('output_dir', help='Path to the output directory')
    
    args = parser.parse_args()
    
    folder_mkv_path = args.input_dir
    output_folder = args.output_dir
    
    # Result_folder = os.path.join(output_folder, "Result")
    
    os.makedirs(output_folder, exist_ok=True)
    # os.makedirs(diff_folder, exist_ok=True)
    # os.makedirs(Result_folder, exist_ok=True)
    yolov5_service = Yolov5Service(
        engine_path="/home/ubuntu/chenminUI/tensorrtx_yolov5_v5/yolov5/build_FP16/bestRT.engine",
        plugin_library="/home/ubuntu/chenminUI/tensorrtx_yolov5_v5/yolov5/build_FP16/libmyplugins.so"
    )
    yolov5_context.set_yolov5_service(yolov5_service)
    print("Yolov5Service created.")
    
    #time.sleep(300)
    logger.info("Start to process the first 4 pipeline videos.")
    
    Process_time = 0
    try:
        while True:
            
            mkv_files = [f for f in os.listdir(folder_mkv_path) 
                         if f.endswith('.mkv') and f not in processed_mkv_files]
            
            if mkv_files:
                
                batch = mkv_files[:4]
                logger.info(f"Found {len(batch)} new mkv files to process: {batch}")
                start_time_1 = time.time()
                
                with multiprocessing.Pool(processes=4) as pool:
                    pool.starmap(process_video_file, [(os.path.join(folder_mkv_path, f), folder_mkv_path) for f in batch])
                
                
                processed_mkv_files.update(batch)
                logger.info(f"Finished processing MKV files: {batch}")
                
                
                process_all_csv_files(folder_mkv_path, yolov5_service, output_folder)
                logger.info("Finished processing CSV files.")
                end_time_1 = time.time()
                Process_time = start_time_1 - end_time_1
                if Process_time < 300:
                    time.sleep(300 - Process_time)
            else:
                logger.info("No new MKV files to process. Waiting for new files...")
            
            
            time.sleep(10) 
    except KeyboardInterrupt:
        logger.info("Program interrupted by user. Exiting...")
        yolov5_service.destroy()
    finally:
        
        yolov5_service.destroy()


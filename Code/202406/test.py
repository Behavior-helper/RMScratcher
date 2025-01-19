import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import pandas as pd
import logging
from yolov5_dependency import Yolov5Service
import yolov5_context
from calculate_frame_diff_GPU import calculate_frame_differences
from behavior_identification import Behaviors_identification_by_videos
import csv
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

processed_files = set() 

def process_csv_file(csv_path, yolov5_service, Identify_scraching_region=False):
    if csv_path in processed_files:
        return 

    processed_files.add(csv_path)  


    start_time = time.time()
    time_5min = Behaviors_identification_by_videos(csv_path, Identify_scraching_region)
    end_time = time.time()

    print(f"Processed {csv_path} - Time cost: {end_time - start_time} s")


    result_csv_path = csv_path.split(".")[0] + "_Result." + csv_path.split(".")[1]
    np.savetxt(result_csv_path, time_5min, fmt='%s', delimiter=',')

def process_all_files(folder_path, yolov5_service):
    
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        
        futures = [executor.submit(process_csv_file, csv_file, yolov5_service) for csv_file in csv_files]
        
        
        for future in futures:
            future.result()  

class NewFileHandler(FileSystemEventHandler):
    def __init__(self, yolov5_service):
        self.yolov5_service = yolov5_service
    
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".csv"):
            process_csv_file(event.src_path, self.yolov5_service)

def start_processing(folder_path, yolov5_service):
    
    process_all_files(folder_path, yolov5_service)

    
    event_handler = NewFileHandler(yolov5_service)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    folder_path = "/home/ubuntu/chenminUI/testvideo"
    
    yolov5_service = Yolov5Service(
        engine_path="/home/ubuntu/chenminUI/tensorrtx_yolov5_v5/yolov5/build_FP16/bestRT.engine",
        plugin_library="/home/ubuntu/chenminUI/tensorrtx_yolov5_v5/yolov5/build_FP16/libmyplugins.so"
    )
    yolov5_context.set_yolov5_service(yolov5_service)
    

    start_processing(folder_path, yolov5_service)

    yolov5_service.destroy()

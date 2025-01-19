import cv2
import os
import sys
import numpy as np
import csv
import yolov5_context

import warnings
warnings.filterwarnings("ignore")

def Yolov8_classfication(video_path, frame_positions, cap):
    frame_positions = np.array(frame_positions)
    csv_path = video_path[:-4] + ".csv"
    time_positions = []


    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        rows = list(csv_reader) 
        for position in frame_positions:
            if position < len(rows):
                time_positions.append(rows[position][1])


    
    detected_frames_count = 0
    
    for time_position in time_positions:
        time_pos = float(time_position)

        cap.set(cv2.CAP_PROP_POS_MSEC, time_pos * 1000)
        # print(time_pos * 1000)
        ret, frame = cap.read()
        yolov5_service = yolov5_context.get_yolov5_service()
        if yolov5_service is None:
            raise RuntimeError("Yolov5Service has not been initialized!")
        if ret:   
            final_boxes, final_classids, use_time, batch_image_raw = yolov5_service.infer(frame)
            #print("final_classids:", final_classids)
            num_claw = np.sum(final_classids == 2.0)
            if num_claw == 1:
                detected_frames_count += 1
    
    cap.release()
    
    if detected_frames_count >= 4:
        scratch_behavior = 1
    else:
        scratch_behavior = 0

    return scratch_behavior

if __name__ == "__main__":
    video_name = sys.argv[1]
    frame_positions = list(map(int, sys.argv[2].split(',')))

    scratch_behavior = Yolov8_classfication(video_name, frame_positions)
    print(scratch_behavior)


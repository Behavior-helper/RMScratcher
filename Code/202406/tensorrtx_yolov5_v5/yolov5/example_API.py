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
from yolov5_RTsingle import YoLov5TRT

categories = ['ear', 'mouse', 'claw', 'not_scratch']
PLUGIN_LIBRARY = "/home/ubuntu/chenminUI/tensorrtx_yolov5_v5/yolov5/build_FP16/libmyplugins.so"
engine_file_path = "/home/ubuntu/chenminUI/tensorrtx_yolov5_v5/yolov5/build_FP16/bestRT.engine"
# load custom plugin and engine
image_path = "/home/ubuntu/tensorrtx-yolov5-v5.0/yolov5/build/coco_calib/81211-0_05_46.982.jpg"
ctypes.CDLL(PLUGIN_LIBRARY)

# a YoLov5TRT instance
yolov5_wrapper = YoLov5TRT(engine_file_path)


final_boxes, final_classids, use_time, batch_image_raw = yolov5_wrapper.infer(yolov5_wrapper.get_raw_image([image_path]))
for box, classid in zip(final_boxes, final_classids):
    print(f" Box: {box}, Class ID: {classid}, Class Name: {categories[int(classid)]}")
    # 打印出classid的类型
    print(type(classid))
    # 打印出box的类型
    print(type(box))

yolov5_wrapper.destroy()

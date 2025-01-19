# yolov5_dependency.py
import numpy as np
import ctypes
from tensorrtx_yolov5_v5.yolov5.yolov5_RTsingle import YoLov5TRT

class Yolov5Service:
    def __init__(self, engine_path, plugin_library):
        ctypes.CDLL(plugin_library)
        self.yolov5_wrapper = YoLov5TRT(engine_path)


    def infer(self, image_input):
        if isinstance(image_input, np.ndarray):
            return self.yolov5_wrapper.infer([image_input])
        elif isinstance(image_input, str):

            return self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image([image_input]))
        else:
            raise ValueError("Unsupported input type. Expected str (file path) or numpy.ndarray (image data).")


    def destroy(self):
        self.yolov5_wrapper.destroy()


# yolov5_context.py
yolov5_service = None

def set_yolov5_service(service):
    global yolov5_service
    yolov5_service = service

def get_yolov5_service():
    return yolov5_service

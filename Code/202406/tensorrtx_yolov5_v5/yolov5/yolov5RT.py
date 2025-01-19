import cv2
import numpy as np
import torch

def yolov5RT_infer(image):
    # 初始化设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载 YOLOv5 模型
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to(device)
    model.eval()
    
    # 图像预处理
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (640, 640))
    img_input = np.transpose(img_resized, (2, 0, 1)) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    img_input = torch.from_numpy(img_input).float().to(device)
    
    # 推理
    with torch.no_grad():
        output = model(img_input)

    # 解析结果
    result = output[0]

    # 返回检测结果
    return result



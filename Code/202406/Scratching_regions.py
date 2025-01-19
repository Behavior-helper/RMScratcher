import warnings
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw  
warnings.filterwarnings('ignore')
import csv
# 20240820
import yolov5_context

def calculate_midpoint(box):

    x_mid = (box[0] + box[2]) / 2
    y_mid = (box[1] + box[3]) / 2
    return x_mid, y_mid

def calculate_distance(point1, point2):

    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def extend_line(point1, point2, factor):

    extended_point1 = (point1[0] - factor * (point2[0] - point1[0]), point1[1] - factor * (point2[1] - point1[1]))
    extended_point2 = (point2[0] + factor * (point2[0] - point1[0]), point2[1] + factor * (point2[1] - point1[1]))
    return extended_point1, extended_point2

def caculate_theta(S):

    if S[0,0]>0 and S[1,0]>0:
        theta = np.arccos(S[0,0])

    elif S[0,0]>0 and S[1,0]<0:
        theta = 2*np.pi - np.arccos(S[0,0])

    elif S[0,0]<0 and S[1,0]<0:
        theta = 0.5*np.pi + np.arccos(S[0,0])

    elif S[0,0]<0 and S[1,0]>0:
         theta = np.arccos(S[0,0])
    return theta

def behavior_predict(video_path, frame_positions):

    cap = cv2.VideoCapture(video_path)

    csv_path = video_path[:-4] + ".csv"

    time_positions = []


    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        rows = list(csv_reader)
        for position in frame_positions:
            if position < len(rows):
                time_positions.append(rows[position][3])
    

    if not cap.isOpened():
        print(f"Error: Unable to open video file at {video_path}")
        return
    

    scratch_region_n = 0
    total_frames_processed = len(frame_positions)
    scratch_region_num = []
    

    yolov5_service = yolov5_context.get_yolov5_service()

    for time_position in time_positions:
        time_pos = float(time_position)

        cap.set(cv2.CAP_PROP_POS_MSEC, time_pos*1000)

        ret, frame = cap.read() 

        if yolov5_service is None:
            raise RuntimeError("Yolov5Service has not been initialized!")   
        if ret:   
           
            output_file = f"{video_path[-14:-4]}_{time_position[:6]}.jpg"
            
            class_ids_list = []
            box_list = []
            final_boxes, final_classids, use_time, batch_image_raw = yolov5_service.infer(frame)
            
            for box, classid in zip(final_boxes, final_classids):
               
                classid = int(classid)
                
                class_ids_list.append(classid)
                
                box_list.append(box)

         
            class_ids = np.array(class_ids_list)
            boxes = np.array(box_list)
            
            num_ear = np.sum(class_ids == 0)
            num_body = np.sum(class_ids == 1)
            num_claw = np.sum(class_ids == 2)
            
            if num_ear == 2 and num_body == 1 and num_claw ==1:
                xyxy_boxes = boxes.xyxy.cpu().numpy().astype(int)
                img_array = r.plot(conf=False, line_width=1, font_size=1.5)
                img = Image.fromarray(img_array[..., ::-1])
                img_draw = ImageDraw.Draw(img)

                ear_cls_indices = np.where(class_ids == 0)[0]  
                ear_centers = [calculate_midpoint(xyxy_boxes[i]) for i in ear_cls_indices]

                
                e_point = ((ear_centers[0][0] + ear_centers[1][0]) / 2, (ear_centers[0][1] + ear_centers[1][1]) / 2)

                
                img_draw.text(e_point, "e", fill=(255, 0, 0))  

                
                ear_center1, ear_center2 = ear_centers
                extended_ear_center1, extended_ear_center2 = extend_line(ear_center1, ear_center2, 0.5)
                img_draw.line([extended_ear_center1, extended_ear_center2], fill=(0, 255, 0)) 

             
                mouse_cls_indices = np.where(class_ids == 1)[0]
                mouse_centers = [calculate_midpoint(xyxy_boxes[i]) for i in mouse_cls_indices]
                mouse_center = mouse_centers[0]
                img_draw.text(mouse_center, "z", fill=(0, 0, 255)) 
                
      
                claw_cls_indices = np.where(class_ids == 2)[0]
                claw_centers = [calculate_midpoint(xyxy_boxes[i]) for i in claw_cls_indices]
                claw_center = claw_centers[0]
                img_draw.text(claw_center, "c", fill=(0, 0, 255)) 

              
                ez_vector = (mouse_center[0] - e_point[0], mouse_center[1] - e_point[1])
                
                
                x_vector = (ear_center1[0] - ear_center2[0], ear_center1[1] - ear_center2[0])
                
               
                extended_e_point, extended_mouse_center = extend_line(e_point, mouse_center, 0.5)
                img_draw.line([extended_e_point, extended_mouse_center], fill=(0, 255, 0))  
                
                
                img_draw.line([e_point, claw_center], fill=(255, 0, 0))  
                ec_vector = (claw_center[0] - e_point[0], claw_center[1] - e_point[1])

              
                ez_unit_vec = ez_vector / np.linalg.norm(ez_vector)
                ec_unit_vec = ec_vector / np.linalg.norm(ec_vector)
                x_unit_vec = x_vector / np.linalg.norm(x_vector)
                
                ez_unit_vec = np.array([ez_unit_vec])
                ec_unit_vec = np.array([ec_unit_vec])
                x_unit_vec = np.array([x_unit_vec])

                ez_unit_vec_T = ez_unit_vec.T
                ec_unit_vec_T = ec_unit_vec.T
                x_unit_vec_T = x_unit_vec.T

                K_1 = np.array([[ec_unit_vec[0,0],ec_unit_vec[0,1]],[ec_unit_vec[0,1],-ec_unit_vec[0,0]]])
                C_1 = ez_unit_vec_T
                S_1 = np.linalg.solve(K_1,C_1)
                R_1 = np.array([[S_1[0,0],S_1[1,0]],[-S_1[1,0],S_1[0,0]]])
                #print(S_1)
                #print(R_1)
                K_2 = np.array([[x_unit_vec[0,0],x_unit_vec[0,1]],[x_unit_vec[0,1],-x_unit_vec[0,0]]])
                C_2 = ez_unit_vec_T
                S_2 = np.linalg.solve(K_2,C_2)
                R_2 = np.array([[S_2[0,0],S_2[1,0]],[-S_2[1,0],S_2[0,0]]])
                #print(R_2)
                
                theta_alpha = caculate_theta(S_1)
                theta_beta = caculate_theta(S_2)
                
                if theta_beta>np.pi:
                    theta_beta = theta_beta-np.pi
                else:
                    theta_beta = theta_beta
                theta_alpha_degrees = np.degrees(theta_alpha)
                theta_beta_degrees = np.degrees(theta_beta)

                #print(theta_alpha_degrees)
                #print(theta_beta_degrees)

                if theta_alpha >0 and theta_alpha<=theta_beta:
                    scratch_region_n = 0
                    
                elif theta_beta < theta_alpha and theta_alpha<= np.pi:
                    scratch_region_n = 1
                   
                elif theta_alpha > np.pi and theta_alpha < theta_beta + np.pi:
                    scratch_region_n = 2
                    
                elif theta_alpha > theta_beta + np.pi and theta_alpha<=2*np.pi:
                    scratch_region_n = 3
          
                img_array = np.array(img)
                cv2.imwrite(f"scratch_region/{output_file}", img_array)
            
            elif num_ear == 1 and num_body == 1 and num_claw ==1:
               
                xyxy_boxes = boxes.xyxy.cpu().numpy().astype(int)
                img_array = r.plot(conf=False, line_width=1, font_size=1.5)
                img = Image.fromarray(img_array[..., ::-1])
                img_draw = ImageDraw.Draw(img)

                ear_cls_indices = np.where(class_ids == 0)[0] 
                ear_centers = [calculate_midpoint(xyxy_boxes[i]) for i in ear_cls_indices]

                
                e_point = (ear_centers[0])
    
                
                mouse_cls_indices = np.where(class_ids == 1)[0]
                mouse_centers = [calculate_midpoint(xyxy_boxes[i]) for i in mouse_cls_indices]
                mouse_center = mouse_centers[0]
                img_draw.text(mouse_center, "z", fill=(0, 0, 255))  
                
                
                claw_cls_indices = np.where(class_ids == 2)[0]
                claw_centers = [calculate_midpoint(xyxy_boxes[i]) for i in claw_cls_indices]
                claw_center = claw_centers[0]
                img_draw.text(claw_center, "c", fill=(0, 0, 255))  

                
                ez_vector = (mouse_center[0] - e_point[0], mouse_center[1] - e_point[1])
                            
                
                extended_e_point, extended_mouse_center = extend_line(e_point, mouse_center, 0.5)
                img_draw.line([extended_e_point, extended_mouse_center], fill=(0, 255, 0))  
                
                
                img_draw.line([e_point, claw_center], fill=(255, 0, 0))  
                ec_vector = (claw_center[0] - e_point[0], claw_center[1] - e_point[1])

                
                ez_unit_vec = ez_vector / np.linalg.norm(ez_vector)
                ec_unit_vec = ec_vector / np.linalg.norm(ec_vector)

                
                ez_unit_vec = np.array([ez_unit_vec])
                ec_unit_vec = np.array([ec_unit_vec])


                ez_unit_vec_T = ez_unit_vec.T
                ec_unit_vec_T = ec_unit_vec.T


                K_1 = np.array([[ec_unit_vec[0,0],ec_unit_vec[0,1]],[ec_unit_vec[0,1],-ec_unit_vec[0,0]]])
                C_1 = ez_unit_vec_T
                S_1 = np.linalg.solve(K_1,C_1)
                R_1 = np.array([[S_1[0,0],S_1[1,0]],[-S_1[1,0],S_1[0,0]]])
                #print(S_1)
                #print(R_1)

                
                theta_alpha = caculate_theta(S_1)

                # if theta_beta>np.pi:
                #     theta_beta = theta_beta-np.pi
                # else:
                #     theta_beta = theta_beta
                # theta_alpha_degrees = np.degrees(theta_alpha)

                #print(theta_alpha_degrees)
                #print(theta_beta_degrees)

                if theta_alpha >0 and theta_alpha<=np.pi:
                    scratch_region_n = 2
                 
                else:
                    scratch_region_n = 1
                    
                
                img_array = np.array(img)
                cv2.imwrite(f"scratch_region/{output_file}", img_array)
        
        scratch_region_num.append(scratch_region_n)
    
    
    scratch_region_count = np.bincount(scratch_region_num)
    
    scratch_region = np.argmax(scratch_region_count)
    return scratch_region

if __name__ == "__main__":
  
    video_name = sys.argv[1]
    frame_positions = list(map(int, sys.argv[2].split(',')))

 
    scratch_region = behavior_predict(video_name, frame_positions)

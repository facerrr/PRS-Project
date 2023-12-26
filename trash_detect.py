from ultralytics import YOLO
import cv2
import time
import threading
from ultralytics.utils.plotting import Annotator, colors
from copy import deepcopy
import torch
import math
import numpy as np
from trash_classes import garbage_dict
from loadStream import loadStream
def restrict(point):
    x,y = point
    return max(0, min(x, 640)), max(0, min(y, 480))

def toint(tuple):
    x,y = tuple
    return (int(x),int(y))

def box(box_c,box_size):
    x,y = box_c
    x1,y1 = restrict((x-box_size/2,y-box_size/2))
    x2,y2 = restrict((x+box_size/2,y+box_size/2))
    return int(x1),int(y1),int(x2),int(y2)

def derivate_box_center(elbow_xy, wrist_xy, offset):
    x1,y1 = toint(elbow_xy)
    x2,y2 = toint(wrist_xy)
    
    deltaX = x2-x1
    deltaY = y2-y1

    if deltaX==0 and deltaY==0:
        return (x2,x1)
    elif deltaX==0 :
        return (x2,y2+np.sign(deltaY)*offset/2)
    elif deltaY==0:
        return (x2+np.sign(deltaX)*offset/2, y2)
    else:      
        new_x,new_y = (0,0)
        if abs(deltaX) > abs(deltaY):
            new_x = x2 + np.sign(deltaX)*offset/2
            new_y = y2 + abs(deltaY/deltaX)*np.sign(deltaY)*offset/2
        elif abs(deltaX) < abs(deltaY):
            new_x = x1 + abs(deltaX*((abs(deltaY)+offset/2))/(deltaY))*np.sign(deltaX)
            new_y = y2 + np.sign(deltaY)*offset/2
        else:
            new_x = x2 + abs(deltaY/deltaX)*np.sign(deltaX)*offset/2
            new_y = y2 + abs(deltaY/deltaX)*np.sign(deltaY)*offset/2

        return restrict((new_x,new_y))

def get_wrist_elbow_point(pred_boxes=None,keypoints=None):
    box_area = []
    for d in pred_boxes:
        box = d.xyxy.squeeze()
        if isinstance(box, torch.Tensor):
            box = box.tolist()
            box_area.append((int(box[2])-int(box[0]))*(int(box[3])-int(box[1])))
    maxOne = box_area.index(max(box_area))

    e_w = None
    if keypoints is not None:
        for i,k in enumerate(keypoints):
            if i == maxOne:
                points = k.xy.cpu().numpy()[0]
                if points.shape[0] >= 17:
                    e_w = []
                    r_eblow = points[8]
                    l_eblow = points[7]
                    r_wrist = points[10]
                    l_wrist = points[9]
                    e_w.append([(r_eblow[0],r_eblow[1]),(r_wrist[0],r_wrist[1])])
                    e_w.append([(l_eblow[0],l_eblow[1]),(l_wrist[0],l_wrist[1])])
                break
    return e_w

                
def get_wrist_point(keypoints=None):
    wrist_right = None
    wrist_left = None
    if keypoints is not None:
        for k in keypoints:
            points = k.xy.cpu().numpy()[0]
            if np.all(points[10] != 0):
                wrist_right = (points[10][0], points[10][1])
            if np.all(points[9] != 0):
                wrist_left = (points[9][0], points[9][1])
            break
    return wrist_left, wrist_right

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
                
# Fusion use pose coordinate and trash center coordinate
def fusionM1_plot(img=None, pred_boxes = None, names = None, conf=True, right_xy = None, left_xy = None,):
    font='Arial.ttf'
    annotator = Annotator(
            im=deepcopy(img),
            font=font,
            example=names)
    
    dist_right_wrist = []
    dist_left_wrist = []
    nearest_right_index = None
    nearest_lef_index = None

    if pred_boxes:
        if right_xy or left_xy:
            for d in pred_boxes:
                box = d.xyxy.squeeze()
                if isinstance(box, torch.Tensor):
                    box = box.tolist()
                center = ((int(box[0])+int(box[2]))/2, (int(box[1])+int(box[3]))/2)
                if right_xy:
                    dist_right_wrist.append(calculate_distance(center,right_xy))
                if left_xy:
                    dist_left_wrist.append(calculate_distance(center,left_xy))
            if right_xy:
                nearest_right_index = np.argmin(dist_right_wrist)
            if left_xy:
                nearest_lef_index = np.argmin(dist_left_wrist)

            for i,d in enumerate(pred_boxes):
                if i == nearest_lef_index or i == nearest_right_index:
                    c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                    if i == nearest_lef_index:
                        s = '  nearest_lef'
                    elif i == nearest_right_index:
                        s = '  nearest_right'
                    name = ('' if id is None else f'id:{id} ') + names[c]
                    name = garbage_dict.get(name)
                    label = (f'{name} {conf:.2f}' if conf else name)
                    annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

    return annotator.result()

def fusionM0_plot(pose_result=None, model=None, conf=True):
    e_ws = get_wrist_elbow_point(pose_result.boxes,pose_result.keypoints)
    font='Arial.ttf'
    img_in = pose_result.orig_img
    img_out = None
    names = model.names
    annotator = Annotator(
            im=deepcopy(img_in),
            font=font,
            example=names)
    if e_ws is not None:
        # for e_w in e_ws:
        bcx,bcy = derivate_box_center(e_ws[1][0],e_ws[1][1],150)
        x1,y1,x2,y2= box((bcx,bcy),200)
        crop = img_in[y1:y2, x1:x2]
        trash_result = model(crop)[0]
        trash_pred_boxes = trash_result.boxes
        for d in trash_pred_boxes:
            c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
            name = ('' if id is None else f'id:{id} ') + names[c]
            name = garbage_dict.get(name)
            label = (f'{name} {conf:.2f}' if conf else name)
            annotator.box_label([x1,y1,x2,y2], label, color=colors(c, True))
        img_out = annotator.result()
    else:
        img_out = img_in
    return img_out


def trash_detect_inference(img, trash_model):
    global trash_result
    # time.sleep(3)
    result = trash_model(img)
    trash_result = result[0]


def pose_detect_inference(img, pose_model):
    global pose_result
    # time.sleep(2)
    result = pose_model(img)
    pose_result = result[0]


def main(imgae=False):
    trash_result = None
    pose_result = None

    pose_model = YOLO("checkpoints/yolov8s-pose.pt") 
    trash_model = YOLO("weights/yolov8m_epoch150/weights/best.pt") 

    if imgae:
        img = cv2.imread("docs/test.png")
        pose_result = pose_model(img)[0]
        imgToShow = fusionM0_plot(pose_result = pose_result, model=trash_model)
        cv2.imwrite('docs/test_fusion_result.png',imgToShow)

    else:
        dataset = loadStream('0')

        fusionMode = 0

        for img in dataset:

            if fusionMode == 0:
                pose_result = pose_model(img)[0]
                imgToShow = fusionM0_plot(pose_result = pose_result, model=trash_model)
                cv2.imshow('img',imgToShow)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elif fusionMode == 1:
                trash_thread = threading.Thread(target=trash_detect_inference,args=(img,trash_model))
                pose_thread = threading.Thread(target=pose_detect_inference,args=(img,pose_model))
                trash_thread.start()
                pose_thread.start()

                trash_thread.join()
                pose_thread.join()

                wrist_left, wrist_right = get_wrist_point(pose_result.keypoints)   #得到中心人物的手腕坐标

                imgToShow = fusionM1_plot(img=pose_result.orig_img, 
                                    pred_boxes=trash_result.boxes, 
                                    names=trash_result.names, 
                                    left_xy=wrist_left, right_xy=wrist_right)

                cv2.imshow('img',imgToShow)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break



if __name__ == "__main__":
    main(imgae=True)


from ultralytics import YOLO
import cv2
from loadStream import loadStream

model = YOLO("weights/yolov8m_epoch150/weights/best.pt") 
dataset = loadStream('0')
for img in dataset:

    results = model.track(img, persist=True)
    annotated_frame = results[0].plot()
    cv2.imshow('Tracking',annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

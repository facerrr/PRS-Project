from ultralytics import YOLO
import cv2
from loadStream import loadStream
    
image = True
image_path = "docs/test.png"
out_path = "docs/test_origin_result.png"
model = YOLO("checkpoints/yolov8s.pt")

if image:
    img = cv2.imread(image_path)
    result = model(img)
    imgToShow = result[0].plot()
    cv2.imwrite(out_path,imgToShow)
else:
    dataset = loadStream('0')
    for img in dataset:
        result = model(img)
        imgToShow = result[0].plot()
        cv2.imshow('img',imgToShow)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break







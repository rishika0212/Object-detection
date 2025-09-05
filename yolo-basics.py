from ultralytics import YOLO
import cv2

model = YOLO("yolo-weights/yolov8m.pt")
results=model("Images/q.jpg",show=True)
cv2.waitKey(0)
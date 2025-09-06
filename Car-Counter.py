import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import os
import torch
from sort import *
 
# Resolve paths relative to this script
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(ROOT_DIR, "Videos", "cars.mp4.crdownload")
weights_path = os.path.join(ROOT_DIR, "yolo-weights", "yolov8l.pt")
mask_path = os.path.join(ROOT_DIR, "Images", "mask.png")
graphics_path = os.path.join(ROOT_DIR, "Images", "graphics.png")

# Open video or fallback to webcam
cap = cv2.VideoCapture(video_path)  # For Video
if not cap.isOpened():
    print(f"Warning: Could not open video at {video_path}. Falling back to webcam 0.")
    cap = cv2.VideoCapture(0)

# Load model weights
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"YOLO weights not found at {weights_path}")
model = YOLO(weights_path)

# Select device (GPU if available)
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else 'cpu'
half_precision = use_cuda  # use FP16 on CUDA
print("Device:", "CUDA" if use_cuda else "CPU", (f"({torch.cuda.get_device_name(0)})" if use_cuda else ""))
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
 
# Mask handling: load grayscale and binarize (white=keep, black=remove)
MASK_KEEP_WHITE = True  # set to False if your mask is inverted
mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask_gray is None:
    print(f"Warning: Mask not found at {mask_path}. Proceeding without region masking.")
else:
    # Threshold to binary mask to avoid semi-transparent pixels
    _, mask_gray = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
 
# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Preload overlay graphic once
imgGraphics_cached = cv2.imread(graphics_path, cv2.IMREAD_UNCHANGED)

limits = [400, 297, 673, 297]
totalCount = []

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Info: End of stream or cannot read frame. Exiting.")
        break

    # Apply region mask if available and shape-compatible
    if mask_gray is not None:
        # Resize mask to match frame size if needed
        if mask_gray.shape[:2] != img.shape[:2]:
            mask_resized = cv2.resize(mask_gray, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask_gray
        # Invert if mask semantics are opposite
        if not MASK_KEEP_WHITE:
            mask_resized = cv2.bitwise_not(mask_resized)
        # Use mask to keep ROI (white areas)
        imgRegion = cv2.bitwise_and(img, img, mask=mask_resized)
    else:
        imgRegion = img

    # Overlay cached graphics if available
    if imgGraphics_cached is not None:
        img = cvzone.overlayPNG(img, imgGraphics_cached, (0, 0))

    results = model(imgRegion, stream=True, device=device, half=half_precision)
 
    detections = np.empty((0, 5))
 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = float(box.conf[0])
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Keep only vehicle classes with minimum confidence
            if currentClass in {"car", "truck", "bus", "motorbike"} and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
 
    resultsTracker = tracker.update(detections)
 
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
 
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
 
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
 
    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
 
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
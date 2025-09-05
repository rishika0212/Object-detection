# Object Detection with YOLOv8 (Webcam + Images)

Simple YOLOv8 demos using OpenCV and cvzone for drawing boxes and labels. Includes:
- `yolo_webcam.py`: Real-time detection from your webcam
- `yolo-basics.py`: Single-image detection
- `Images/`: Sample images
- `yolo-weights/`: Pre-downloaded YOLOv8 weights (`n`, `m`, `l`)

## Requirements
- Python 3.10 (recommended)
- Windows 11 / PowerShell
- A working webcam (for `yolo_webcam.py`)

## Setup
1. (Optional) Create and activate a virtual environment
   - PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
2. Install dependencies
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

Notes:
- Torch will automatically use CPU or CUDA if available. If you hit CUDA issues, prefer CPU by ensuring no CUDA-specific torch build is installed, or set `CUDA_VISIBLE_DEVICES=""` when using shells that honor it.

## Usage
### 1) Webcam detection
Run the webcam demo (uses `yolo-weights/yolov8n.pt` by default):
```powershell
python yolo_webcam.py
```
- If you have multiple cameras, change the index in `cv2.VideoCapture(0)` to `1`, `2`, etc.
- To test on a video file instead of webcam, open `yolo_webcam.py` and uncomment the `Video.mp4` line, commenting out the webcam line.

Change the model size (trade accuracy vs speed) by editing this line in `yolo_webcam.py`:
```python
model = YOLO("yolo-weights/yolov8n.pt")  # options: yolov8n.pt, yolov8m.pt, yolov8l.pt
```

### 2) Single image detection
`yolo-basics.py` runs inference on a single image. Update the image path to a file in the `Images/` folder (e.g., `Images/1.jpg`) if needed, then run:
```powershell
python yolo-basics.py
```

## Troubleshooting
- Import errors (e.g., cv2, ultralytics, cvzone): Ensure the virtual environment is activated and `pip install -r requirements.txt` completed without errors.
- Webcam not opening: Try another index in `cv2.VideoCapture(index)`. Make sure no other app is using the camera.
- Slow performance: Use `yolov8n.pt` (already set for webcam). Reduce input resolution or increase model speed.
- Window not closing: Focus the OpenCV window and press any key. If stuck, Ctrl+C in the terminal.

## Project Structure
```
├─ Images/
│  ├─ 1.jpg
│  ├─ 2.jpg
│  ├─ 3.jpeg
│  └─ 4.jpg
├─ yolo-weights/
│  ├─ yolov8n.pt
│  ├─ yolov8m.pt
│  └─ yolov8l.pt
├─ yolo_webcam.py
├─ yolo-basics.py
├─ requirements.txt
└─ README.md
```

## License
Specify your license here (e.g., MIT).
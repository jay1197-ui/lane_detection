# Lane & Vehicle Detection (ADAS)

Real-time Advanced Driver Assistance System (ADAS) combining classical computer vision (lane detection) and deep learning (YOLOv11 for vehicle detection).

![ADAS Demo](adas_demo.gif)

---

## Pipeline

```
Grayscale → Gaussian Blur → Canny Edge Detection → ROI Mask → Hough Lines → Line Averaging
```

1. **Grayscale** — reduces 3 channels to 1, only intensity matters for edge detection
2. **Gaussian Blur** — removes noise so only strong edges survive Canny
3. **Canny Edge Detection** — finds pixels where intensity changes sharply (lane boundaries)
4. **ROI Mask** — triangular mask isolates the lane region, ignores sky/trees/cars
5. **Hough Line Transform** — votes on which edge pixels form straight lines
6. **Line Averaging** — groups lines by slope (left/right), averages into one clean line each

---

## Features
- **Lane Detection:** Classical CV pipeline (Canny edges, Hough lines) to find road boundaries.
- **Vehicle Detection:** YOLOv11 deep learning model to draw bounding boxes around cars.
- **Combined ADAS:** Overlays both lane lines and object bounding boxes on the same video feed in real-time.

---

## Files

| File | Description |
|------|-------------|
| `lane_detect.py` | Runs lane detection on a single image |
| `lane_detect_video.py` | Runs lane detection on a video in real time |
| `detect.py` | Runs YOLOv11 object detection on a video |
| `combined_detect.py` | Runs both Lane Detection and YOLO simultaneously |

---

## Usage

**Lane Detection (Image):**
```bash
python3 lane_detect.py
```

**Lane Detection (Video):**
```bash
python3 lane_detect_video.py
```

**Object Detection (YOLO):**
```bash
python3 detect.py
```

**Combined ADAS System (Lane + YOLO):**
```bash
python3 combined_detect.py
```
Press `q` to quit the video window.

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy ultralytics
```

---

## Tech Stack

- Python 3
- OpenCV
- NumPy
- Ultralytics (YOLO)

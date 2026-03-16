import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from combined_detect import get_lane_overlay

print("Loading model and video...")
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("solidWhiteRight.mp4")

frames = []
count = 0
max_frames = 60 # 2 seconds

print("Processing frames (this may take a minute)...")
while True:
    ret, frame = cap.read()
    if not ret or count >= max_frames:
        break
    
    # Process
    results = model(frame, verbose=False)
    yolo_frame = results[0].plot()
    lane_lines = get_lane_overlay(frame)
    combined = cv2.addWeighted(yolo_frame, 1, lane_lines, 1, 0)
    
    # Resize for GIF to keep file size reasonable
    h, w = combined.shape[:2]
    scale = 500 / w
    resized = cv2.resize(combined, (int(w * scale), int(h * scale)))
    
    # Convert BGR to RGB for PIL
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    frames.append(Image.fromarray(rgb))
    
    count += 1

cap.release()

if frames:
    print(f"Saving {len(frames)} frames to adas_demo.gif...")
    frames[0].save(
        "adas_demo.gif",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=33, # ~30 fps
        optimize=True
    )
    import os
    size_mb = os.path.getsize("adas_demo.gif") / (1024 * 1024)
    print(f"Saved adas_demo.gif successfully! Size: {size_mb:.2f} MB")
else:
    print("No frames processed.")

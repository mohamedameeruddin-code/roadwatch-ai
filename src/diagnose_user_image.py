"""
Crop test image from user's screenshot and run inference at varying confidence levels.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from src.detector import PotholeDetector
from ultralytics import YOLO

screenshot_path = Path(r"C:/Users/Asus/.gemini/antigravity/brain/c4a9771c-2e19-42c9-949a-1691e2eb1e06/.user_uploaded/media_1794774710195_260345be.jpg")
img = Image.open(screenshot_path)
w, h = img.size
print(f"Screenshot size: {w}x{h}")

# In the 1024x512 (approx) screenshot:
# The left road image is located roughly at:
# x: [160, 495], y: [180, 600] scaled by w/1024, h/512
crop_box = (
    int(w * 0.16),
    int(h * 0.18),
    int(w * 0.49),
    int(h * 0.61),
)
road_crop = img.crop(crop_box)
road_crop_path = Path("outputs/predictions/user_test_cropped.jpg")
road_crop.save(road_crop_path)
print(f"Saved road crop to {road_crop_path} (size: {road_crop.size})")

# Load model directly with YOLO
model = YOLO("models/best.pt")

print("\n--- Testing Model Predictions at Confidences 0.01 to 0.40 ---")
for conf in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
    res = model.predict(source=str(road_crop_path), conf=conf, imgsz=640, verbose=False)
    boxes = res[0].boxes
    num_b = len(boxes) if boxes is not None else 0
    confs = [round(float(c), 3) for c in boxes.conf] if num_b > 0 else []
    print(f"Conf {conf:.2f}: {num_b} boxes found. Confidence scores: {confs}")

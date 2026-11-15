"""
Visualize detections on user's cropped image at lower threshold
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from src.preprocessing import draw_bounding_boxes

model = YOLO("models/best.pt")
crop_path = "outputs/predictions/user_test_cropped.jpg"
img_rgb = np.array(Image.open(crop_path).convert("RGB"))

results = model.predict(source=img_rgb, conf=0.03, imgsz=640, verbose=False)
boxes = results[0].boxes

boxes_list = [b.xyxy[0].cpu().numpy().astype(int).tolist() for b in boxes]
confs_list = [float(b.conf[0].cpu().numpy()) for b in boxes]
classes_list = ["pothole"] * len(boxes_list)

print(f"Detected {len(boxes_list)} boxes at conf=0.03:")
for i, (b, c) in enumerate(zip(boxes_list, confs_list), 1):
    print(f"  Box {i}: {b}, Confidence: {c*100:.2f}%")

annotated = draw_bounding_boxes(
    image=img_rgb,
    boxes=boxes_list,
    confidences=confs_list,
    class_names=classes_list,
    box_color_bgr=(36, 75, 60),
    line_thickness=2,
)

out_file = Path("outputs/predictions/user_test_visualized.jpg")
cv2.imwrite(str(out_file), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
print(f"Saved visualization to: {out_file.resolve()}")

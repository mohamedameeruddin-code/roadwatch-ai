"""
Finds test images where potholes are detected and saves annotated images to outputs/predictions/
"""

import glob
from pathlib import Path
import cv2
from src.detector import PotholeDetector


def main():
    detector = PotholeDetector(model_path="models/best.pt")
    print(f"PotholeDetector initialized. Ready: {detector.is_ready}")

    test_images = sorted(glob.glob("data/processed/images/test/India_*.jpg"))
    print(f"Scanning {len(test_images)} test images for detections...")

    out_dir = Path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    positive_count = 0
    # Try with confidence 0.20
    for img_path in test_images:
        res = detector.predict(img_path, conf_threshold=0.20)
        if res.num_detections > 0:
            stem = Path(img_path).stem
            positive_count += 1
            print(f"Found detection in {stem}: {res.num_detections} potholes (max conf: {res.highest_confidence*100:.1f}%, latency: {res.inference_time_ms} ms)")
            out_file = out_dir / f"{stem}_detected.jpg"
            bgr = cv2.cvtColor(res.annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_file), bgr)
            if positive_count >= 8:
                break

    print(f"Total positive annotated samples saved: {positive_count}")


if __name__ == "__main__":
    main()

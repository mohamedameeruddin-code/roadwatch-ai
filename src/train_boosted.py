"""
Boosted Training Pipeline for RoadWatch AI.
Trains YOLOv8n on the RDD2022 India pothole dataset with balanced negative sampling
and transfer learning to achieve high detection confidence.
"""

import os
from pathlib import Path
import random
import shutil
import yaml
import torch
from ultralytics import YOLO

def setup_balanced_training_data():
    """
    Creates a balanced training split containing all positive pothole images
    and a 1:0.5 ratio of negative images, preventing background suppression
    while training the model to recognize potholes with high confidence.
    """
    processed_dir = Path("data/processed")
    train_labels = processed_dir / "labels" / "train"
    train_images = processed_dir / "images" / "train"
    
    # Identify positive vs negative training images
    pos_images = []
    neg_images = []
    
    for lbl_file in sorted(train_labels.glob("*.txt")):
        stem = lbl_file.stem
        img_file = train_images / f"{stem}.jpg"
        if not img_file.exists():
            continue
            
        with open(lbl_file, "r") as f:
            content = f.read().strip()
            
        if content:
            pos_images.append((img_file, lbl_file))
        else:
            neg_images.append((img_file, lbl_file))
            
    print(f"Total training positives: {len(pos_images)}")
    print(f"Total training negatives: {len(neg_images)}")
    
    # Sample balanced negatives (~500 images)
    random.seed(42)
    selected_negs = random.sample(neg_images, min(600, len(neg_images)))
    balanced_set = pos_images + selected_negs
    random.shuffle(balanced_set)
    print(f"Balanced training set size: {len(balanced_set)} ({len(pos_images)} pos, {len(selected_negs)} neg)")
    
    # Create train_boosted directories
    boosted_img_dir = processed_dir / "images" / "train_boosted"
    boosted_lbl_dir = processed_dir / "labels" / "train_boosted"
    boosted_img_dir.mkdir(parents=True, exist_ok=True)
    boosted_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    for img_p, lbl_p in balanced_set:
        dst_img = boosted_img_dir / img_p.name
        dst_lbl = boosted_lbl_dir / lbl_p.name
        if not dst_img.exists():
            try:
                os.link(img_p, dst_img)
            except OSError:
                shutil.copy2(img_p, dst_img)
        if not dst_lbl.exists():
            try:
                os.link(lbl_p, dst_lbl)
            except OSError:
                shutil.copy2(lbl_p, dst_lbl)
                
    # Create dataset_boosted.yaml
    dataset_yaml = Path("data/dataset_boosted.yaml")
    yaml_content = {
        "path": processed_dir.resolve().as_posix(),
        "train": "images/train_boosted",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "pothole"},
    }
    with open(dataset_yaml, "w") as f:
        yaml.dump(yaml_content, f)
        
    return dataset_yaml

def run_boosted_training(epochs=25, batch=16, imgsz=640):
    dataset_yaml = setup_balanced_training_data()
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Starting boosted training on device: {device} for {epochs} epochs...")
    
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=0,
        project="outputs/train",
        name="boosted_yolov8n",
        save=True,
        plots=True,
        patience=12,
        mosaic=0.7,
        mixup=0.1,
        degrees=10.0,
        fliplr=0.5,
        scale=0.5,
        verbose=True,
    )
    
    train_dir = Path("outputs/train/boosted_yolov8n")
    best_weights = train_dir / "weights" / "best.pt"
    if not best_weights.exists():
        candidates = list(Path("outputs/train").glob("**/weights/best.pt"))
        if candidates:
            best_weights = candidates[-1]
            train_dir = best_weights.parent.parent
            
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    target_best = models_dir / "best.pt"
    
    if best_weights.exists():
        shutil.copy2(best_weights, target_best)
        print(f"[Success] Boosted best.pt saved to: {target_best.resolve()}")
        
    # Copy plots
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    for plot_name in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png", "F1_curve.png"]:
        p = train_dir / plot_name
        if p.exists():
            shutil.copy2(p, plots_dir / plot_name)
            
    print("Evaluating boosted model on test set...")
    import subprocess
    subprocess.run([
        r"D:\roadwatch_ai_venv\Scripts\python.exe",
        "-m", "src.evaluate",
        "--model", str(target_best),
        "--data", str(dataset_yaml),
        "--split", "test"
    ])
    
if __name__ == "__main__":
    run_boosted_training()

"""
Model Training Script for RoadWatch AI.
Fine-tunes YOLOv8n on RDD2022 pothole dataset using transfer learning.
"""

import argparse
import os
from pathlib import Path
import shutil
import torch

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


def detect_optimal_device() -> str:
    """Detects available hardware acceleration device (CUDA or CPU)."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] CUDA GPU detected: {gpu_name}")
        return "0"
    else:
        print("[Device] CUDA not available. Executing on CPU.")
        return "cpu"


def train_pothole_model(
    data_yaml: str = "data/dataset.yaml",
    epochs: int = 15,
    batch_size: int = 16,
    img_size: int = 640,
    pretrained_weights: str = "yolov8n.pt",
    project: str = "outputs/train",
    name: str = "yolov8n_pothole",
    device: str = None,
    workers: int = 0,
) -> Path:
    """
    Executes transfer learning on YOLOv8n using the prepared dataset.
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "Ultralytics is not installed. Please install dependencies: `pip install ultralytics`"
        )

    data_path = Path(data_yaml)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset configuration '{data_yaml}' not found. "
            "Please run `python src/dataset.py` first to prepare data and generate dataset.yaml."
        )

    if device is None:
        device = detect_optimal_device()

    print("\n" + "=" * 60)
    print(" RoadWatch AI - YOLOv8n Pothole Model Training")
    print("=" * 60)
    print(f" Dataset Config  : {data_path.resolve()}")
    print(f" Pretrained Model: {pretrained_weights}")
    print(f" Epochs          : {epochs}")
    print(f" Batch Size      : {batch_size}")
    print(f" Image Size      : {img_size}")
    print(f" Device          : {device}")
    print(f" Workers         : {workers}")
    print(f" Output Project  : {project}")
    print("=" * 60 + "\n")

    # Load base YOLOv8n model
    model = YOLO(pretrained_weights)

    # Train model
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=workers,
        project=project,
        name=name,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        # Augmentation hyperparameters
        mosaic=0.5,
        fliplr=0.5,
        scale=0.5,
    )

    # Locate the best weights checkpoint
    # Ultralytics can save in project/name/weights or runs/detect/...
    train_dir = Path(project) / name
    best_weights_path = train_dir / "weights" / "best.pt"

    if not best_weights_path.is_file():
        # Search runs/ or project/ recursively
        candidates = list(Path(".").glob(f"**/{name}*/weights/best.pt"))
        if candidates:
            best_weights_path = candidates[0]
            train_dir = best_weights_path.parent.parent

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    target_best = models_dir / "best.pt"

    if best_weights_path.is_file():
        shutil.copy2(best_weights_path, target_best)
        print(f"\n[Success] Training complete!")
        print(f"Best weights successfully copied to: {target_best.resolve()}")
    else:
        print(f"[Warning] Best weights not found at {best_weights_path}. Check {train_dir}")

    # Copy output plots to outputs/plots/
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    for plot_name in [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
    ]:
        src_plot = train_dir / plot_name
        if src_plot.is_file():
            shutil.copy2(src_plot, plots_dir / plot_name)
            print(f"Copied plot {plot_name} -> {plots_dir / plot_name}")

    return target_best


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n Pothole Detector")
    parser.add_argument("--data", type=str, default="data/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--pretrained", type=str, default="yolov8n.pt", help="Pretrained base model")
    parser.add_argument("--device", type=str, default=None, help="Device to run training (cuda or cpu)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 for Windows)")
    parser.add_argument("--project", type=str, default="outputs/train", help="Training runs output directory")
    args = parser.parse_args()

    train_pothole_model(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        pretrained_weights=args.pretrained,
        project=args.project,
        device=args.device,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

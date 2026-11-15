# RoadWatch AI – CNN-Based Road Pothole Detection System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-244B3C.svg)](https://www.python.org/)
[![PyTorch 2.6 CUDA](https://img.shields.io/badge/PyTorch-2.6%20CUDA-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-0055FF.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

RoadWatch AI is an intelligent computer vision application designed to inspect roadway photographs and automatically detect potholes using a Convolutional Neural Network (Ultralytics YOLOv8n) fine-tuned on the Road Damage Dataset 2022 (RDD2022).

Developed as an academic demonstration and engineering prototype, the project features a decoupled architecture separating dataset preparation, training, evaluation, inference, and an accessible light-themed Streamlit presentation layer.

---

## 📊 Empirical Model Evaluation Results (Test Set)

Evaluated on **1,166 held-out test images** (containing 445 pothole ground-truth instances and 956 negative clean-road images):

| Metric | Empirical Score | Description |
| :--- | :--- | :--- |
| **Precision** | **28.31%** | True positive detection accuracy |
| **Recall** | **29.21%** | Detection sensitivity |
| **F1-Score** | **28.76%** | Harmonic mean of precision & recall |
| **mAP@0.5** | **22.67%** | Mean Average Precision at IoU = 0.50 |
| **mAP@0.5:0.95** | **8.18%** | Primary COCO-standard benchmark |
| **Inference Latency** | **7.17 ms** | Forward pass per frame on NVIDIA RTX 2050 |
| **Total Latency** | **11.60 ms** | Complete pipeline (Preprocess + Inference + NMS) |

---

## 📁 Project Architecture & Directory Layout

```
roadwatch-ai/
│
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python runtime dependencies
├── README.md                   # Project documentation & instructions
├── config.toml                 # Application settings & color definitions
│
├── .streamlit/
│   └── config.toml             # Streamlit server & light-theme parameters
│
├── docs/
│   ├── 01_SRS.md               # Software Requirements Specification
│   ├── 02_PRD.md               # Product Requirements Document
│   ├── 03_TRD.md               # Technical Requirements Document
│   └── 04_UI_SPEC.md           # UI/UX Frontend Design Specification
│
├── data/
│   ├── raw/                    # Raw RDD2022 dataset directory
│   ├── processed/              # Processed YOLO dataset (7,706 images)
│   └── dataset.yaml            # YOLO dataset configuration
│
├── models/
│   └── best.pt                 # Trained YOLOv8n checkpoint weights (3.2 MB)
│
├── src/
│   ├── __init__.py             # Package initializer
│   ├── dataset.py              # VOC XML parser, D40 extractor & YOLO splitter
│   ├── extract_rdd_india.py    # India subset extractor & label mapper from RDD_SPLIT
│   ├── preprocessing.py        # Image validation, letterboxing & box rendering
│   ├── train.py                # Transfer learning training pipeline (CUDA/CPU)
│   ├── evaluate.py             # Empirical test set evaluation pipeline
│   ├── detector.py             # Standalone reusable PotholeDetector engine
│   ├── generate_sample_predictions.py # Sample detection exporter
│   └── utils.py                # Custom CSS, card renderers & metric formatters
│
├── outputs/
│   ├── predictions/            # Exported detection outputs (annotated images)
│   ├── metrics/                # evaluation_metrics.json & evaluation_summary.csv
│   └── plots/                  # BoxPR_curve.png, confusion_matrix.png, results.png
│
└── tests/
    ├── test_dataset.py         # Unit tests for XML parsing and YOLO conversion
    ├── test_preprocessing.py   # Unit tests for validation & resizing
    └── test_detector.py        # Unit tests for inference & fallback handling
```

---

## 🚀 Quick Start: Launching the Web Application

The environment is already prepared with PyTorch CUDA and the trained model weights are saved at `models/best.pt`.

To launch the Streamlit web dashboard:
```bash
# From the project directory:
.venv\Scripts\streamlit.exe run app.py
```
Open your browser to: **`http://localhost:8501`**

---

## 🔄 End-to-End Pipeline Workflow

### Step 1: Dataset Extraction & Preparation
Extract the India subset from `archive (2).zip`, map class `D40` to `pothole` (class 0), and preserve negative samples:
```bash
.venv\Scripts\python.exe -m src.extract_rdd_india
```
Generates 7,706 images across `train` (5,368), `val` (1,172), and `test` (1,166) splits.

### Step 2: Model Training (Transfer Learning)
Train YOLOv8n on CUDA GPU:
```bash
.venv\Scripts\python.exe src/train.py --data data/dataset.yaml --epochs 15 --batch 16 --workers 0 --device 0
```
Saves best weights to `models/best.pt`.

### Step 3: Test Set Evaluation
Benchmark `models/best.pt` on the held-out test split:
```bash
.venv\Scripts\python.exe src/evaluate.py --model models/best.pt --data data/dataset.yaml --split test --device 0
```
Generates `outputs/metrics/evaluation_metrics.json` and diagnostic curves in `outputs/plots/`.

### Step 4: Sample Prediction Generation
```bash
.venv\Scripts\python.exe -m src.generate_sample_predictions
```
Annotated images are saved in `outputs/predictions/`.

---

## 🎨 UI & Theme Specification
RoadWatch AI strictly follows a **Light Theme** aligned with civil engineering applications:
- **Primary Color:** Forest Green (`#244B3C`)
- **Accent Color:** Muted Sage Green (`#6A8F78`)
- **Background:** Soft Off-White (`#F6F8F5`)
- **Surface Cards:** Pure White (`#FFFFFF`)
- **Borders:** Soft Grey-Green (`#D9E2DA`)

---

## 🧪 Running Unit Tests
Execute the automated test suite with Python's standard `unittest` runner:
```bash
.venv\Scripts\python.exe -m unittest discover tests
```

---

## 📜 Academic Disclaimer
RoadWatch AI is developed strictly for educational, research, and technical prototyping purposes. The system does not guarantee general roadway safety, structural integrity certification, or suitability for autonomous vehicular navigation. Road condition classifications should be verified by certified civil engineering personnel prior to municipal maintenance scheduling.

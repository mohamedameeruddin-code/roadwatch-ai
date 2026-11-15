# Technical Requirements Document (TRD)
## RoadWatch AI – CNN-Based Road Pothole Detection System

---

### Document Control
- **Document Title:** Technical Requirements Document (TRD)
- **Product Name:** RoadWatch AI
- **Version:** 1.0.0
- **Document Status:** Complete & Approved
- **Reference Documents:** [01_SRS.md](01_SRS.md), [02_PRD.md](02_PRD.md)

---

## 1. Technical Objective

The technical objective of RoadWatch AI is to engineer a modular, reproducible, and verifiable deep learning pipeline for road pothole detection using a Convolutional Neural Network (CNN) architecture.

The technical architecture encompasses:
1. **Automated Dataset Ingestion & Conversion:** Parsing Pascal VOC XML annotations from the Road Damage Dataset 2022 (RDD2022), filtering category `D40` (pothole), normalizing coordinates, and formatting into YOLO standard annotations while retaining negative samples.
2. **Deterministic Splitting:** Generating non-overlapping 70% train, 15% validation, and 15% test splits with fixed random seeds.
3. **Transfer Learning Pipeline:** Fine-tuning Ultralytics YOLOv8n on 640x640 resolution inputs with data augmentation and checkpointing.
4. **Empirical Evaluation Pipeline:** Benchmarking the best checkpoint on the held-out test split, exporting formal metrics (Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95) and graphical artifacts.
5. **Decoupled Inference Engine:** Providing a standalone Python class (`PotholeDetector`) that performs preprocessing, inference, confidence filtering, and annotation overlay without UI coupling.
6. **Web Presentation:** Integrating the detector into an accessible, responsive Streamlit interface styled with custom CSS according to a strict light theme.

---

## 2. Technology Stack

| Layer | Component / Library | Version | Technical Justification |
| :--- | :--- | :--- | :--- |
| **Language** | Python | >= 3.10, <= 3.12 | Standard runtime for modern computer vision and deep learning ecosystems. |
| **Deep Learning Framework** | PyTorch (`torch`, `torchvision`) | >= 2.0.0 | High-performance tensor computation with native CUDA and MPS support. |
| **CNN Architecture** | Ultralytics (`ultralytics`) | >= 8.0.0 | Industry-standard implementation of YOLOv8n providing transfer learning, anchor-free detection heads, and optimized postprocessing. |
| **Computer Vision** | OpenCV (`opencv-python-headless`) | >= 4.8.0 | High-speed image decoding, color conversion, bounding box drawing, and letterbox padding. |
| **Image Processing** | Pillow (`PIL`) | >= 10.0.0 | In-memory image serialization and compatibility with Streamlit upload streams. |
| **Numerical & Tabular** | NumPy, Pandas | >= 1.24.0, >= 2.0.0 | Array manipulation, coordinate conversions, and tabular metrics management. |
| **User Interface** | Streamlit | >= 1.28.0 | Python-native reactive web interface enabling rapid demonstration without JavaScript overhead. |
| **Visualization** | Matplotlib, Seaborn | >= 3.7.0, >= 0.12.0 | Generation of training loss trajectories, PR curves, and confusion matrix heatmaps. |
| **Configuration** | PyYAML | >= 6.0.0 | YOLO dataset configuration management (`dataset.yaml`). |

---

## 3. Dataset Specification

### 3.1 Source & Format
- **Dataset:** Road Damage Dataset 2022 (RDD2022), Seki Lab (https://github.com/sekilab/RoadDamageDetector).
- **Target Geographic Subset:** India road damage subset (e.g., `India/` containing `images/` and `annotations/xmls/`).
- **Raw Annotation Standard:** Pascal VOC XML format containing `<annotation>`, `<size>`, and `<object>` elements.

### 3.2 Target Class Mapping
In RDD2022, road damage is categorized into multiple classes:
- `D00`: Wheel mark part (longitudinal crack)
- `D10`: Equal interval (transverse crack)
- `D20`: Partial pavement crack (alligator crack)
- **`D40`: Pothole (TARGET CLASS)**

All objects labeled `D40` are mapped to:
- Class Index: `0`
- Class Label: `pothole`

All other damage categories are excluded. Images containing no `D40` instances are retained as empty text annotation files (negative training examples) to prevent false positives on intact or non-pothole road surfaces.

### 3.3 YOLO Annotation Format Conversion
Given image width $W$ and height $H$, for every bounding box $[X_{min}, Y_{min}, X_{max}, Y_{max}]$:
$$\text{center}_x = \frac{X_{min} + X_{max}}{2 \times W}$$
$$\text{center}_y = \frac{Y_{min} + Y_{max}}{2 \times H}$$
$$\text{width} = \frac{X_{max} - X_{min}}{W}$$
$$\text{height} = \frac{Y_{max} - Y_{min}}{H}$$

Each line in the `.txt` label file adheres to:
```
0 <center_x> <center_y> <width> <height>
```

### 3.4 Dataset Partitioning
The dataset is partitioned from the labeled training collection into three non-overlapping subsets:
- **Training Set (70%):** Model parameter optimization.
- **Validation Set (15%):** Hyperparameter tuning, early stopping, and model checkpointing (`best.pt`).
- **Test Set (15%):** Unbiased empirical performance evaluation.

A deterministic random seed (`seed=42`) is enforced across splits to ensure full experimental reproducibility.

> [!IMPORTANT]
> Official RDD2022 test set folders lack ground-truth XML annotations and are used only for unannotated competitions. Therefore, RoadWatch AI creates its held-out 15% test split strictly from the verified ground-truth collection.

---

## 4. System Architecture

RoadWatch AI is structured in 6 decoupled architectural layers:

```
+------------------------------------------------------------------------+
| 1. User Interface Layer (Streamlit App)                                |
|    - Theme Engine (custom CSS, light palette)                          |
|    - Widgets (File Uploader, Slider, Download Button, Metrics, Tabs)   |
+-----------------------------------+------------------------------------+
                                    |
+-----------------------------------v------------------------------------+
| 2. Image Processing Layer (src/preprocessing.py)                       |
|    - Format validation, Letterbox resize (640x640), Color conversions  |
+-----------------------------------+------------------------------------+
                                    |
+-----------------------------------v------------------------------------+
| 3. CNN Model Layer (src/detector.py)                                   |
|    - YOLOv8n network loading (models/best.pt)                          |
|    - Non-Maximum Suppression (NMS), Confidence thresholding            |
|    - Bounding box denormalization & annotation drawing                 |
+------------------------------------------------------------------------+
         ^                                              |
         | Checkpoint Loading                           | Generates
         |                                              v
+--------+--------------------------+    +-------------------------------+
| 4. Training Layer                 |    | 6. Results Management Layer   |
|    (src/train.py, src/dataset.py) |    |    (outputs/predictions,      |
|    - VOC XML conversion           |    |     outputs/metrics,          |
|    - Ultralytics fine-tuning      |    |     outputs/plots)            |
+-----------------------------------+    +-------------------------------+
         |
         v Evaluates
+-----------------------------------+
| 5. Evaluation Layer               |
|    (src/evaluate.py)              |
|    - Precision, Recall, F1, mAP   |
|    - Confusion matrix, PR curve   |
+-----------------------------------+
```

---

## 5. Project Folder Structure

```
roadwatch-ai/
│
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python runtime dependencies
├── README.md                   # Comprehensive project setup & documentation
├── config.toml                 # Application configuration & default paths
│
├── .streamlit/
│   └── config.toml             # Streamlit server and theme definitions
│
├── docs/
│   ├── 01_SRS.md               # Software Requirements Specification
│   ├── 02_PRD.md               # Product Requirements Document
│   ├── 03_TRD.md               # Technical Requirements Document
│   └── 04_UI_SPEC.md           # User Interface Specification
│
├── data/
│   ├── raw/                    # Raw RDD2022 download directory (gitignored)
│   ├── processed/              # Formatted YOLO dataset (images/ & labels/ for train/val/test)
│   └── dataset.yaml            # YOLO dataset descriptor pointing to processed data
│
├── models/
│   └── best.pt                 # Saved YOLOv8n checkpoint weights (gitignored)
│
├── src/
│   ├── __init__.py             # Module initializer
│   ├── dataset.py              # VOC XML parser, D40 extractor, YOLO conversion & splitting
│   ├── preprocessing.py        # Image validation, letterboxing, tensor transformation
│   ├── train.py                # YOLOv8n transfer learning runner
│   ├── evaluate.py             # Empirical test evaluation & metrics serialization
│   ├── detector.py             # Reusable inference engine class (PotholeDetector)
│   └── utils.py                # Visual styling, plot generation, file utilities
│
├── outputs/
│   ├── predictions/            # Exported detection outputs (annotated images)
│   ├── metrics/                # evaluation_metrics.json and summary tables
│   └── plots/                  # confusion_matrix.png, loss_curve.png, pr_curve.png
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py         # Unit tests for VOC parsing and YOLO conversion
│   ├── test_preprocessing.py   # Unit tests for image validation and resizing
│   └── test_detector.py        # Unit tests for detector initialization & inference
│
└── .gitignore                  # Git exclusions for datasets, weights, and caches
```

---

## 6. Detailed Module Specifications

### 6.1 `src/dataset.py`
- **Class:** `RDDDatasetParser`
- **Functions:**
  - `parse_voc_xml(xml_path: str) -> List[Tuple[str, List[int]]]`: Extracts bounding boxes matching class `D40`.
  - `convert_to_yolo(box: List[int], img_w: int, img_h: int) -> Tuple[float, float, float, float]`: Converts `[xmin, ymin, xmax, ymax]` to YOLO format.
  - `prepare_dataset(raw_dir: str, output_dir: str, split_ratios=(0.70, 0.15, 0.15), seed=42)`: Iterates over images and XMLs, converts annotations, writes negative label files where no potholes exist, splits dataset deterministically, and writes `dataset.yaml`.

### 6.2 `src/preprocessing.py`
- **Functions:**
  - `validate_image_file(file_path_or_bytes) -> Tuple[bool, str]`: Checks file extension, verifies headers with PIL, ensures resolution $> 32 \times 32$ and file size $< 15\text{ MB}$.
  - `letterbox_image(image: np.ndarray, target_size=(640, 640)) -> Tuple[np.ndarray, float, Tuple[int, int]]`: Maintains aspect ratio with pad color `(114, 114, 114)`.
  - `denormalize_coordinates(boxes, scale, pad, original_shape)`: Maps 640x640 model outputs back to original input image resolution.

### 6.3 `src/train.py`
- **Module Purpose:** CLI-executable script for YOLOv8n fine-tuning.
- **Arguments:**
  - `--data`: Path to `dataset.yaml` (default: `data/dataset.yaml`).
  - `--epochs`: Integer number of epochs (default: `50`).
  - `--batch-size`: Batch size (default: `16`, fallback to `8` or `4` on lower VRAM).
  - `--img-size`: Resolution (default: `640`).
  - `--device`: Automatic detection (`cuda:0` if available, else `cpu`).
  - `--project`: Output directory (default: `outputs/train`).
- **Post-Training Action:** Automatically copies the best weights (`runs/detect/.../weights/best.pt`) to `models/best.pt`.

### 6.4 `src/evaluate.py`
- **Module Purpose:** Evaluates `models/best.pt` on the held-out test split defined in `dataset.yaml`.
- **Metrics Computed:**
  - Precision ($P$)
  - Recall ($R$)
  - F1-Score ($F1 = \frac{2 \cdot P \cdot R}{P + R}$)
  - mAP@0.5 and mAP@0.5:0.95
  - Average inference latency (ms)
- **Serialization:** Saves structured results to `outputs/metrics/evaluation_metrics.json` and copies Ultralytics evaluation curves (PR curve, F1 curve, confusion matrix) to `outputs/plots/`.

### 6.5 `src/detector.py`
- **Class:** `PotholeDetector`
- **Methods:**
  - `__init__(model_path: str = "models/best.pt", device: str = None)`: Loads model; flags `is_ready = False` gracefully if model file is not found.
  - `predict(image: Union[str, np.ndarray, Image.Image], conf_threshold: float = 0.35) -> DetectionResult`: Runs YOLO inference, applies confidence filtering, extracts bounding boxes `[xmin, ymin, xmax, ymax]`, confidence scores, computes inference latency, and generates annotated OpenCV image.
- **Data Transfer Object:**
  ```python
  @dataclass
  class DetectionResult:
      annotated_image: np.ndarray
      num_detections: int
      boxes: List[List[float]]  # [xmin, ymin, xmax, ymax]
      confidences: List[float]
      classes: List[str]
      inference_time_ms: float
      is_success: bool
      error_message: Optional[str] = None
  ```

### 6.6 `src/utils.py`
- Helper utilities:
  - `load_css(css_file_path: str)`: Injects custom CSS into Streamlit DOM.
  - `format_bounding_box_table(result: DetectionResult) -> pd.DataFrame`: Formats detection coordinates into clean Pandas DataFrame.
  - `save_annotated_image(image: np.ndarray, output_path: str)`: Saves processed image to disk.

---

## 7. Technical Workflow

```
+--------------------------------------------------------------+
| 1. Dataset Acquisition                                       |
|    Download RDD2022 India dataset into data/raw/             |
+------------------------------+-------------------------------+
                               |
+------------------------------v-------------------------------+
| 2. Conversion & Deterministic Splitting                      |
|    python src/dataset.py --raw data/raw --out data/processed  |
|    Generates train (70%), val (15%), test (15%) + dataset.yaml|
+------------------------------+-------------------------------+
                               |
+------------------------------v-------------------------------+
| 3. Model Fine-Tuning (Transfer Learning)                     |
|    python src/train.py --epochs 50 --batch-size 16           |
|    Saves checkpoint to models/best.pt                        |
+------------------------------+-------------------------------+
                               |
+------------------------------v-------------------------------+
| 4. Test Set Evaluation & Plot Generation                     |
|    python src/evaluate.py --data data/dataset.yaml           |
|    Saves outputs/metrics/evaluation_metrics.json and plots   |
+------------------------------+-------------------------------+
                               |
+------------------------------v-------------------------------+
| 5. Streamlit Application Launch                              |
|    streamlit run app.py                                      |
|    User uploads image, adjusts threshold, views detections   |
+--------------------------------------------------------------+
```

---

## 8. Error Handling and Edge Cases

| Scenario | System Response & Handling |
| :--- | :--- |
| **Model weights (`models/best.pt`) missing** | Detector sets `is_ready = False`. UI displays an informative alert: *"Model weights not found. Please train the model using `python src/train.py` or place pretrained weights in `models/best.pt`."* Application does not crash. |
| **Corrupted / unreadable image uploaded** | `validate_image_file` catches PIL decode exceptions and returns `(False, "Uploaded file is corrupted or not a valid image.")`. |
| **Unsupported file format (e.g., .gif, .bmp)**| UI file uploader restricts extensions; backend rejects with: *"Supported formats are JPG, JPEG, and PNG."* |
| **Image without potholes** | Model yields zero detections. UI displays clean state: *"No potholes detected at the selected confidence threshold."* Metric cards show `Total Detected: 0`. |
| **Missing evaluation metrics file** | Model Performance tab inspects `outputs/metrics/evaluation_metrics.json`. If missing, displays: *"Model evaluation results are not available yet. Run `python src/evaluate.py` to generate metrics."* |
| **CUDA out of memory during training** | `train.py` catches memory exceptions and suggests reducing batch size to 8 or 4. |

---

## 9. Testing Strategy

1. **Unit Testing (`tests/test_dataset.py`):**
   - Validates XML coordinate parsing for edge cases (zero-width boxes, out-of-boundary coordinates).
   - Verifies YOLO normalization formula correctness.
   - Asserts 70/15/15 split distribution and verifies no overlapping file stems between splits.
2. **Preprocessing Tests (`tests/test_preprocessing.py`):**
   - Tests `letterbox_image` aspect ratio preservation and coordinate inversion.
   - Tests file validation with valid, invalid, and oversized mock images.
3. **Inference Engine Tests (`tests/test_detector.py`):**
   - Tests `PotholeDetector` instantiation with and without model weights.
   - Asserts return types of `predict()` match `DetectionResult` schema.
4. **End-to-End Web App Smoke Test:**
   - Launches Streamlit headless and checks page component rendering.

---

## 10. Dependencies and Installation

Dependencies defined in `requirements.txt`:
```text
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
streamlit>=1.28.0
opencv-python-headless>=4.8.0
numpy>=1.24.0,<2.0.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=10.0.0
pyyaml>=6.0.0
```

---

## 11. Deployment and Execution

### Local Development Setup:
```bash
# 1. Create and activate Python virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Run dataset conversion
python src/dataset.py --raw_dir data/raw --out_dir data/processed

# 4. (Optional) Train model
python src/train.py --epochs 50 --batch 16

# 5. (Optional) Evaluate model
python src/evaluate.py

# 6. Launch Streamlit UI
streamlit run app.py
```

---

## 12. Integrity and Ethics Guidelines
- **Real CNN Implementation:** The project relies strictly on YOLOv8n deep learning inference. Mocked or heuristic bounding box generators are strictly prohibited.
- **No Fabricated Metrics:** Evaluation numbers displayed in the interface are loaded directly from disk files generated during actual validation passes.
- **Version Control Exclusions:** Raw dataset archives and binary `.pt` checkpoint weights are excluded via `.gitignore` to maintain a lean code repository.

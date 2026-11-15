# Software Requirements Specification (SRS)
## RoadWatch AI – CNN-Based Road Pothole Detection System

---

### Document Control
- **Document Title:** Software Requirements Specification for RoadWatch AI
- **Version:** 1.0.0
- **Project Type:** Academic Deep Learning Prototype & Demonstration
- **Date:** November 2026
- **Status:** Approved for Implementation

---

## 1. Introduction

### 1.1 Purpose
The purpose of this Software Requirements Specification (SRS) document is to delineate the functional and non-functional requirements of **RoadWatch AI**, an intelligent road pothole detection application. This document serves as the formal specification for developers, academic evaluators, and system testers, defining the operational behavior, system constraints, user interactions, and interface requirements for the system.

### 1.2 Project Scope
RoadWatch AI is an academic deep learning prototype designed to automatically inspect digital road imagery, detect road surface hazards (specifically potholes), annotate detected regions with bounding boxes, and display associated prediction confidence scores. The core model employs a Convolutional Neural Network (CNN) architecture based on Ultralytics YOLOv8n, trained via transfer learning on the publicly available Road Damage Dataset 2022 (RDD2022).

The application provides:
- Single-image upload and format validation.
- Computer vision-based pothole detection and localized bounding box annotation.
- Interactive confidence threshold adjustment.
- Comprehensive detection metric reporting (detection counts, confidence levels, inference latency).
- Export capability for annotated results.
- Model performance evaluation metrics presentation based on empirical test data.

**Scope Boundaries:**
- The system is designed as a standalone demonstration prototype and does not connect to real-time municipal traffic control or dispatch systems.
- The system does not guarantee general road safety or structural road certification.
- User management, subscription monetization, and multi-tenant cloud orchestration are intentionally excluded from the academic scope.

### 1.3 Intended Audience
- **Academic Evaluators & Faculty:** To assess algorithmic implementation, architectural design, and deep learning engineering practices.
- **Machine Learning Engineers & Developers:** To implement, refine, test, and maintain the model pipeline and web interface.
- **Researchers & Students:** To benchmark pothole detection methodologies and explore transfer learning on road damage datasets.
- **Civil Infrastructure Operators:** To review prototype capabilities for automated road asset monitoring.

### 1.4 Definitions, Acronyms, and Abbreviations
- **AI:** Artificial Intelligence.
- **CNN:** Convolutional Neural Network.
- **YOLO:** You Only Look Once (real-time object detection architecture).
- **YOLOv8n:** YOLO version 8 nano (lightweight CNN model variant).
- **RDD2022:** Road Damage Dataset 2022.
- **Pascal VOC:** Visual Object Classes XML annotation standard.
- **mAP:** Mean Average Precision.
- **IoU:** Intersection over Union.
- **UI:** User Interface.
- **FPS:** Frames Per Second.
- **RGB:** Red, Green, Blue color space.

### 1.5 References
1. Seki Lab, "Road Damage Detector (RDD2022)", GitHub Repository: https://github.com/sekilab/RoadDamageDetector
2. Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 (Version 8.0.0) [Computer software].
3. IEEE Std 830-1998, IEEE Recommended Practice for Software Requirements Specifications.
4. Arya, D. et al., "Global Road Damage Detection: State-of-the-art Solutions," IEEE Big Data Conference, 2022.

---

## 2. Overall Description

### 2.1 Product Perspective
RoadWatch AI operates as a self-contained web-based software application powered by a Python backend and a lightweight Streamlit interface. The system bridges deep learning model artifacts with an intuitive user interface, allowing non-technical evaluators to execute model inference without requiring terminal access or programming knowledge.

```
+--------------------------------------------------------------+
|                     User Browser (Client)                    |
+------------------------------+-------------------------------+
                               | HTTP / WebSocket
+------------------------------v-------------------------------+
|                 Streamlit Presentation Layer                 |
|       (Image Upload, Controls, Visualizations, Tables)       |
+------------------------------+-------------------------------+
                               | Function Calls
+------------------------------v-------------------------------+
|               RoadWatch Inference Engine (Python)            |
|       (Preprocessing, Tensor Conversion, Postprocessing)      |
+------------------------------+-------------------------------+
                               | PyTorch / Ultralytics
+------------------------------v-------------------------------+
|               YOLOv8n CNN Weights & Checkpoints              |
|                     (models/best.pt)                         |
+--------------------------------------------------------------+
```

### 2.2 Product Functions
1. **Image Ingestion:** Receive road photographic files from users via file upload.
2. **File Validation:** Verify that files match accepted image formats (JPG, JPEG, PNG) and adhere to resolution/size constraints.
3. **Computer Vision Inference:** Feed validated images through the fine-tuned YOLOv8n network to infer pothole presence and spatial bounding coordinates.
4. **Interactive Parameter Control:** Allow users to dynamically vary confidence thresholds (0.05 to 1.00) to inspect detection sensitivity.
5. **Annotation & Rendering:** Overlay bounding boxes, class labels ("pothole"), and confidence percentages directly onto the output image.
6. **Result Reporting:** Tabulate detection count, individual bounding box coordinates, and processing execution time.
7. **Artifact Export:** Allow the user to download the generated annotated image.
8. **Empirical Evaluation Display:** Present verified validation and test metrics (Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, and confusion matrices).

### 2.3 User Characteristics
The target users encompass university students, instructors, project evaluators, and road inspection inspectors. Users are expected to have basic familiarity with web interfaces and general concepts of visual object detection (such as confidence thresholds and bounding boxes). No prior programming skills are mandated for end users.

### 2.4 Operating Environment
- **Operating Systems:** Windows 10/11, Ubuntu 20.04/22.04 LTS, macOS 12+.
- **Runtime Environment:** Python 3.10 to 3.12.
- **Hardware Acceleration (Inference):** CPU supported (x86_64, ARM64); CUDA-compatible GPU optional for acceleration.
- **Client Web Browsers:** Modern Chromium-based browsers (Chrome 100+, Edge 100+), Firefox 100+, Safari 15+.

### 2.5 Design Constraints
- Must be implemented entirely in Python for algorithmic coherence.
- Must utilize a real CNN architecture (Ultralytics YOLOv8n) without mocked heuristic detections.
- Must execute on consumer-grade workstation CPUs within acceptable latency thresholds (< 2.5 seconds per 640x640 frame on standard CPU).
- UI must follow an accessible Light Theme conforming to specified architectural palettes.

### 2.6 Assumptions and Dependencies
- The system assumes input images are standard perspective road surface photographs in clear to moderate lighting.
- The model depends on PyTorch and Ultralytics runtime libraries.
- Training depends upon the RDD2022 dataset subset containing D40 (pothole) annotations.
- If the trained model file `models/best.pt` is absent, the system must degrade gracefully and display an informative notice rather than crash.

---

## 3. System Features

1. **Upload Road Images:** Drag-and-drop or file selector interaction supporting single image uploads.
2. **Validate Uploaded Image Formats:** Automatic extension and binary integrity validation for `.jpg`, `.jpeg`, and `.png` files under 15 MB.
3. **Process Images Using a Trained CNN Model:** End-to-end normalization, letterbox resizing to 640x640, and forward pass tensor computation.
4. **Detect Potholes in Road Images:** Localize damage coordinates conforming to class D40.
5. **Display Bounding Boxes:** Render clear bounding boxes with custom styling around detected potholes.
6. **Display Pothole Confidence Scores:** Overlay calculated percentage confidences above each bounding box.
7. **Adjust Confidence Threshold:** Real-time slider enabling confidence cutoff adjustment between 0.05 and 1.00 (default: 0.35).
8. **Display Number of Detected Potholes:** Instant summary metric showing aggregate pothole count.
9. **Download Processed Image:** Generate high-resolution annotated image downloadable via a single click.
10. **Display Model Evaluation Metrics:** Dedicated analytical tab displaying empirical test performance (Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, Loss curves, and Confusion Matrix).

---

## 4. Functional Requirements

| Req ID | Requirement Title | Detailed Description | Priority |
| :--- | :--- | :--- | :--- |
| **FR-01** | Image Input Interface | The system shall provide an intuitive file uploader accepting files with extensions `.jpg`, `.jpeg`, and `.png`. | P0 (Must Have) |
| **FR-02** | Image Validation | The system shall verify file headers and dimensions. If a file is corrupt, empty, or exceeds 15 MB, the system shall reject it with a descriptive error message. | P0 (Must Have) |
| **FR-03** | Image Preprocessing | The system shall resize input images to 640x640 pixels using letterbox padding while preserving original aspect ratios before model ingestion. | P0 (Must Have) |
| **FR-04** | CNN Model Loading | The system shall load YOLOv8n model weights from `models/best.pt`. If weights are missing, the system shall display an alert instructing the user to train or provide model weights. | P0 (Must Have) |
| **FR-05** | Object Detection Inference | The system shall perform forward pass inference on the preprocessed image using the loaded YOLOv8n network to detect pothole instances. | P0 (Must Have) |
| **FR-06** | Confidence Threshold Filtering | The system shall provide a slider control (range 0.05 - 1.00, step 0.05, default 0.35) and filter out all detection candidates below the selected threshold. | P0 (Must Have) |
| **FR-07** | Bounding Box Rendering | The system shall draw bounding box outlines and labels ("pothole: XX.X%") on the original resolution image using OpenCV or PIL. | P0 (Must Have) |
| **FR-08** | Side-by-Side Visual Comparison | The system shall display the original uploaded image alongside the annotated image in a responsive two-column layout. | P0 (Must Have) |
| **FR-09** | Metric Summary Dashboard | The system shall display summary cards for: Total Potholes Detected, Highest Confidence Score, Average Confidence Score, and Inference Time (milliseconds). | P1 (Should Have) |
| **FR-10** | Detection Tabular Breakdown | The system shall render a structured table enumerating each detection's index, class label, confidence score, and pixel bounding coordinates `[xmin, ymin, xmax, ymax]`. | P1 (Should Have) |
| **FR-11** | Zero-Detection Handling | When no potholes meet the threshold, the system shall display the message: *"No potholes detected at the selected confidence threshold."* | P0 (Must Have) |
| **FR-12** | Annotated Image Export | The system shall offer a download button allowing users to save the annotated PNG/JPEG image locally. | P1 (Should Have) |
| **FR-13** | Model Performance Reporting | The system shall provide an evaluation tab that reads and displays verified evaluation metrics (Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95) from a persistent metrics file. | P1 (Should Have) |
| **FR-14** | Analytical Plot Rendering | The system shall render confusion matrix, precision-recall curve, and training loss plots if generated during the training phase. | P2 (Could Have) |
| **FR-15** | Missing Metrics Fallback | If the model evaluation metrics file does not exist, the evaluation tab shall display: *"Model evaluation results are not available yet."* | P0 (Must Have) |

---

## 5. Non-Functional Requirements

### 5.1 Performance
- **Inference Latency:** Average inference latency for a standard 640x640 image shall not exceed 2500 ms on a modern 4-core consumer CPU and 100 ms on an NVIDIA GPU (RTX 3060 or equivalent).
- **Page Load Time:** Initial application UI launch shall render in under 3.0 seconds under standard local hosting conditions.

### 5.2 Reliability & Robustness
- **Fault Tolerance:** Invalid image files, unreadable formats, or missing model checkpoints shall not cause the application process to terminate. All exceptions must be caught and communicated via user-friendly UI banners.
- **Determinism:** Given the same image and identical confidence threshold, the inference output and bounding coordinates shall be deterministic.

### 5.3 Usability
- **Design Paradigm:** Clean, academic, and professional Light Theme interface conforming to the specified civil engineering palette.
- **Guidance:** Plain-language hints and tooltips explaining how to adjust confidence and interpret bounding boxes.
- **Responsiveness:** Layout must gracefully adjust to screen resolutions ranging from 1024x768 (tablets/laptops) to 3840x2160 (4K monitors).

### 5.4 Maintainability & Modularity
- **Code Organization:** Clear separation of concerns into distinct modules (`src/dataset.py`, `src/preprocessing.py`, `src/detector.py`, `src/train.py`, `src/evaluate.py`, `src/utils.py`).
- **Extensibility:** The inference module (`PotholeDetector`) must operate as a standalone class capable of being imported into CLI scripts or alternate web frameworks (such as FastAPI or Flask).

### 5.5 Security
- **Local Sandbox Execution:** Uploaded images shall be processed in-memory or in temporary session storage without unauthorized filesystem persistence.
- **Input Sanitization:** Filenames must be stripped of traversal tokens (`../`) before file handling.

### 5.6 Scalability & Compatibility
- **Cross-Platform:** The codebase must execute identically on Windows, Linux, and macOS without platform-specific binary dependencies.
- **Python Compatibility:** Compatible with standard CPython runtimes 3.10, 3.11, and 3.12.

---

## 6. External Interface Requirements

### 6.1 User Interface
The user interface is built on Streamlit with custom CSS. It consists of:
- **Header:** System title ("RoadWatch AI"), subtitle ("Intelligent Road Condition Analysis"), model readiness badge.
- **Dashboard Tab:** Upload drop-zone, threshold slider, execution button, side-by-side comparison view, metric cards, and tabular results.
- **Model Performance Tab:** Tabulated quantitative metrics (mAP, Precision, Recall) and static graphical plots (loss curves, confusion matrix).
- **About Tab:** Academic context, RDD2022 dataset metadata, YOLOv8n architecture description, and disclaimer.

### 6.2 Hardware Interfaces
- Standard input peripherals (mouse, keyboard, touch screens).
- Optional CUDA-enabled GPU (NVIDIA Compute Capability 6.0+) accessed through standard PyTorch CUDA drivers.

### 6.3 Software Interfaces
- **Operating System:** Standard OS file I/O interfaces.
- **PyTorch / Torchvision:** Tensor operations and GPU memory management.
- **Ultralytics:** YOLO model deserialization and layer execution.
- **OpenCV / Pillow:** Image decoding, color space transformation, and graphics drawing primitives.

### 6.4 Dataset Interfaces
- **Format:** Pascal VOC XML format from the RDD2022 dataset.
- **Target Category:** Class `D40` (pothole).
- **Output Target:** Normalized YOLO text format (`class_id center_x center_y width height`).

---

## 7. System Workflow

The end-to-end processing lifecycle is depicted below:

```
+-----------------------------------------------------------+
| 1. User Uploads Image (JPG/PNG) via Streamlit Web App     |
+-----------------------------+-----------------------------+
                              |
+-----------------------------v-----------------------------+
| 2. File Validation (Format check, file size <= 15MB)      |
+-----------------------------+-----------------------------+
                              | Valid
+-----------------------------v-----------------------------+
| 3. Image Preprocessing (Letterbox resize 640x640, RGB)    |
+-----------------------------+-----------------------------+
                              |
+-----------------------------v-----------------------------+
| 4. CNN Model Inference (Ultralytics YOLOv8n forward pass) |
+-----------------------------+-----------------------------+
                              |
+-----------------------------v-----------------------------+
| 5. Post-Processing & Filtering (Apply Confidence Filter)   |
+-----------------------------+-----------------------------+
                              |
+-----------------------------v-----------------------------+
| 6. Bounding Box & Coordinate Calculation                  |
+-----------------------------+-----------------------------+
                              |
+-----------------------------v-----------------------------+
| 7. Result Visualization (Side-by-side render, tables)     |
+-----------------------------+-----------------------------+
                              |
+-----------------------------v-----------------------------+
| 8. Export Option (Download annotated image as PNG)        |
+-----------------------------------------------------------+
```

---

## 8. Use Case Descriptions

### UC-01: Upload Road Image
- **Primary Actor:** End User / Evaluator
- **Preconditions:** System web interface is loaded.
- **Main Flow:**
  1. User clicks the upload drop-zone or drags an image file.
  2. System validates file type and file size.
  3. System renders a preview of the raw uploaded image.
- **Alternate Flow:** If file is invalid, system displays an error banner and clears the preview.

### UC-02: Detect Potholes
- **Primary Actor:** End User
- **Preconditions:** Valid road image is uploaded and model weights are accessible.
- **Main Flow:**
  1. User sets desired confidence threshold (or leaves default 0.35).
  2. User clicks *"Analyse Road Image"*.
  3. System preprocesses image and runs YOLOv8n inference.
  4. System draws bounding boxes and renders detection metrics.
- **Alternate Flow:** If `models/best.pt` is missing, system displays an alert indicating weights are required.

### UC-03: Adjust Confidence Threshold
- **Primary Actor:** End User
- **Preconditions:** Image has been processed.
- **Main Flow:**
  1. User moves the confidence threshold slider.
  2. Detections are dynamically re-filtered against the new threshold.
  3. Summary metrics, tables, and bounding boxes update immediately.

### UC-04: Download Annotated Image
- **Primary Actor:** End User
- **Preconditions:** Pothole detection inference has completed.
- **Main Flow:**
  1. User clicks the *"Download Annotated Image"* button.
  2. Browser initiates a file download with the processed image.

### UC-05: View Model Performance
- **Primary Actor:** Evaluator / Student
- **Preconditions:** Evaluation metrics JSON/CSV file exists.
- **Main Flow:**
  1. User selects the "Model Performance" tab.
  2. System loads and displays Precision, Recall, F1, mAP@0.5, and mAP@0.5:0.95 alongside visual curves.
- **Alternate Flow:** If metrics are not generated yet, system displays: *"Model evaluation results are not available yet."*

---

## 9. Hardware and Software Requirements

### 9.1 Hardware Requirements
- **Minimum:**
  - CPU: Dual-Core Intel/AMD x86_64, 2.0 GHz or Apple Silicon M1.
  - RAM: 4 GB.
  - Disk Space: 2 GB free disk space (excluding raw datasets).
- **Recommended (for Model Training):**
  - CPU: Quad-Core Intel Core i5/i7 or AMD Ryzen 5+.
  - RAM: 16 GB.
  - GPU: NVIDIA GPU with at least 6 GB VRAM (CUDA 11.8+ or 12.0+).
  - Disk Space: 20 GB free SSD storage for RDD2022 dataset and checkpoints.

### 9.2 Software Requirements
- Operating System: Windows 10/11, Ubuntu 20.04+, or macOS Monterey+.
- Programming Language: Python 3.10, 3.11, or 3.12.
- Core Python Libraries:
  - `torch`, `torchvision`
  - `ultralytics`
  - `streamlit`
  - `opencv-python-headless` (or `opencv-python`)
  - `numpy`, `pandas`
  - `matplotlib`, `seaborn`
  - `pillow`, `pyyaml`

---

## 10. Project Limitations
1. **Academic Demonstration Scope:** Designed strictly as a research prototype; not certified for autonomous vehicular guidance or automated municipal enforcement.
2. **Atmospheric and Environmental Variations:** Detection accuracy may degrade under severe weather conditions (e.g., torrential rain, heavy snow, severe road glare, night scenes without artificial lighting).
3. **Single Class Target:** Specializes exclusively in potholes (RDD2022 category D40). Other road defects (such as longitudinal cracks, alligator cracks, or rutting) are not targeted in this prototype version.
4. **Static Image Focus:** Designed primarily for static single-frame photographs; streaming video pipelines and GPS telemetry integrations are designated for future phases.

---

## 11. Acceptance Criteria
- **AC-01:** System successfully launches via `streamlit run app.py` without unhandled dependency errors.
- **AC-02:** System accepts JPG, JPEG, and PNG images and accurately rejects non-image formats.
- **AC-03:** Pothole detections generated by YOLOv8n show valid coordinates mapped accurately to the original visual dimensions.
- **AC-04:** Confidence slider dynamically controls visible bounding boxes without requiring page reloads.
- **AC-05:** Annotated images can be saved locally in full resolution.
- **AC-06:** Graceful fallback messaging occurs when weights or evaluation files are missing.
- **AC-07:** The visual theme strictly adheres to the specified light-themed civil engineering color scheme.

---

## 12. Conclusion
This Software Requirements Specification establishes a rigorous, realistic, and implementable foundation for the RoadWatch AI pothole detection prototype. By combining modern CNN architectures (YOLOv8n) with a clean, domain-appropriate presentation layer, the application fulfills academic demonstration goals while adhering to industry software engineering principles.

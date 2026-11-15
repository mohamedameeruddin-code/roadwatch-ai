# Product Requirements Document (PRD)
## RoadWatch AI – CNN-Based Road Pothole Detection System

---

### Document Information
- **Document Title:** Product Requirements Document (PRD)
- **Product Name:** RoadWatch AI
- **Version:** 1.0.0
- **Document Status:** Complete & Approved
- **Reference Document:** [01_SRS.md](01_SRS.md)

---

## 1. Product Overview

### 1.1 Product Name
**RoadWatch AI** (Subtitle: *Intelligent Road Condition Analysis*)

### 1.2 Product Description
RoadWatch AI is a specialized computer vision application built to automate the identification and spatial localization of potholes in road photographs. Operating as an academic deep learning prototype, the application allows users to upload standard roadway images, processes them using a fine-tuned Convolutional Neural Network (YOLOv8n), and presents intuitive visual overlays featuring bounding boxes, detection counts, and confidence scores.

### 1.3 Product Vision
To provide civil engineers, researchers, and academic evaluators with an accessible, transparent, and accurate artificial intelligence tool that demonstrates the practical application of modern deep learning to civil infrastructure monitoring and road condition assessment.

### 1.4 Problem Statement
Road surface degradation—predominantly potholes caused by water seepage, heavy traffic loads, and thermal expansion—poses significant hazards to vehicle suspension, road safety, and municipal budgets. Existing inspection mechanisms rely heavily on manual surveys, which are time-consuming, expensive, hazardous for field personnel, and prone to subjective human error.

### 1.5 Proposed Solution
RoadWatch AI provides an automated, objective, image-based defect inspection system. By leveraging transfer learning on the lightweight YOLOv8n CNN architecture trained on the Road Damage Dataset 2022 (RDD2022), the platform delivers near-instantaneous pothole detection with customizable sensitivity controls and full performance metric transparency.

---

## 2. Problem Analysis

### 2.1 Problems Caused by Road Potholes
- **Vehicle Wear and Damage:** Potholes cause frequent wheel rim dents, tire blowouts, suspension misalignment, and early shock absorber failure.
- **Traffic Safety Risks:** Drivers abruptly swerving to avoid potholes frequently cause sideswipe or head-on collisions.
- **Economic Inefficiency:** Delayed detection of micro-defects permits water ingress, turning minor cracks into severe structural failures that cost tenfold more to repair.

### 2.2 Challenges in Manual Pothole Inspection
- **Labor Intensive:** Field surveyors must drive or walk road networks, manually logging coordinates and defect descriptions.
- **High Risk:** Surveyors working directly on active arterial roadways or highways face severe pedestrian hazard risks.
- **Inconsistency:** Disparate inspection crews evaluate severity subjectively, leading to unstandardized repair prioritization.
- **Delayed Processing:** Traditional paper or spreadsheet logs take days or weeks to be aggregated into maintenance work orders.

### 2.3 Need for Automated Pothole Detection
Automated vision systems eliminate human subjectivity, accelerate the scanning of hundreds of road photographs per minute, and generate standardized, digital bounding boxes that can be archived, analyzed, and mapped.

### 2.4 Benefits of Computer Vision
- **Consistency:** Algorithmic inference applies identical mathematical criteria across all input images.
- **Speed:** Deep neural networks process individual high-resolution frames in milliseconds.
- **Non-Invasive:** Images captured via dashcams, handheld phones, or drones can be processed without disrupting roadway traffic.

---

## 3. Product Objectives

1. **Automate Pothole Identification:** Accurately localize pothole instances within digital road photographs using deep learning.
2. **Demonstrate CNN-Based Object Detection:** Showcase real-world transfer learning utilizing a state-of-the-art CNN architecture (Ultralytics YOLOv8n).
3. **Provide Visual Detection Results:** Render high-clarity bounding boxes, percentage confidence ratings, and side-by-side original/annotated comparisons.
4. **Evaluate the Trained Model:** Empirically quantify and report test performance using standard computer vision metrics (Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95).
5. **Create an Easy-to-Use Demonstration Interface:** Deliver a clean, professional, light-themed web interface tailored for academic demonstrations and technical reviews.

---

## 4. Target Users

| User Persona | Role & Context | Primary Need |
| :--- | :--- | :--- |
| **Students & Researchers** | Deep learning and civil engineering students exploring applied vision models. | Understand model performance, test various image scenarios, and observe threshold effects. |
| **Academic Project Evaluators** | Professors, teaching assistants, and review panels assessing project quality. | Validate code architecture, verify real CNN execution, and inspect empirical evaluation results. |
| **Road Inspection Personnel** | Municipal civil technicians or quality inspectors testing automated workflows. | Quick, clear visual validation of road damage with downloadable annotated reports. |
| **General Observers** | Tech enthusiasts and road safety advocates. | Simple, no-code web demonstration of AI applied to everyday infrastructure problems. |

---

## 5. Core Product Features

### FEATURE 1: Image Upload
- Drag-and-drop file ingestion zone supporting standard formats (`.jpg`, `.jpeg`, `.png`).
- File size boundary checks (maximum 15 MB) and corruption rejection.
- Immediate raw image preview.

### FEATURE 2: AI Pothole Detection
- Automated letterbox resizing to 640x640 resolution.
- Forward pass inference utilizing fine-tuned YOLOv8n weights (`models/best.pt`).
- Non-Maximum Suppression (NMS) and spatial coordinate mapping back to original image dimensions.

### FEATURE 3: Visual Results
- Clean visual comparison featuring side-by-side columns:
  - **Left:** Original input image.
  - **Right:** Annotated output with green-accented bounding boxes and confidence tags.

### FEATURE 4: Detection Summary
- Prominent metrics display:
  - Total Potholes Detected
  - Highest Confidence Score
  - Average Confidence Score
  - Inference Execution Latency (ms)
- Structured tabular breakdown of detected coordinates and individual probabilities.

### FEATURE 5: Confidence Adjustment
- Interactive slider allowing users to adjust the detection confidence threshold (from 0.05 to 1.00 in increments of 0.05).
- Instant re-filtering of detections to observe precision/recall trade-offs without re-uploading.

### FEATURE 6: Model Evaluation
- Dedicated performance dashboard displaying empirical metrics calculated on the held-out test split:
  - Precision, Recall, F1-score
  - mAP@0.5 and mAP@0.5:0.95
  - Graphical plots: Confusion matrix, Precision-Recall curve, and training/validation loss.
- Honest state management: If evaluation has not been executed, displays *"Model evaluation results are not available yet."*

### FEATURE 7: Download Results
- Export processed image containing visual bounding boxes directly to the user's local disk in PNG format.

---

## 6. User Journey

```
+-----------------------------------------------------------------+
| 1. Discovery & Launch                                           |
|    User opens the Streamlit application via browser.            |
|    Observes clean civil engineering light theme and model status|
+-------------------------------+---------------------------------+
                                |
+-------------------------------v---------------------------------+
| 2. Image Selection & Ingestion                                  |
|    User drags a road photo into the upload drop-zone.           |
|    System validates file format and displays immediate preview. |
+-------------------------------+---------------------------------+
                                |
+-------------------------------v---------------------------------+
| 3. Parameter Configuration & Analysis Trigger                   |
|    User adjusts confidence slider (e.g. 0.35 default).          |
|    User clicks "Analyse Road Image".                            |
+-------------------------------+---------------------------------+
                                |
+-------------------------------v---------------------------------+
| 4. Inspection & Evaluation of Detections                        |
|    User views side-by-side comparison and summary metric cards. |
|    User reviews bounding box coordinates in the details table.  |
+-------------------------------+---------------------------------+
                                |
+-------------------------------v---------------------------------+
| 5. Export or In-Depth Evaluation                                |
|    User downloads annotated image for reporting.                |
|    User switches to "Model Performance" tab to inspect mAP/loss.|
+-----------------------------------------------------------------+
```

---

## 7. User Stories

### US-01: Image Upload and Validation
- **As a user,** I want to upload a road image in JPG or PNG format,  
- **So that** the application can inspect it for surface defects.  
- **Acceptance Criteria:**
  - Files matching `.jpg`, `.jpeg`, `.png` are accepted.
  - Invalid formats trigger a clear warning: *"Unsupported format. Please upload a JPG or PNG image."*
  - An image preview is immediately rendered upon successful upload.

### US-02: Automated Pothole Detection
- **As a user,** I want the application to analyze my uploaded road image using a trained CNN,  
- **So that** I do not have to manually locate potholes.  
- **Acceptance Criteria:**
  - Clicking *"Analyse Road Image"* triggers YOLOv8n inference.
  - Detected potholes are demarcated with bounding boxes and percentage labels.
  - If no defects meet the threshold, system shows *"No potholes detected at the selected confidence threshold."*

### US-03: Dynamic Sensitivity Control
- **As a user,** I want to adjust the confidence threshold slider,  
- **So that** I can observe high-confidence detections versus borderline candidates.  
- **Acceptance Criteria:**
  - Slider ranges from 0.05 to 1.00 with 0.05 increments.
  - Detections and summary metrics update reactively based on the slider value.

### US-04: Exporting Annotated Evidence
- **As an inspector,** I want to download the annotated image with bounding boxes,  
- **So that** I can include it in maintenance audit reports.  
- **Acceptance Criteria:**
  - Clicking *"Download Annotated Image"* downloads the full-resolution processed image file.

### US-05: Reviewing Empirical Model Performance
- **As an academic evaluator,** I want to view verified test metrics (mAP, Precision, Recall) on a dedicated tab,  
- **So that** I can assess the scientific validity of the trained model.  
- **Acceptance Criteria:**
  - Evaluation tab displays actual test metrics loaded from `outputs/metrics/`.
  - If metrics do not exist, the UI displays *"Model evaluation results are not available yet."* without fabricating numbers.

---

## 8. Feature Prioritisation

| Priority | Feature ID | Feature Name | Rationale |
| :--- | :--- | :--- | :--- |
| **P0 (Must Have)** | F1, F2, F3, F5 | Core Detection Pipeline | The project's essential MVP purpose is uploading road photos and visualizing CNN pothole detections with adjustable sensitivity. |
| **P0 (Must Have)** | F4 (Basic) | Detection Summary | Displaying count and processing status is essential to verify model activity. |
| **P1 (Should Have)**| F7 | Export Annotated Image | Allows evaluators to save results locally for documentation. |
| **P1 (Should Have)**| F6 | Model Evaluation Tab | Critical for an academic submission to prove model training and validation. |
| **P2 (Could Have)** | F4 (Advanced)| Coordinate Tables & Latency | Tabulated bounding box coordinates and precise execution timing. |

---

## 9. Success Metrics

1. **Successful Image Processing:** 100% of valid JPG/PNG images within 15 MB boundary execute inference without software crashes.
2. **Detection Result Generation:** Detected bounding boxes accurately reflect model tensor coordinates mapped back to original image aspect ratio.
3. **Inference Latency:** Average single-frame inference time under 2.5 seconds on modern CPU hardware.
4. **Model Evaluation Transparency:** 100% of reported metrics (Precision, Recall, mAP) derive directly from held-out test splits without synthetic fabrication.
5. **Usability & Aesthetic Compliance:** 100% adherence to the defined civil engineering Light Theme (#244B3C forest green, #F6F8F5 background, #FFFFFF cards) with zero dark-theme AI gradient artifacts.

*(Note: In accordance with academic integrity guidelines, numerical model accuracy targets such as >95% mAP are not artificially predefined.)*

---

## 10. Product Scope

### 10.1 Included Features
- Single-image upload, validation, and preview.
- YOLOv8n deep learning inference on road imagery.
- Pothole bounding box overlays with class label and confidence score.
- Adjustable confidence threshold control.
- Summary metrics and bounding box coordinate table.
- Downloadable annotated images.
- Empirical test evaluation presentation (mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, Loss plots).
- Clean, responsive Streamlit light-themed UI.

### 10.2 Excluded Features
- Real-time video stream ingestion (e.g. RTSP or live webcam feeds).
- GPS-based spatial road mapping or GIS dashboard integration.
- Multi-user authentication, roles, and profiles.
- Paid cloud subscriptions or external API billing.
- Broad-spectrum road defect classification beyond RDD2022 category D40 (potholes).
- Guarantee of road structural safety.

---

## 11. Risks and Limitations

| Risk Factor | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Adverse Weather / Poor Lighting** | Shadows, water reflections, or night imagery may trigger false positives or missed detections. | Document boundary limitations clearly in the About section and PRD. |
| **Missing Model Checkpoint** | Application would fail if `models/best.pt` is not found. | Graceful fallback banner informing user to run `train.py` or supply weights. |
| **Hardware Constraints on CPU** | Large batches or slow CPU inference could cause UI latency. | Single-image inference optimization with letterbox caching; clear progress spinners. |
| **Annotation Inaccuracies in Public Data** | Public datasets contain noise and varying labeling boundaries. | Standardize VOC parsing, filter strictly on D40, and preserve negative samples. |

---

## 12. Future Enhancements

- **Phase 2:** Continuous video stream processing for vehicle dashcam footage with frame-skipping optimizations.
- **Phase 3:** Integration of GPS telemetry metadata to automatically map potholes onto municipal GIS systems (e.g. OpenStreetMap).
- **Phase 4:** Expansion to multi-defect classification (longitudinal cracks, alligator cracking, road rutting).
- **Phase 5:** Edge deployment on embedded hardware (NVIDIA Jetson, Raspberry Pi with Coral TPU).

---

## 13. Product Release Phases

1. **Phase 1: Dataset Preparation:** Download RDD2022 India subset, parse VOC XML annotations, isolate D40 pothole class, convert to YOLO format, and generate 70/15/15 train/val/test splits.
2. **Phase 2: Model Development:** Fine-tune YOLOv8n utilizing transfer learning with data augmentations, saving the best checkpoint to `models/best.pt`.
3. **Phase 3: Model Evaluation:** Execute evaluation against the held-out test split, exporting verified metrics and diagnostic figures into `outputs/metrics/` and `outputs/plots/`.
4. **Phase 4: UI Development:** Build the Streamlit interface using custom CSS adhering strictly to the forest green / off-white palette.
5. **Phase 5: Integration & Testing:** Wire `PotholeDetector` into Streamlit, conduct end-to-end integration tests, and verify graceful edge-case handling.

---

## 14. Conclusion
The RoadWatch AI Product Requirements Document defines a balanced, technically viable, and academically sound product roadmap. By strictly constraining scope to single-image pothole detection while adhering to professional UI principles and transparent evaluation, RoadWatch AI delivers a high-impact prototype suitable for academic evaluation and research demonstration.

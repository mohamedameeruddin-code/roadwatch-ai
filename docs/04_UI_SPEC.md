# RoadWatch AI – Premium UI/UX Design Specification

---

### Document Control
- **Document Title:** Premium UI/UX Design Specification (TRD-UI)
- **Product Name:** RoadWatch AI
- **Tagline:** Intelligent Road Condition Analysis
- **Purpose:** CNN-Based Road Pothole Detection System
- **Document Version:** 2.0.0
- **Status:** Approved for Implementation
- **Reference Documents:** [01_SRS.md](01_SRS.md), [02_PRD.md](02_PRD.md), [03_TRD.md](03_TRD.md)

---

## 1. Design Vision & Philosophy

RoadWatch AI is envisioned as a premium, visually rich, sophisticated civil infrastructure monitoring dashboard. The visual identity bridges cutting-edge computer vision research with civil and highway engineering standards.

### Key Visual Tenets:
1. **Light-Themed Elegance with Contrast:** The interface deliberately avoids both stark, clinical pure-white backgrounds and dark "gamer/hacker" AI styling. Instead, it utilizes subtle sage greens, warm ivory panels, and deep evergreen anchors to create depth, comfort, and professional gravitas.
2. **Infrastructure Visual Identity:** Design accents incorporate subtle road geometry, asphalt textures, and civil survey styling (e.g., coordinate brackets, station metrics, road striping accents).
3. **Structured Information Architecture:** High data-density views balanced by generous whitespace, clear card groupings, and unambiguous typography.

---

## 2. Strict Colour Palette & Design System

The application strictly enforces the following unified colour system across custom CSS, components, and Matplotlib chart styling.

### 2.1 Primary Palette
- **Forest Green (`#244B3C`):** Primary action buttons, active tabs, metric badges, and brand highlights.
- **Deep Evergreen (`#19382E`):** Navigation sidebar background, header mastheads, and terminal card borders.
- **Sage Green (`#789582`):** Secondary interactive elements, sliders, active navigation indicators, and graphic strokes.
- **Soft Sage (`#E3EDE4`):** Hero gradient start, subtle card highlights, and status container fills.

### 2.2 Background & Surface Palette
- **Main Background (`#F1F4EF`):** Primary canvas background for the entire application workspace.
- **Secondary Background (`#E8EEE7`):** Grouping containers, nested control wells, and input frames.
- **Warm Ivory (`#FAF9F4`):** Secondary cards, alternate table rows, and hero gradient endpoint.
- **White Surface (`#FFFFFF`):** High-priority cards, image display viewports, and primary analytic containers.

### 2.3 Accent & Neutral Palette
- **Muted Gold (`#C69B5B`):** Warning indicators, high-severity detection badges, and secondary chart lines.
- **Soft Amber (`#EBD9B9`):** Accent pill fills, threshold notifications, and subtle tag borders.
- **Road Asphalt (`#414A45`):** Coordinate data text, technical metadata tags, and bounding box outlines.

### 2.4 Typography Colours
- **Primary Text (`#26332B`):** Main headings, prominent metrics, card titles (high contrast, WCAG AAA compliant).
- **Secondary Text (`#69766E`):** Body paragraphs, field labels, metadata descriptors.
- **Muted Text (`#929C94`):** Captions, timestamps, placeholder hints, disclaimers.

### 2.5 Borders & Dividers
- **Default Border (`#D7E0D7`):** Standard card borders, container outlines (1px solid).
- **Subtle Border (`#E7EBE5`):** Inner grid lines, table row dividers, and control separators.

### 2.6 Functional Status Colours
- **Success (`#3D7957`):** Normal operation, zero hazard, model validation pass.
- **Warning (`#BB812F`):** Pothole hazard detected, low confidence alert.
- **Error (`#B6534C`):** Invalid file format, inference exception, missing model file.

---

## 3. Typography Hierarchy

The primary font family is **Inter**, with fallbacks to system sans-serif fonts:
`font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;`

| Element | Size | Weight | Line Height | Color | Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hero Title** | 32–34px | 700 (Bold) | 1.2 | `#19382E` | Landing hero header |
| **Section Heading** | 22–24px | 600 (SemiBold) | 1.3 | `#244B3C` | Primary section titles |
| **Card Heading** | 16–17px | 600 (SemiBold) | 1.4 | `#26332B` | Module and card titles |
| **Metric Display** | 26–30px | 700 (Bold) | 1.1 | `#19382E` | KPI card numerical numbers |
| **Body Text** | 14–15px | 400 (Regular) | 1.5 | `#26332B` | General copy and descriptions |
| **Supporting / Meta** | 12–13px | 500 (Medium) | 1.4 | `#69766E` | Badges, coordinates, table headers |
| **Micro Labels** | 11–12px | 500 (Medium) | 1.3 | `#929C94` | Footers, disclaimers, status hints |

---

## 4. Application Architecture & Layout

### 4.1 Left Navigation Sidebar
- **Background:** Solid Deep Evergreen (`#19382E`).
- **Brand Masthead:**
  - Application Icon: Road inspection / scanner glyph.
  - Title: **RoadWatch AI** (White, 20px, Bold).
  - Subtitle: *Intelligent Road Condition Analysis* (Soft Sage `#E3EDE4`, 11px).
  - Architecture Tag: `YOLOv8n • RDD2022 India` (Badge with border `#789582`).
- **Navigation Menu:**
  - 1. **🔍 Pothole Detection** (Default active view)
  - 2. **📊 Model Performance** (Evaluation metrics & loss curves)
  - 3. **ℹ️ About Project** (Architecture & academic specifications)
- **Navigation Active State:**
  - Left border indicator: 4px solid `#789582` (Sage Green).
  - Background fill: `rgba(120, 149, 130, 0.18)`.
  - Text color: `#FFFFFF`.
- **Sidebar Footer:**
  - Hardware status: CPU/GPU device indicator.
  - Model status badge: `best.pt [Loaded]`.
  - Version: `v1.0.0 (Academic Build)`.

### 4.2 Main Application Area
- **Canvas Background:** Soft Sage-Grey (`#F1F4EF`).
- **Max Content Width:** Centered container with max-width `1280px` and adaptive margins.

---

## 5. View Specifications

### 5.1 View 1: Pothole Detection (Main Workspace)

#### A. Hero Masthead
- **Background:** Linear gradient `linear-gradient(135deg, #E3EDE4 0%, #FAF9F4 100%)`.
- **Border:** 1px solid `#D7E0D7`, border-radius: `14px`.
- **Content:**
  - Title: **Road Intelligence Dashboard**
  - Subtitle: *Advanced computer vision for identifying and analysing road surface damage.*
  - Road Inspection Graphic: Subtle civil road coordinate badge and status indicators (`RDD2022 India Dataset • Single-Class Pothole Model D40`).

#### B. Upload & Configuration Container
- **Two-Column Split Layout:**
  - **Left Sub-Panel: Image Ingestion**
    - Drag & drop file uploader with dashed border: `2px dashed #789582`.
    - Allowed formats: `JPG, JPEG, PNG` (Max size: 20MB).
    - Image preview box showing dimensions (`W x H px`), filename, and file size.
    - Quick sample selector button: Allows instant testing with bundled real test road images.
  - **Right Sub-Panel: Detection Parameters**
    - Confidence Threshold Slider:
      - Default: `0.35` (35%).
      - Range: `0.05` to `0.95`, step `0.05`.
      - Color accent: `#244B3C`.
    - Primary Action Button:
      - Label: **ANALYSE ROAD IMAGE**
      - Styling: Background `#244B3C`, text `#FFFFFF`, font-weight 600, border-radius 8px.
      - Hover state: Background `#19382E`, shadow `0 4px 12px rgba(36, 75, 60, 0.25)`.

#### C. Detection Results & Visual Comparison
- **Visual Presentation:** Two equal-width viewport cards.
  - **Left Card: Original Road Image** (Clean input preview).
  - **Right Card: AI Detection Result**
    - Overlaid with exact bounding boxes generated by YOLOv8n.
    - Box color: Forest Green / Asphalt outline with high-contrast text tags (`pothole: XX.X%`).
- **Real-Time KPI Metric Cards (Grid of 4):**
  1. **Total Potholes Detected:** Numerical count (e.g., `2`).
  2. **Highest Confidence:** Max confidence score (e.g., `87.4%`).
  3. **Average Confidence:** Mean confidence score (e.g., `79.1%`).
  4. **Inference Latency:** Processing time in milliseconds (e.g., `18.2 ms`).
- **Detection Table:**
  - Columns: `#`, `Class Code`, `Class Name`, `Confidence`, `Bounding Box (X1, Y1, X2, Y2)`.
  - Clean table styled with `#FAF9F4` zebra striping and `#26332B` text.
- **Export Action:**
  - **Download Annotated Image** button (Forest Green secondary style, exports high-res JPEG).
- **Empty State (Zero Potholes):**
  - Elegant ivory banner (`#FAF9F4`) with sage border:
  - Text: *"No potholes detected at the selected confidence threshold."*
  - Contextual note: *"Adjust the confidence threshold slider or upload another image."*

---

### 5.2 View 2: Model Performance & Analytics

#### A. Performance KPI Summary
Top metrics strip displaying verified empirical test evaluation:
- **Precision:** Model precision at IoU=0.5.
- **Recall:** Model recall rate.
- **F1-Score:** Harmonic mean of precision and recall.
- **mAP@0.5:** Mean Average Precision at IoU threshold 0.5.
- **mAP@0.5:0.95:** COCO-standard mAP across 0.50 to 0.95.

#### B. Diagnostic Visualizations (2x2 Grid)
1. **Confusion Matrix:** Heatmap showing true positive pothole detections vs background negatives.
2. **Precision-Recall Curve:** Precision trajectory across varying recall points with calculated AUC.
3. **Training & Validation Loss:** Box loss, Class loss, and DFL loss curves across epochs.
4. **F1-Confidence Curve:** Optimal confidence operating threshold graph.
- *Styling Rule:* Charts render with Sage Green (`#789582`), Forest Green (`#244B3C`), and Muted Gold (`#C69B5B`) palettes against white/ivory card backgrounds.
- If evaluation metrics are pending, displays clear notification: *"Model evaluation results are not available yet."*

---

### 5.3 View 3: About Project & Architecture

#### A. Academic Overview
- Project Title: **RoadWatch AI – CNN-Based Road Pothole Detection System**
- Course / Domain: Academic Deep Learning Prototype & Civil Infrastructure Monitoring.
- Objectives, background challenges, and automated vision advantages.

#### B. Technical Architecture
- **Dataset:** Road Damage Dataset 2022 (RDD2022) India Subset (7,706 images, 70:15:15 split).
- **Target Category:** Class `D40` (Pothole).
- **Model:** Ultralytics YOLOv8n (nano CNN architecture, 3.2M parameters).
- **Runtime Stack:** Python 3.11, PyTorch, Torchvision, OpenCV, Streamlit.

#### C. Formal Academic Prototype Disclaimer
- Framed in an elegant ivory callout box with a subtle gold accent:
  > **Academic Prototype Disclaimer:** RoadWatch AI is engineered exclusively for academic demonstration and research benchmarking. It does not certify structural highway safety or replace certified civil engineering roadway compliance inspections.

---

## 6. Micro-Interactions, Feedback & Responsive Behavior

1. **Button Hover:** Smooth 0.2s cubic-bezier transition with subtle elevation shadow (`0 4px 10px rgba(0,0,0,0.08)`).
2. **Card Hover:** Micro-lift (1px Y-offset) with border highlight transition.
3. **Inference Spinner:** Styled Streamlit loading spinner: *"Executing YOLOv8n CNN inference..."*.
4. **Responsive Grid:**
   - Desktop (>1024px): Two-column comparison view, 4-column metric grid.
   - Tablet (768px–1024px): Stacked comparison view, 2-column metric grid.
   - Mobile (<768px): Single-column stacked layout, full-width buttons.

---

## 7. Configuration & File Structure

- **Streamlit Theme:** `.streamlit/config.toml`
  - `primaryColor = "#244B3C"`
  - `backgroundColor = "#F1F4EF"`
  - `secondaryBackgroundColor = "#E8EEE7"`
  - `textColor = "#26332B"`
  - `font = "sans serif"`
- **Custom CSS Engine:** Embedded inside `app.py` or `src/styles.css` applying all color codes, card radii (12-14px), custom buttons, and metric typography.

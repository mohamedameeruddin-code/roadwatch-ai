# RoadWatch AI – UI/UX Design Specification

---

### Document Control
- **Document Title:** Infrastructure Monitoring Dashboard Specification (TRD-UI)
- **Product Name:** RoadWatch AI
- **Tagline:** Autonomous Pothole Detection and Pavement Condition Assessment
- **Document Version:** 3.0.0 (Medium-Dark Visual Theme)
- **Status:** Approved for Implementation
- **Reference Documents:** [01_SRS.md](01_SRS.md), [02_PRD.md](02_PRD.md), [03_TRD.md](03_TRD.md)

---

## 1. Executive Summary & Design Direction

RoadWatch AI is an intelligent civil infrastructure monitoring platform that bridges academic computer vision research with production-grade engineering software.

The version 3.0.0 redesign transitions the interface to a **Medium-Dark visual theme**. The aesthetic avoids harsh pure-black OLED tones and generic "hacker/cyberpunk" neon palettes, drawing inspiration from modern engineering suites, SCADA control consoles, and enterprise infrastructure monitoring tools.

### Key Design Pillars:
1. **Medium-Dark Surfaces with Depth:** Layered charcoal-green surfaces (`#26332E`, `#1C2924`, `#34423B`, `#3D4C44`) that reduce eye fatigue during prolonged road inspection sessions.
2. **Strict Zero-Emoji Policy:** Unicode emojis are completely eliminated across all UI surfaces (navigation, headers, metrics, notifications, empty states). Interface cues rely exclusively on clean, monochromatic Lucide SVG icons.
3. **Information Density & Hierarchy:** Distinct visual separation between global KPIs, inspection workspace controls, side-by-side verification viewports, and localization telemetry tables.
4. **Accessible Contrast:** Primary typography (`#F0F3EF`) on charcoal-green surfaces meets WCAG AAA standards (> 7:1 contrast ratio).

---

## 2. Strict Colour Palette & Design System

The application strictly enforces the following unified colour system across custom CSS, Streamlit components, Matplotlib plots, and the mobile progressive web client.

### 2.1 Core Palette Tokens

| Token Name | Hex Code | Visual Role | Contrast / Usage |
| :--- | :--- | :--- | :--- |
| **MAIN BACKGROUND** | `#26332E` | Canvas background | Primary viewport canvas |
| **SIDEBAR BACKGROUND** | `#1C2924` | Sidebar drawer | Deep charcoal-green anchor |
| **PRIMARY CARD BACKGROUND** | `#34423B` | Content cards & panels | Mid charcoal-green surface |
| **ELEVATED CARD BACKGROUND** | `#3D4C44` | Elevated wells & controls | High-focus input zones |
| **PRIMARY GREEN** | `#78977F` | Interactive accents & buttons | Primary buttons, active indicators |
| **SAGE ACCENT** | `#A0B7A3` | Secondary accents & icons | Selected states, secondary icons |
| **LIGHT GREEN** | `#C0D0C1` | Highlight badges | Subtle value callouts |
| **MUTED GOLD** | `#C2A36D` | High-certainty metric badge | Peak confidence indicators |
| **PRIMARY TEXT** | `#F0F3EF` | Headings & primary labels | Ultra-high contrast readable text |
| **SECONDARY TEXT** | `#B5C0B7` | Body copy & descriptions | Crisp secondary readable text |
| **MUTED TEXT** | `#8F9B92` | Captions & inactive labels | De-emphasized metadata |
| **BORDER** | `#4B5A51` | Dividers & bounding boxes | Subtle delineation lines |
| **SUCCESS** | `#78A987` | Normal state & ready dots | Validation pass indicators |
| **WARNING** | `#D2A35D` | Detection alerts & empty states | Sensitivity warnings |
| **ERROR** | `#C7756D` | Validation failure & errors | File validation exceptions |

---

## 3. Iconography System: Lucide Icons & Zero Emojis

### 3.1 Strict Prohibition of Unicode Emojis
Unicode emojis (such as 🛣️, 🔍, 📊, ℹ️, ⚙️, 📤, 🎯, ⚡, ⚠️) are strictly disallowed across all user-facing elements:
- Sidebar navigation labels
- Hero headers and subheadings
- Action buttons and file upload triggers
- Metric card labels and value badges
- Empty state and error notification alerts
- Table headers and column names

### 3.2 Standardized Lucide SVG Icon Catalog

All icons are rendered as vector SVGs (`24x24` viewBox) with a consistent `2.0px` stroke weight and monochromatic fills:

| Icon Name | Context / Placement | Default Colour |
| :--- | :--- | :--- |
| `LayoutDashboard` | Main Dashboard navigation and hero | `#A0B7A3` |
| `ScanSearch` | Pothole Detection workspace & hero | `#78977F` |
| `ChartNoAxesCombined` | Model Performance navigation & analytics | `#A0B7A3` |
| `Upload` | Ingestion card header & drop zone | `#78977F` |
| `Image` | Original road surface image viewport | `#8F9B92` |
| `Activity` | System health, precision metrics, active state | `#78A987` |
| `CircleCheck` | Validation standards, high confidence badge | `#78A987` |
| `AlertTriangle` | Disclaimers, sensitivity empty state alert | `#D2A35D` |
| `Download` | Annotated high-resolution export button | `#F0F3EF` |
| `Settings` | Detection parameters and slider panel | `#A0B7A3` |
| `Info` | About Project navigation & methodology | `#C0D0C1` |
| `FileImage` | Localization coordinates table header | `#A0B7A3` |
| `Gauge` | Hardware inference latency indicator | `#8F9B92` |
| `Clock` | Execution time & inference telemetry | `#8F9B92` |
| `Database` | RDD2022 India dataset breakdown | `#8F9B92` |
| `Road` | Brand logo mark and road surface empty state | `#78977F` |

---

## 4. Layout Architecture & Screen Specifications

### 4.1 Global Shell & Sidebar
- **Width:** Fixed 290px expanded, collapsible on mobile.
- **Background:** `#1C2924` (Charcoal-Green).
- **Brand Header:** RoadWatch AI typography with a glowing status dot (`#78A987`) indicating active model initialization (`YOLOv8n Active | CUDA GPU`).
- **Navigation Menu:** Clear vertical text options:
  1. `Dashboard`
  2. `Pothole Detection`
  3. `Model Performance`
  4. `About Project`
- **Active Navigation Indicator:** Background `#34423B`, text `#A0B7A3`, left border 3px solid `#78977F`.
- **Sidebar Footer:** Embedded compute specifications (`CUDA GPU / CPU`), dataset citation (`RDD2022 India`), and version stamp.

---

### 4.2 Module 1: Main Dashboard
- **Hero Banner:** Full-width panel (`#34423B` with border `#4B5A51`) featuring title *"Road Intelligence Dashboard"*, subtitle *"Autonomous Pothole Detection and Pavement Condition Assessment"*, and `LayoutDashboard` icon.
- **Top Metric Cards Grid (4 columns):**
  1. **Model Precision:** `22.7% mAP@0.5` (Held-out RDD2022 test split)
  2. **Supported Class:** `D40 Pothole` (Single-class CNN object detection)
  3. **Inference Latency:** `7.2 ms` (Tensor Core single-frame execution)
  4. **Dataset Volume:** `7,706 Images` (RDD2022 India 70:15:15 split)
- **Module Shortcuts (3 columns):** Direct action cards for *Pothole Detection*, *Model Performance*, and *Project Overview*, featuring descriptions and direct one-click routing buttons.
- **System Health Table:** Real-time summary detailing model readiness, precision target, hardware acceleration, and PWA client status.

---

### 4.3 Module 2: Pothole Detection Workspace
- **Hero Banner:** *"Pothole Detection Workspace"* with `ScanSearch` icon.
- **Input & Parameter Zone (2 Columns):**
  - **Left (3/5 Width) - Image Ingestion (`#3D4C44`):**
    - File uploader accepting `.jpg`, `.jpeg`, `.png` up to 25 MB.
    - Quick benchmark selector buttons for pre-loaded RDD2022 test samples (`Road #1`, `Road #2`, `Road #3`).
  - **Right (2/5 Width) - Detection Parameters (`#3D4C44`):**
    - Calibrated confidence slider (range `0.01` to `0.95`, step `0.01`, default `0.15`).
    - Real-time sensitivity badge displaying percentage sensitivity.
    - Prominent `ANALYSE ROAD IMAGE` button (`#78977F`).
- **Telemetry Chip:** Compact banner displaying image filename, dimensions (`W x H px`), file size (`KB`), and model architecture.
- **Side-by-Side Verification Viewport:**
  - **Left Card:** *"Original Road Surface"* with responsive viewport framing.
  - **Right Card:** *"AI Detection Result (CNN)"* displaying annotated image with `#78977F` bounding boxes, `#34423B` label pills, and `#F0F3EF` confidence text.
- **Detection Summary Grid (4 Cards):**
  - *Total Potholes Detected*
  - *Highest Confidence*
  - *Average Confidence*
  - *Inference Latency*
- **Localization Data Table:** Clean dark table with columns: `Detection #`, `Class`, `Confidence`, `Bounding Box [xmin, ymin, xmax, ymax]`, `Box Width`, `Box Height`.
- **Export Control:** High-resolution PNG download button with `Download` Lucide icon.
- **Defect-Free Empty State:** Informative card with `AlertTriangle` icon explaining that no candidates exceeded the threshold, suggesting a lower slider value.

---

### 4.4 Module 3: Model Performance & Analytics
- **Hero Banner:** *"Model Performance & Analytics"* with `ChartNoAxesCombined` icon.
- **Empirical Metric Cards:**
  - Precision: `28.3%`
  - Recall: `29.2%`
  - F1-Score: `28.8%`
  - mAP@0.5: `22.7%`
- **Quantitative Benchmark Panels:**
  - Table 1: Detailed metrics including `mAP@0.5:0.95 (10.9%)`, preprocess (`1.8 ms`), inference (`7.2 ms`), and postprocess (`1.1 ms`).
  - Table 2: RDD2022 India partitioning breakdown (Train: 5,394, Val: 1,155, Test: 1,157).
- **Diagnostic Visualizations Grid (2x2):**
  - Confusion Matrix (`confusion_matrix.png`)
  - Training Loss Trajectory (`results.png`)
  - Precision-Recall Curve (`PR_curve.png`)
  - F1-Confidence Curve (`F1_curve.png`)

---

### 4.5 Module 4: About Project & Methodology
- **Hero Banner:** *"About RoadWatch AI"* with `Info` icon.
- **Left Column:**
  - Project Objectives & Rationale (Safety hazards, survey costs, computer vision automation).
  - Technical Pipeline & Methodology (6-step pipeline from Pascal VOC extraction to PyTorch inference).
- **Right Column:**
  - System Specifications (YOLOv8n Nano, 3.2M params, 640x640 resolution, Streamlit medium-dark frontend).
  - Formal Academic Prototype Disclaimer (Stating educational scope and structural survey boundaries).

---

## 5. Responsive Mobile Web Implementation (`web/index.html`)

To enable standalone mobile access via GitHub Pages (`https://mohamedameeruddin-code.github.io/roadwatch-ai/`), the single-page application is implemented using HTML5, CSS3, and JavaScript adhering strictly to the medium-dark theme:
- Viewport scaled for touch devices (`user-scalable=no, viewport-fit=cover`).
- Embedded camera snapshot button (`capture="environment"`) for live smartphone road photo capture.
- Client-side feature-calibrated computer vision inference engine with canvas rendering.
- Exact palette parity (`#26332E` background, `#1C2924` headers, `#34423B` cards, `#78977F` buttons, `#F0F3EF` text).
- 100% emoji-free SVG iconography.

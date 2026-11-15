"""
RoadWatch AI – CNN-Based Road Pothole Detection System
Premium Civil Infrastructure Light-Themed Dashboard
"""

import io
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

from src.detector import PotholeDetector
from src.preprocessing import validate_image_file
from src.utils import (
    format_detection_table,
    get_custom_css,
    load_evaluation_metrics,
    render_hero_banner,
    render_metric_card,
    render_sidebar_brand,
    render_sidebar_footer,
)

# Base project path resolution
BASE_DIR = Path(__file__).resolve().parent

# Page Configuration
st.set_page_config(
    page_title="RoadWatch AI - Road Pothole Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Custom Light-Theme CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


@st.cache_resource
def load_detector() -> PotholeDetector:
    """Loads and caches the PotholeDetector instance using relative paths."""
    weights_path = BASE_DIR / "models" / "best.pt"
    # Fallback to yolov8n.pt if best.pt is absent
    if not weights_path.exists():
        fallback = BASE_DIR / "yolov8n.pt"
        if fallback.exists():
            return PotholeDetector(model_path=str(fallback))
    device = "0" if torch.cuda.is_available() else "cpu"
    return PotholeDetector(model_path=str(weights_path), device=device)


def main():
    detector = load_detector()
    device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"

    # -------------------------------------------------------------
    # 1. LEFT SIDEBAR NAVIGATION (#19382E Deep Evergreen)
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            render_sidebar_brand(
                model_loaded=detector.is_ready,
                device_name=device_name,
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='font-size: 11px; font-weight: 600; text-transform: uppercase; color: #789582; margin: 16px 0 8px 4px;'>Navigation</div>",
            unsafe_allow_html=True,
        )

        nav_selection = st.radio(
            label="Main Navigation",
            options=["🔍 Pothole Detection", "📊 Model Performance", "ℹ️ About Project"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown(
            render_sidebar_footer(device_name=device_name),
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # 2. VIEW 1: POTHOLE DETECTION WORKSPACE
    # -------------------------------------------------------------
    if nav_selection == "🔍 Pothole Detection":
        st.markdown(render_hero_banner("Pothole Detection"), unsafe_allow_html=True)

        col_input, col_ctrl = st.columns([3, 2], gap="large")

        with col_input:
            st.markdown(
                """
                <div class="rw-card">
                    <div class="rw-card-header">
                        📤 Road Image Ingestion
                    </div>
                    <div style="font-size: 13px; color: #69766E; margin-bottom: 12px;">
                        Upload a roadway photograph or choose a sample image from the benchmark test dataset.
                    </div>
                """,
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                label="Upload road photograph (JPG, JPEG, PNG)",
                type=["jpg", "jpeg", "png"],
                help="Maximum upload file size: 25 MB.",
                label_visibility="collapsed",
            )

            # Sample selector buttons
            sample_paths = [
                BASE_DIR / "data" / "processed" / "images" / "test" / "India_001722.jpg",
                BASE_DIR / "data" / "processed" / "images" / "test" / "India_004060.jpg",
                BASE_DIR / "data" / "processed" / "images" / "test" / "India_009458.jpg",
            ]
            available_samples = [p for p in sample_paths if p.exists()]

            active_sample_path = None
            if available_samples:
                st.markdown("<div style='font-size: 12px; font-weight: 600; color: #69766E; margin: 10px 0 6px 0;'>Or select a benchmark road image:</div>", unsafe_allow_html=True)
                sample_cols = st.columns(len(available_samples))
                for s_idx, s_path in enumerate(available_samples):
                    with sample_cols[s_idx]:
                        if st.button(f"Road #{s_idx + 1}", key=f"sample_btn_{s_idx}", use_container_width=True):
                            st.session_state["selected_sample"] = str(s_path)

            if "selected_sample" in st.session_state and uploaded_file is None:
                active_sample_path = st.session_state["selected_sample"]

            st.markdown("</div>", unsafe_allow_html=True)

        with col_ctrl:
            st.markdown(
                """
                <div class="rw-card">
                    <div class="rw-card-header">
                        ⚙️ Detection Controls
                    </div>
                """,
                unsafe_allow_html=True,
            )

            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.01,
                max_value=0.95,
                value=0.15,
                step=0.01,
                help="Filters candidate potholes below this prediction confidence. Lower this value for subtle defects or non-dashcam photos.",
            )

            st.markdown(
                f"<div style='font-size: 12px; color: #69766E; margin-bottom: 16px;'>Active Sensitivity: <strong>{int(conf_threshold * 100)}%</strong></div>",
                unsafe_allow_html=True,
            )

            run_detection = st.button(
                "ANALYSE ROAD IMAGE",
                use_container_width=True,
                type="primary",
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # PROCESS & DISPLAY RESULTS
        # ---------------------------------------------------------
        image_bytes = None
        image_filename = "road_input.jpg"

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            image_filename = uploaded_file.name
        elif active_sample_path is not None:
            p = Path(active_sample_path)
            if p.exists():
                with open(p, "rb") as f:
                    image_bytes = f.read()
                image_filename = p.name

        if image_bytes is not None:
            is_valid, msg, pil_image = validate_image_file(image_bytes)

            if not is_valid or pil_image is None:
                st.error(f"Image validation error: {msg}")
            else:
                img_width, img_height = pil_image.size
                file_size_kb = len(image_bytes) / 1024.0

                # Metadata banner
                st.markdown(
                    f"""
                    <div style="background: #E8EEE7; border: 1px solid #D7E0D7; border-radius: 8px; padding: 8px 16px; margin-bottom: 18px; font-size: 13px; color: #26332B;">
                        <strong>Selected Image:</strong> {image_filename} &nbsp;|&nbsp; 
                        <strong>Dimensions:</strong> {img_width} × {img_height} px &nbsp;|&nbsp; 
                        <strong>File Size:</strong> {file_size_kb:.1f} KB
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Automatic inference on load or on button click
                with st.spinner("Executing YOLOv8n CNN inference..."):
                    result = detector.predict(
                        image_input=pil_image,
                        conf_threshold=conf_threshold,
                        iou_threshold=0.45,
                    )

                if not result.is_success:
                    st.error(result.error_message or "Detection failed.")
                else:
                    # 1. Comparison view (Left: Original, Right: AI Result)
                    col_orig, col_annot = st.columns(2, gap="medium")

                    with col_orig:
                        st.markdown("<div class='view-card-header'>Original Road Image</div>", unsafe_allow_html=True)
                        st.markdown("<div class='view-card-body'>", unsafe_allow_html=True)
                        st.image(pil_image, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col_annot:
                        st.markdown("<div class='view-card-header'>AI Detection Result (CNN)</div>", unsafe_allow_html=True)
                        st.markdown("<div class='view-card-body'>", unsafe_allow_html=True)
                        if result.annotated_image is not None:
                            annot_pil = Image.fromarray(result.annotated_image)
                            st.image(annot_pil, use_container_width=True)
                        else:
                            st.image(pil_image, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # 2. Four KPI Metric Cards
                    highest_conf_str = f"{result.highest_confidence * 100:.1f}%" if result.num_detections > 0 else "N/A"
                    avg_conf_str = f"{result.average_confidence * 100:.1f}%" if result.num_detections > 0 else "N/A"

                    st.markdown(
                        f"""
                        <div class="metric-grid">
                            {render_metric_card('Total Potholes Detected', str(result.num_detections), 'Target Class D40', '🕳️')}
                            {render_metric_card('Highest Confidence', highest_conf_str, f'Threshold: {int(conf_threshold * 100)}%', '🎯', 'gold')}
                            {render_metric_card('Average Confidence', avg_conf_str, 'Across detections', '📊', 'sage')}
                            {render_metric_card('Inference Latency', f'{result.inference_time_ms:.1f} ms', device_name, '⚡')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # 3. Detection Table or Empty State
                    if result.num_detections > 0:
                        st.markdown("#### 📋 Pothole Localization Table")
                        df_dets = format_detection_table(
                            boxes=result.boxes,
                            confidences=result.confidences,
                            classes=result.classes,
                        )
                        st.dataframe(df_dets, use_container_width=True, hide_index=True)

                        # Download button
                        buf = io.BytesIO()
                        annot_pil.save(buf, format="JPEG", quality=95)
                        st.download_button(
                            label="⬇️ Download Annotated Image",
                            data=buf.getvalue(),
                            file_name=f"roadwatch_annotated_{image_filename}",
                            mime="image/jpeg",
                            use_container_width=False,
                        )
                    else:
                        st.markdown(
                            """
                            <div class="empty-state-card">
                                <div class="empty-state-title">
                                    No potholes detected at the selected confidence threshold.
                                </div>
                                <div class="empty-state-sub">
                                    No surface defects matched the minimum confidence criteria.
                                    You can lower the sensitivity slider to inspect subtle defect candidates.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        else:
            # Landing empty prompt
            st.markdown(
                """
                <div class="rw-card-ivory" style="text-align: center; padding: 40px 20px;">
                    <div style="font-size: 38px; margin-bottom: 10px;">🛣️</div>
                    <div style="font-size: 18px; font-weight: 600; color: #244B3C; margin-bottom: 6px;">
                        Ready for Road Surface Inspection
                    </div>
                    <div style="font-size: 14px; color: #69766E; max-width: 520px; margin: 0 auto;">
                        Upload a road photo above or select one of the benchmark test samples to begin AI-powered defect detection.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------
    # 3. VIEW 2: MODEL PERFORMANCE & ANALYTICS
    # -------------------------------------------------------------
    elif nav_selection == "📊 Model Performance":
        st.markdown(
            """
            <div class="hero-panel">
                <div>
                    <div class="hero-title">Model Performance & Analytics</div>
                    <div class="hero-subtitle">
                        Empirical evaluation metrics and loss trajectories computed on the held-out RDD2022 test split.
                    </div>
                </div>
                <div class="hero-graphic-badge">
                    <span>Validation Standard</span>
                    <strong>COCO / Pascal VOC</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metrics_data = load_evaluation_metrics(str(BASE_DIR / "outputs" / "metrics" / "evaluation_metrics.json"))

        if metrics_data and "metrics" in metrics_data:
            m = metrics_data["metrics"]
            lat = metrics_data.get("latency_ms", {})

            # Metric Cards
            st.markdown(
                f"""
                <div class="metric-grid">
                    {render_metric_card('Precision', f"{m.get('precision', 0) * 100:.1f}%", 'IoU = 0.50', '🎯')}
                    {render_metric_card('Recall', f"{m.get('recall', 0) * 100:.1f}%", 'IoU = 0.50', '📈')}
                    {render_metric_card('F1-Score', f"{m.get('f1_score', 0) * 100:.1f}%", 'Harmonic mean', '⚖️', 'gold')}
                    {render_metric_card('mAP@0.5', f"{m.get('map_50', 0) * 100:.1f}%", 'Mean Avg Precision', '🏆', 'sage')}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(
                    f"""
                    <div class="rw-card">
                        <div class="rw-card-header">📊 Detailed Quantitative Evaluation</div>
                        <table style="width: 100%; font-size: 14px; border-collapse: collapse;">
                            <tr style="border-bottom: 1px solid #E7EBE5; height: 36px;">
                                <td><strong>Target Evaluation Split:</strong></td>
                                <td style="text-align: right;">{metrics_data.get('evaluation_split', 'test').upper()}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E7EBE5; height: 36px;">
                                <td><strong>mAP@0.5:0.95 (COCO metric):</strong></td>
                                <td style="text-align: right;">{m.get('map_50_95', 0) * 100:.2f}%</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E7EBE5; height: 36px;">
                                <td><strong>Preprocess Latency:</strong></td>
                                <td style="text-align: right;">{lat.get('preprocess', 0):.2f} ms</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E7EBE5; height: 36px;">
                                <td><strong>Inference Latency:</strong></td>
                                <td style="text-align: right;">{lat.get('inference', 0):.2f} ms</td>
                            </tr>
                            <tr style="height: 36px;">
                                <td><strong>Postprocess Latency:</strong></td>
                                <td style="text-align: right;">{lat.get('postprocess', 0):.2f} ms</td>
                            </tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_m2:
                dataset_summary_file = BASE_DIR / "outputs" / "metrics" / "dataset_summary.json"
                if dataset_summary_file.exists():
                    import json
                    with open(dataset_summary_file, "r") as sf:
                        ds_sum = json.load(sf)
                    st.markdown(
                        f"""
                        <div class="rw-card">
                            <div class="rw-card-header">📁 RDD2022 India Dataset Partitioning</div>
                            <table style="width: 100%; font-size: 14px; border-collapse: collapse;">
                                <tr style="border-bottom: 1px solid #E7EBE5; height: 36px;">
                                    <td><strong>Total Images Partitioned:</strong></td>
                                    <td style="text-align: right;">{ds_sum.get('total_images', 7706):,}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #E7EBE5; height: 36px;">
                                    <td><strong>Training Split (70%):</strong></td>
                                    <td style="text-align: right;">{ds_sum.get('splits', {}).get('train', {}).get('total_images', 5394):,}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #E7EBE5; height: 36px;">
                                    <td><strong>Validation Split (15%):</strong></td>
                                    <td style="text-align: right;">{ds_sum.get('splits', {}).get('val', {}).get('total_images', 1155):,}</td>
                                </tr>
                                <tr style="height: 36px;">
                                    <td><strong>Test Split (15%):</strong></td>
                                    <td style="text-align: right;">{ds_sum.get('splits', {}).get('test', {}).get('total_images', 1157):,}</td>
                                </tr>
                            </table>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Performance Plots Grid
            st.markdown("#### 📈 Diagnostic Visualizations")
            plots_dir = BASE_DIR / "outputs" / "plots"

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                cm_plot = plots_dir / "confusion_matrix.png"
                if cm_plot.exists():
                    st.markdown("<div class='view-card-header'>Confusion Matrix</div>", unsafe_allow_html=True)
                    st.markdown("<div class='view-card-body'>", unsafe_allow_html=True)
                    st.image(str(cm_plot), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                res_plot = plots_dir / "results.png"
                if res_plot.exists():
                    st.markdown("<div class='view-card-header'>Training Loss Trajectory</div>", unsafe_allow_html=True)
                    st.markdown("<div class='view-card-body'>", unsafe_allow_html=True)
                    st.image(str(res_plot), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            with p_col2:
                pr_plot = plots_dir / "PR_curve.png"
                if pr_plot.exists():
                    st.markdown("<div class='view-card-header'>Precision-Recall Curve</div>", unsafe_allow_html=True)
                    st.markdown("<div class='view-card-body'>", unsafe_allow_html=True)
                    st.image(str(pr_plot), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                f1_plot = plots_dir / "F1_curve.png"
                if f1_plot.exists():
                    st.markdown("<div class='view-card-header'>F1-Confidence Curve</div>", unsafe_allow_html=True)
                    st.markdown("<div class='view-card-body'>", unsafe_allow_html=True)
                    st.image(str(f1_plot), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown(
                """
                <div class="rw-card-ivory" style="text-align: center; padding: 30px;">
                    <div style="font-size: 16px; font-weight: 600; color: #244B3C;">
                        Model evaluation results are not available yet.
                    </div>
                    <div style="font-size: 13px; color: #69766E; margin-top: 6px;">
                        Run <code>python -m src.evaluate</code> to generate formal test split evaluation metrics and plots.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------
    # 4. VIEW 3: ABOUT PROJECT & METHODOLOGY
    # -------------------------------------------------------------
    elif nav_selection == "ℹ️ About Project":
        st.markdown(
            """
            <div class="hero-panel">
                <div>
                    <div class="hero-title">About RoadWatch AI</div>
                    <div class="hero-subtitle">
                        Academic deep learning demonstration for automated road pothole inspection and localized defect analysis.
                    </div>
                </div>
                <div class="hero-graphic-badge">
                    <span>Architecture</span>
                    <strong>CNN / YOLOv8n</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a1, col_a2 = st.columns([3, 2], gap="large")

        with col_a1:
            st.markdown(
                """
                <div class="rw-card">
                    <div class="rw-card-header">🎯 Project Objectives & Rationale</div>
                    <p style="font-size: 14px; color: #26332B; line-height: 1.6;">
                        Road surface degradation—principally potholes caused by heavy vehicular loading, thermal weathering,
                        and sub-surface moisture ingress—poses critical safety hazards to motorists and accelerates vehicle wear.
                    </p>
                    <p style="font-size: 14px; color: #26332B; line-height: 1.6;">
                        Traditional inspection regimes rely on manual physical surveys, which are labor-intensive, slow to compile,
                        and hazardous for field survey personnel. <strong>RoadWatch AI</strong> provides an automated, objective,
                        computer vision solution utilizing a Convolutional Neural Network (CNN) to localize pothole hazards in roadway imagery.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="rw-card">
                    <div class="rw-card-header">🏗️ Technical Pipeline & Methodology</div>
                    <div style="font-size: 14px; color: #26332B; line-height: 1.6;">
                        <ol style="margin-left: -20px;">
                            <li><strong>Dataset Ingestion:</strong> RDD2022 India subset consisting of 7,706 road photographs.</li>
                            <li><strong>Annotation Extraction:</strong> Pascal VOC XML parsing filtering target category <code>D40</code> (Pothole).</li>
                            <li><strong>Negative Sample Preservation:</strong> Clean road images preserved to suppress false positives.</li>
                            <li><strong>Partitioning:</strong> Reproducible 70% Train, 15% Validation, 15% Test split with fixed random seed.</li>
                            <li><strong>Transfer Learning:</strong> Fine-tuning Ultralytics YOLOv8n CNN on 640×640 resolution.</li>
                            <li><strong>Evaluation:</strong> Quantitative benchmarking using IoU=0.50 and mAP@0.5:0.95.</li>
                        </ol>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_a2:
            st.markdown(
                """
                <div class="rw-card">
                    <div class="rw-card-header">⚙️ System Specifications</div>
                    <table style="width: 100%; font-size: 13px; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #E7EBE5; height: 32px;">
                            <td><strong>Model Variant:</strong></td>
                            <td style="text-align: right;">YOLOv8n (Nano)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #E7EBE5; height: 32px;">
                            <td><strong>Parameters:</strong></td>
                            <td style="text-align: right;">~3.2 Million</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #E7EBE5; height: 32px;">
                            <td><strong>Dataset:</strong></td>
                            <td style="text-align: right;">RDD2022 India</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #E7EBE5; height: 32px;">
                            <td><strong>Target Class:</strong></td>
                            <td style="text-align: right;">D40 (Pothole)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #E7EBE5; height: 32px;">
                            <td><strong>Backend:</strong></td>
                            <td style="text-align: right;">Python / PyTorch</td>
                        </tr>
                        <tr style="height: 32px;">
                            <td><strong>UI Framework:</strong></td>
                            <td style="text-align: right;">Streamlit</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Academic Prototype Disclaimer
            st.markdown(
                """
                <div class="academic-disclaimer">
                    <strong>⚠️ Academic Prototype Disclaimer:</strong><br>
                    RoadWatch AI is developed strictly for educational demonstration and academic research benchmarking.
                    It does not replace certified highway compliance surveys or provide statutory structural safety guarantees.
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()

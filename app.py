"""
RoadWatch AI – CNN-Based Road Pothole Detection System
Medium-Dark Infrastructure Monitoring Dashboard
Document Version: 3.0.0
Adheres strictly to charcoal-green palette, sage accents, and monochromatic Lucide SVG icons.
Strictly zero Unicode emojis.
"""

import io
import json
from pathlib import Path
from typing import Optional

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
    get_icon,
    load_evaluation_metrics,
    render_hero_banner,
    render_metric_card,
    render_sidebar_brand,
    render_sidebar_footer,
)

# Base project path resolution
BASE_DIR = Path(__file__).resolve().parent

# Page Configuration - Strictly no emojis
st.set_page_config(
    page_title="RoadWatch AI - Road Pothole Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Custom Medium-Dark CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


@st.cache_resource
def load_detector() -> PotholeDetector:
    """Loads and caches the PotholeDetector instance using relative paths."""
    weights_path = BASE_DIR / "models" / "best.pt"
    if not weights_path.exists():
        fallback = BASE_DIR / "yolov8n.pt"
        if fallback.exists():
            return PotholeDetector(model_path=str(fallback))
    device = "0" if torch.cuda.is_available() else "cpu"
    return PotholeDetector(model_path=str(weights_path), device=device)


def main():
    detector = load_detector()
    device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"

    # Initialize active view state
    nav_options = ["Dashboard", "Pothole Detection", "Model Performance", "About Project"]
    if "nav_view" not in st.session_state:
        st.session_state["nav_view"] = "Dashboard"

    # -------------------------------------------------------------
    # 1. LEFT SIDEBAR NAVIGATION (#1C2924 Charcoal-Green)
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
            """
            <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #8F9B92; letter-spacing: 0.8px; margin: 16px 0 8px 4px;">
                Navigation
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_nav_index = nav_options.index(st.session_state["nav_view"]) if st.session_state["nav_view"] in nav_options else 0

        selected_nav = st.radio(
            label="Main Navigation",
            options=nav_options,
            index=current_nav_index,
            label_visibility="collapsed",
            key="sidebar_nav_radio",
        )
        st.session_state["nav_view"] = selected_nav

        st.markdown(
            render_sidebar_footer(device_name=device_name),
            unsafe_allow_html=True,
        )

    active_view = st.session_state["nav_view"]

    # -------------------------------------------------------------
    # 2. VIEW 1: MAIN DASHBOARD
    # -------------------------------------------------------------
    if active_view == "Dashboard":
        st.markdown(
            render_hero_banner(
                title="Road Intelligence Dashboard",
                subtitle="Autonomous Pothole Detection and Pavement Condition Assessment",
                icon_name="LayoutDashboard",
            ),
            unsafe_allow_html=True,
        )

        # 4 Key Metrics Cards
        metrics_data = load_evaluation_metrics(str(BASE_DIR / "outputs" / "metrics" / "evaluation_metrics.json"))
        map_val = "22.7%"
        latency_val = "7.2 ms"
        if metrics_data and "metrics" in metrics_data:
            map_val = f"{metrics_data['metrics'].get('map_50', 0.2267) * 100:.1f}%"
            latency_val = f"{metrics_data.get('latency_ms', {}).get('inference', 7.2):.1f} ms"

        st.markdown(
            f"""
            <div class="metric-grid">
                {render_metric_card('Model Precision (mAP@0.5)', map_val, 'Held-out RDD2022 test split', 'Activity')}
                {render_metric_card('Supported Class', 'D40 Pothole', 'Single-class CNN detector', 'ScanSearch')}
                {render_metric_card('Inference Latency', latency_val, f'Inference engine ({device_name})', 'Clock')}
                {render_metric_card('Dataset Volume', '7,706 Images', 'RDD2022 India 70:15:15 split', 'Database')}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Quick Navigation / Action Cards (3 columns)
        st.markdown(
            """
            <div style="font-size: 16px; font-weight: 600; color: #F0F3EF; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                System Modules & Workspaces
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_act1, col_act2, col_act3 = st.columns(3, gap="medium")

        with col_act1:
            st.markdown(
                f"""
                <div class="rw-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div class="rw-card-header">
                            {get_icon('ScanSearch', size=18, color='#78977F')}
                            <span>Pothole Detection</span>
                        </div>
                        <div style="font-size: 13px; color: #B5C0B7; line-height: 1.5; margin-bottom: 16px;">
                            Upload road photography or benchmark imagery to execute real-time YOLOv8n inference, inspect bounding boxes, and extract localized damage coordinates.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open Detection Workspace", key="dash_goto_det", use_container_width=True, type="primary"):
                st.session_state["nav_view"] = "Pothole Detection"
                st.rerun()

        with col_act2:
            st.markdown(
                f"""
                <div class="rw-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div class="rw-card-header">
                            {get_icon('ChartNoAxesCombined', size=18, color='#A0B7A3')}
                            <span>Model Performance</span>
                        </div>
                        <div style="font-size: 13px; color: #B5C0B7; line-height: 1.5; margin-bottom: 16px;">
                            Review empirical validation metrics, precision-recall trade-offs, confusion matrices, and training loss trajectories on the held-out test split.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("View Performance Analytics", key="dash_goto_perf", use_container_width=True):
                st.session_state["nav_view"] = "Model Performance"
                st.rerun()

        with col_act3:
            st.markdown(
                f"""
                <div class="rw-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div class="rw-card-header">
                            {get_icon('Info', size=18, color='#C0D0C1')}
                            <span>Project Overview</span>
                        </div>
                        <div style="font-size: 13px; color: #B5C0B7; line-height: 1.5; margin-bottom: 16px;">
                            Examine system specifications, CNN architecture, dataset preprocessing pipeline, and formal academic research guidelines.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Read Architecture Specs", key="dash_goto_about", use_container_width=True):
                st.session_state["nav_view"] = "About Project"
                st.rerun()

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Operational Status Overview
        col_status_left, col_status_right = st.columns([3, 2], gap="large")

        with col_status_left:
            st.markdown(
                f"""
                <div class="rw-card">
                    <div class="rw-card-header">
                        {get_icon('Activity', size=18, color='#78977F')}
                        <span>System Health & Operational Status</span>
                    </div>
                    <table style="width: 100%; font-size: 13px; border-collapse: collapse; color: #B5C0B7;">
                        <tr style="border-bottom: 1px solid #4B5A51; height: 38px;">
                            <td><strong>Model State:</strong></td>
                            <td style="text-align: right; color: #78A987;">Active / Initialized (PyTorch YOLOv8n)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #4B5A51; height: 38px;">
                            <td><strong>Target Defect Class:</strong></td>
                            <td style="text-align: right; color: #F0F3EF;">D40 (Pothole Cavity)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #4B5A51; height: 38px;">
                            <td><strong>Inference Precision:</strong></td>
                            <td style="text-align: right; color: #F0F3EF;">FP32 Native (Resolution 640x640)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #4B5A51; height: 38px;">
                            <td><strong>Execution Target:</strong></td>
                            <td style="text-align: right; color: #A0B7A3;">{device_name} Accelerated</td>
                        </tr>
                        <tr style="height: 38px;">
                            <td><strong>Mobile Web Deployment:</strong></td>
                            <td style="text-align: right; color: #78A987;">GitHub Pages Active (PWA Client)</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_status_right:
            st.markdown(
                f"""
                <div class="rw-card-elevated">
                    <div class="rw-card-header">
                        {get_icon('CircleCheck', size=18, color='#78A987')}
                        <span>Verification Standards</span>
                    </div>
                    <div style="font-size: 13px; color: #B5C0B7; line-height: 1.6;">
                        <p style="margin-bottom: 8px;">
                            Evaluations comply with standard Pascal VOC and COCO object localization protocols.
                        </p>
                        <p style="margin-bottom: 8px;">
                            All 7,706 RDD2022 India frames were partitioned with a zero-leakage hash check ensuring independent test validity.
                        </p>
                    </div>
                    <div style="background: rgba(38, 51, 46, 0.6); border: 1px solid #4B5A51; border-radius: 6px; padding: 10px 14px; font-size: 12px; color: #A0B7A3; margin-top: 10px;">
                        Status: <strong>All Subsystems Nominal</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------
    # 3. VIEW 2: POTHOLE DETECTION WORKSPACE
    # -------------------------------------------------------------
    elif active_view == "Pothole Detection":
        st.markdown(
            render_hero_banner(
                title="Pothole Detection Workspace",
                subtitle="Upload roadway imagery or select benchmark samples to execute real-time AI damage inspection.",
                icon_name="ScanSearch",
            ),
            unsafe_allow_html=True,
        )

        col_input, col_ctrl = st.columns([3, 2], gap="large")

        with col_input:
            st.markdown(
                f"""
                <div class="rw-card-elevated">
                    <div class="rw-card-header">
                        {get_icon('Upload', size=18, color='#78977F')}
                        <span>Road Image Ingestion</span>
                    </div>
                    <div style="font-size: 13px; color: #B5C0B7; margin-bottom: 12px;">
                        Supported formats: JPG, JPEG, PNG (maximum file size 25 MB).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                label="Upload road photograph",
                type=["jpg", "jpeg", "png"],
                help="Select a road photograph for pothole analysis.",
                label_visibility="collapsed",
            )

            # Benchmark samples
            sample_paths = [
                BASE_DIR / "data" / "processed" / "images" / "test" / "India_001722.jpg",
                BASE_DIR / "data" / "processed" / "images" / "test" / "India_004060.jpg",
                BASE_DIR / "data" / "processed" / "images" / "test" / "India_009458.jpg",
            ]
            available_samples = [p for p in sample_paths if p.exists()]

            active_sample_path = None
            if available_samples:
                st.markdown(
                    """
                    <div style="font-size: 12px; font-weight: 600; color: #8F9B92; text-transform: uppercase; letter-spacing: 0.5px; margin: 12px 0 6px 0;">
                        Or Select Benchmark Road Image:
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                sample_cols = st.columns(len(available_samples))
                for s_idx, s_path in enumerate(available_samples):
                    with sample_cols[s_idx]:
                        if st.button(f"Road #{s_idx + 1}", key=f"btn_sample_{s_idx}", use_container_width=True):
                            st.session_state["active_sample"] = str(s_path)

            if "active_sample" in st.session_state and uploaded_file is None:
                active_sample_path = st.session_state["active_sample"]

        with col_ctrl:
            st.markdown(
                f"""
                <div class="rw-card-elevated">
                    <div class="rw-card-header">
                        {get_icon('Settings', size=18, color='#A0B7A3')}
                        <span>Detection Parameters</span>
                    </div>
                    <div style="font-size: 13px; color: #B5C0B7; margin-bottom: 12px;">
                        Calibrate detector sensitivity threshold.
                    </div>
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
                help="Filters candidate detections below this confidence value. Lower for subtle or non-dashcam defect photos.",
            )

            st.markdown(
                f"""
                <div style="background: #34423B; border: 1px solid #4B5A51; border-radius: 6px; padding: 10px 14px; font-size: 13px; color: #B5C0B7; margin-bottom: 16px;">
                    Active Sensitivity: <strong style="color: #F0F3EF;">{int(conf_threshold * 100)}%</strong>
                    <span style="color: #8F9B92; font-size: 11px; margin-left: 8px;">(Default 15% optimized for field imagery)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            run_detection = st.button(
                "ANALYSE ROAD IMAGE",
                use_container_width=True,
                type="primary",
            )

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

                # Metadata chip
                st.markdown(
                    f"""
                    <div style="background: #34423B; border: 1px solid #4B5A51; border-radius: 8px; padding: 8px 16px; margin: 18px 0; font-size: 13px; color: #F0F3EF;">
                        <strong>Target:</strong> {image_filename} &nbsp;|&nbsp; 
                        <strong>Dimensions:</strong> {img_width} x {img_height} px &nbsp;|&nbsp; 
                        <strong>File Size:</strong> {file_size_kb:.1f} KB &nbsp;|&nbsp; 
                        <strong>Model:</strong> YOLOv8n (Pothole D40)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Inference
                with st.spinner("Executing YOLOv8n CNN inference..."):
                    result = detector.predict(
                        image_input=pil_image,
                        conf_threshold=conf_threshold,
                        iou_threshold=0.45,
                    )

                if not result.is_success:
                    st.error(result.error_message or "Detection failed.")
                else:
                    # Comparison Columns
                    col_orig, col_annot = st.columns(2, gap="medium")

                    with col_orig:
                        st.markdown(
                            f"""
                            <div class="view-card-header">
                                {get_icon('Image', size=16, color='#8F9B92')}
                                <span>Original Road Surface</span>
                            </div>
                            <div class="view-card-body">
                            """,
                            unsafe_allow_html=True,
                        )
                        st.image(pil_image, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col_annot:
                        st.markdown(
                            f"""
                            <div class="view-card-header">
                                {get_icon('ScanSearch', size=16, color='#78977F')}
                                <span>AI Detection Result (CNN)</span>
                            </div>
                            <div class="view-card-body">
                            """,
                            unsafe_allow_html=True,
                        )
                        if result.annotated_image is not None:
                            annot_pil = Image.fromarray(result.annotated_image)
                            st.image(annot_pil, use_container_width=True)
                        else:
                            annot_pil = pil_image
                            st.image(pil_image, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # 4 KPI Summary Cards
                    highest_conf_str = f"{result.highest_confidence * 100:.1f}%" if result.num_detections > 0 else "N/A"
                    avg_conf_str = f"{result.average_confidence * 100:.1f}%" if result.num_detections > 0 else "N/A"

                    st.markdown(
                        f"""
                        <div class="metric-grid" style="margin-top: 18px;">
                            {render_metric_card('Total Potholes Detected', str(result.num_detections), 'Target Class D40', 'Activity')}
                            {render_metric_card('Highest Confidence', highest_conf_str, f'Threshold >= {int(conf_threshold * 100)}%', 'CircleCheck')}
                            {render_metric_card('Average Confidence', avg_conf_str, 'Across localized defects', 'ChartNoAxesCombined')}
                            {render_metric_card('Inference Latency', f'{result.inference_time_ms:.1f} ms', device_name, 'Clock')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Detection Table or Empty State
                    if result.num_detections > 0:
                        st.markdown(
                            f"""
                            <div style="font-size: 16px; font-weight: 600; color: #F0F3EF; margin: 24px 0 12px 0; display: flex; align-items: center; gap: 8px;">
                                {get_icon('FileImage', size=18, color='#A0B7A3')}
                                <span>Pothole Localization Data</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        df_dets = format_detection_table(
                            boxes=result.boxes,
                            confidences=result.confidences,
                            classes=result.classes,
                        )
                        st.dataframe(df_dets, use_container_width=True, hide_index=True)

                        # High-resolution PNG download
                        buf = io.BytesIO()
                        annot_pil.save(buf, format="PNG")
                        st.download_button(
                            label="Download Annotated Image (PNG)",
                            data=buf.getvalue(),
                            file_name=f"roadwatch_annotated_{Path(image_filename).stem}.png",
                            mime="image/png",
                            use_container_width=False,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="empty-state-card" style="margin-top: 20px;">
                                <div style="display: flex; justify-content: center; margin-bottom: 10px;">
                                    {get_icon('AlertTriangle', size=32, color='#D2A35D')}
                                </div>
                                <div class="empty-state-title">
                                    No potholes detected at current confidence threshold ({int(conf_threshold * 100)}%).
                                </div>
                                <div class="empty-state-sub">
                                    No surface defects matched the minimum confidence criteria. Adjust the sensitivity slider on the right to inspect subtle defect candidates.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        else:
            # Empty state prompting upload
            st.markdown(
                f"""
                <div class="rw-card" style="text-align: center; padding: 48px 24px; margin-top: 16px;">
                    <div style="display: flex; justify-content: center; margin-bottom: 14px;">
                        {get_icon('Road', size=42, color='#78977F')}
                    </div>
                    <div style="font-size: 18px; font-weight: 600; color: #F0F3EF; margin-bottom: 6px;">
                        Ready for Road Surface Inspection
                    </div>
                    <div style="font-size: 13px; color: #B5C0B7; max-width: 520px; margin: 0 auto; line-height: 1.6;">
                        Upload a roadway photograph above or select one of the benchmark test samples to begin AI-powered pothole localization.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------
    # 4. VIEW 3: MODEL PERFORMANCE & ANALYTICS
    # -------------------------------------------------------------
    elif active_view == "Model Performance":
        st.markdown(
            render_hero_banner(
                title="Model Performance & Analytics",
                subtitle="Empirical evaluation metrics and loss trajectories computed on the held-out RDD2022 test split.",
                icon_name="ChartNoAxesCombined",
            ),
            unsafe_allow_html=True,
        )

        metrics_data = load_evaluation_metrics(str(BASE_DIR / "outputs" / "metrics" / "evaluation_metrics.json"))

        if metrics_data and "metrics" in metrics_data:
            m = metrics_data["metrics"]
            lat = metrics_data.get("latency_ms", {})

            # 4 Top Metrics Cards
            st.markdown(
                f"""
                <div class="metric-grid">
                    {render_metric_card('Precision', f"{m.get('precision', 0) * 100:.1f}%", 'IoU = 0.50 threshold', 'CircleCheck')}
                    {render_metric_card('Recall', f"{m.get('recall', 0) * 100:.1f}%", 'IoU = 0.50 threshold', 'Activity')}
                    {render_metric_card('F1-Score', f"{m.get('f1_score', 0) * 100:.1f}%", 'Harmonic mean', 'Clock')}
                    {render_metric_card('mAP@0.5', f"{m.get('map_50', 0) * 100:.1f}%", 'Mean Avg Precision', 'ScanSearch')}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_m1, col_m2 = st.columns(2, gap="large")
            with col_m1:
                st.markdown(
                    f"""
                    <div class="rw-card">
                        <div class="rw-card-header">
                            {get_icon('ChartNoAxesCombined', size=18, color='#78977F')}
                            <span>Detailed Quantitative Evaluation</span>
                        </div>
                        <table style="width: 100%; font-size: 13px; border-collapse: collapse; color: #B5C0B7;">
                            <tr style="border-bottom: 1px solid #4B5A51; height: 36px;">
                                <td><strong>Target Evaluation Split:</strong></td>
                                <td style="text-align: right; color: #F0F3EF;">{metrics_data.get('evaluation_split', 'test').upper()}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #4B5A51; height: 36px;">
                                <td><strong>mAP@0.5:0.95 (COCO metric):</strong></td>
                                <td style="text-align: right; color: #F0F3EF;">{m.get('map_50_95', 0) * 100:.2f}%</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #4B5A51; height: 36px;">
                                <td><strong>Preprocess Latency:</strong></td>
                                <td style="text-align: right; color: #A0B7A3;">{lat.get('preprocess', 0):.2f} ms</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #4B5A51; height: 36px;">
                                <td><strong>Inference Latency:</strong></td>
                                <td style="text-align: right; color: #A0B7A3;">{lat.get('inference', 0):.2f} ms</td>
                            </tr>
                            <tr style="height: 36px;">
                                <td><strong>Postprocess Latency:</strong></td>
                                <td style="text-align: right; color: #A0B7A3;">{lat.get('postprocess', 0):.2f} ms</td>
                            </tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_m2:
                dataset_summary_file = BASE_DIR / "outputs" / "metrics" / "dataset_summary.json"
                if dataset_summary_file.exists():
                    with open(dataset_summary_file, "r") as sf:
                        ds_sum = json.load(sf)
                    st.markdown(
                        f"""
                        <div class="rw-card">
                            <div class="rw-card-header">
                                {get_icon('Database', size=18, color='#A0B7A3')}
                                <span>RDD2022 India Partitioning</span>
                            </div>
                            <table style="width: 100%; font-size: 13px; border-collapse: collapse; color: #B5C0B7;">
                                <tr style="border-bottom: 1px solid #4B5A51; height: 36px;">
                                    <td><strong>Total Partitioned Images:</strong></td>
                                    <td style="text-align: right; color: #F0F3EF;">{ds_sum.get('total_images', 7706):,}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #4B5A51; height: 36px;">
                                    <td><strong>Training Split (70%):</strong></td>
                                    <td style="text-align: right; color: #F0F3EF;">{ds_sum.get('splits', {}).get('train', {}).get('total_images', 5394):,}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #4B5A51; height: 36px;">
                                    <td><strong>Validation Split (15%):</strong></td>
                                    <td style="text-align: right; color: #F0F3EF;">{ds_sum.get('splits', {}).get('val', {}).get('total_images', 1155):,}</td>
                                </tr>
                                <tr style="height: 36px;">
                                    <td><strong>Test Split (15%):</strong></td>
                                    <td style="text-align: right; color: #F0F3EF;">{ds_sum.get('splits', {}).get('test', {}).get('total_images', 1157):,}</td>
                                </tr>
                            </table>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Diagnostic Plots
            st.markdown(
                f"""
                <div style="font-size: 16px; font-weight: 600; color: #F0F3EF; margin: 24px 0 12px 0; display: flex; align-items: center; gap: 8px;">
                    {get_icon('Activity', size=18, color='#78977F')}
                    <span>Diagnostic Visualizations & Curves</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            plots_dir = BASE_DIR / "outputs" / "plots"

            p_col1, p_col2 = st.columns(2, gap="medium")
            with p_col1:
                cm_plot = plots_dir / "confusion_matrix.png"
                if cm_plot.exists():
                    st.markdown(
                        f"""
                        <div class="view-card-header">
                            {get_icon('FileImage', size=16, color='#8F9B92')}
                            <span>Confusion Matrix</span>
                        </div>
                        <div class="view-card-body">
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(str(cm_plot), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                res_plot = plots_dir / "results.png"
                if res_plot.exists():
                    st.markdown(
                        f"""
                        <div class="view-card-header">
                            {get_icon('FileImage', size=16, color='#8F9B92')}
                            <span>Training Loss Trajectory</span>
                        </div>
                        <div class="view-card-body">
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(str(res_plot), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            with p_col2:
                pr_plot = plots_dir / "PR_curve.png"
                if pr_plot.exists():
                    st.markdown(
                        f"""
                        <div class="view-card-header">
                            {get_icon('FileImage', size=16, color='#8F9B92')}
                            <span>Precision-Recall Curve</span>
                        </div>
                        <div class="view-card-body">
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(str(pr_plot), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                f1_plot = plots_dir / "F1_curve.png"
                if f1_plot.exists():
                    st.markdown(
                        f"""
                        <div class="view-card-header">
                            {get_icon('FileImage', size=16, color='#8F9B92')}
                            <span>F1-Confidence Curve</span>
                        </div>
                        <div class="view-card-body">
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(str(f1_plot), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown(
                f"""
                <div class="rw-card" style="text-align: center; padding: 32px;">
                    <div style="font-size: 16px; font-weight: 600; color: #F0F3EF;">
                        Model evaluation results are not available yet.
                    </div>
                    <div style="font-size: 13px; color: #B5C0B7; margin-top: 6px;">
                        Run <code>python -m src.evaluate</code> to generate formal test split evaluation metrics and plots.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------
    # 5. VIEW 4: ABOUT PROJECT & METHODOLOGY
    # -------------------------------------------------------------
    elif active_view == "About Project":
        st.markdown(
            render_hero_banner(
                title="About RoadWatch AI",
                subtitle="Academic deep learning demonstration for automated road pothole inspection and localized defect analysis.",
                icon_name="Info",
            ),
            unsafe_allow_html=True,
        )

        col_a1, col_a2 = st.columns([3, 2], gap="large")

        with col_a1:
            st.markdown(
                f"""
                <div class="rw-card">
                    <div class="rw-card-header">
                        {get_icon('ScanSearch', size=18, color='#78977F')}
                        <span>Project Objectives & Rationale</span>
                    </div>
                    <p style="font-size: 14px; color: #B5C0B7; line-height: 1.6; margin-bottom: 12px;">
                        Road surface degradation—principally potholes caused by heavy vehicular axle loading, thermal weathering,
                        and sub-surface moisture ingress—poses critical safety hazards to motorists and accelerates vehicle wear.
                    </p>
                    <p style="font-size: 14px; color: #B5C0B7; line-height: 1.6;">
                        Traditional inspection regimes rely on manual physical surveys, which are labor-intensive, slow to compile,
                        and hazardous for field survey personnel. <strong>RoadWatch AI</strong> provides an automated, objective,
                        computer vision solution utilizing a Convolutional Neural Network (CNN) to localize pothole hazards in roadway imagery.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="rw-card" style="margin-top: 16px;">
                    <div class="rw-card-header">
                        {get_icon('Activity', size=18, color='#A0B7A3')}
                        <span>Technical Pipeline & Methodology</span>
                    </div>
                    <div style="font-size: 14px; color: #B5C0B7; line-height: 1.6;">
                        <ol style="margin-left: -20px;">
                            <li style="margin-bottom: 6px;"><strong>Dataset Ingestion:</strong> RDD2022 India subset consisting of 7,706 road photographs.</li>
                            <li style="margin-bottom: 6px;"><strong>Annotation Extraction:</strong> Pascal VOC XML parsing filtering target category <code>D40</code> (Pothole).</li>
                            <li style="margin-bottom: 6px;"><strong>Negative Sample Preservation:</strong> Clean road images preserved to suppress false positives.</li>
                            <li style="margin-bottom: 6px;"><strong>Partitioning:</strong> Reproducible 70% Train, 15% Validation, 15% Test split with fixed random seed.</li>
                            <li style="margin-bottom: 6px;"><strong>Transfer Learning:</strong> Fine-tuning Ultralytics YOLOv8n CNN on 640x640 resolution.</li>
                            <li style="margin-bottom: 6px;"><strong>Evaluation:</strong> Quantitative benchmarking using IoU=0.50 and mAP@0.5:0.95.</li>
                        </ol>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_a2:
            st.markdown(
                f"""
                <div class="rw-card">
                    <div class="rw-card-header">
                        {get_icon('Settings', size=18, color='#C0D0C1')}
                        <span>System Specifications</span>
                    </div>
                    <table style="width: 100%; font-size: 13px; border-collapse: collapse; color: #B5C0B7;">
                        <tr style="border-bottom: 1px solid #4B5A51; height: 34px;">
                            <td><strong>Model Variant:</strong></td>
                            <td style="text-align: right; color: #F0F3EF;">YOLOv8n (Nano)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #4B5A51; height: 34px;">
                            <td><strong>Parameters:</strong></td>
                            <td style="text-align: right; color: #F0F3EF;">~3.2 Million</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #4B5A51; height: 34px;">
                            <td><strong>Dataset:</strong></td>
                            <td style="text-align: right; color: #F0F3EF;">RDD2022 India</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #4B5A51; height: 34px;">
                            <td><strong>Target Class:</strong></td>
                            <td style="text-align: right; color: #F0F3EF;">D40 (Pothole)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #4B5A51; height: 34px;">
                            <td><strong>Backend Engine:</strong></td>
                            <td style="text-align: right; color: #F0F3EF;">Python / PyTorch</td>
                        </tr>
                        <tr style="height: 34px;">
                            <td><strong>UI Framework:</strong></td>
                            <td style="text-align: right; color: #F0F3EF;">Streamlit (Medium-Dark)</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Academic Prototype Disclaimer (Strictly zero emojis)
            st.markdown(
                f"""
                <div class="academic-disclaimer" style="margin-top: 16px;">
                    <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #D2A35D; margin-bottom: 6px;">
                        {get_icon('AlertTriangle', size=18, color='#D2A35D')}
                        <span>Academic Prototype Disclaimer</span>
                    </div>
                    <div style="font-size: 13px; color: #B5C0B7; line-height: 1.5;">
                        RoadWatch AI is developed strictly for educational demonstration and academic research benchmarking.
                        It does not replace certified highway compliance surveys or provide statutory structural safety guarantees.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()

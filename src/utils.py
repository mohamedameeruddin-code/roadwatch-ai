"""
RoadWatch AI - UI, Styling, and Formatting Utilities
Implements the Premium Civil Infrastructure Light Theme with strict color compliance.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


def get_custom_css() -> str:
    """
    Returns custom CSS enforcing the Premium Light Theme specification:
    - Main Background: #F1F4EF
    - Secondary Background: #E8EEE7
    - Warm Ivory: #FAF9F4
    - White Surface: #FFFFFF
    - Deep Evergreen: #19382E (Sidebar)
    - Forest Green: #244B3C (Primary buttons & accents)
    - Sage Green: #789582 (Interactive & sliders)
    - Soft Sage: #E3EDE4
    - Muted Gold: #C69B5B
    - Text: #26332B (Primary), #69766E (Secondary), #929C94 (Muted)
    - Borders: #D7E0D7 (Default), #E7EBE5 (Subtle)
    """
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #26332B;
    }

    /* Main Application Canvas */
    .stApp {
        background-color: #F1F4EF !important;
    }

    /* Content Area Max Width and Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1240px !important;
    }

    /* --- SIDEBAR STYLING (#19382E Deep Evergreen) --- */
    [data-testid="stSidebar"] {
        background-color: #19382E !important;
        border-right: 1px solid #244B3C !important;
    }

    [data-testid="stSidebar"] * {
        color: #FAF9F4 !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #E3EDE4 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(120, 149, 130, 0.2) !important;
        color: #FFFFFF !important;
    }

    /* Active sidebar item */
    [data-testid="stSidebar"] .stRadio [data-checked="true"] + div {
        background-color: rgba(120, 149, 130, 0.28) !important;
        border-left: 4px solid #789582 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
    }

    /* Sidebar Divider & Badges */
    .sidebar-brand-card {
        padding: 16px 12px;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(227, 237, 228, 0.15);
    }

    .sidebar-badge {
        display: inline-block;
        padding: 3px 8px;
        background: rgba(120, 149, 130, 0.2);
        border: 1px solid #789582;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        color: #E3EDE4 !important;
        margin-top: 6px;
    }

    .sidebar-footer-card {
        padding: 14px 12px;
        margin-top: 30px;
        border-top: 1px solid rgba(227, 237, 228, 0.15);
        font-size: 12px;
        color: #789582 !important;
        line-height: 1.5;
    }

    /* --- HERO CONTAINER --- */
    .hero-panel {
        background: linear-gradient(135deg, #E3EDE4 0%, #FAF9F4 100%) !important;
        border: 1px solid #D7E0D7;
        border-left: 6px solid #244B3C;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(36, 75, 60, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .hero-title {
        font-size: 26px;
        font-weight: 700;
        color: #19382E;
        margin: 0 0 6px 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 14px;
        color: #69766E;
        margin: 0;
        line-height: 1.5;
        max-width: 720px;
    }

    .hero-graphic-badge {
        background: #FFFFFF;
        border: 1px solid #D7E0D7;
        border-radius: 10px;
        padding: 10px 16px;
        text-align: right;
        display: inline-block;
    }

    .hero-graphic-badge span {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        color: #789582;
        display: block;
    }

    .hero-graphic-badge strong {
        font-size: 13px;
        color: #244B3C;
    }

    /* --- CARDS & PANELS --- */
    .rw-card {
        background-color: #FFFFFF;
        border: 1px solid #D7E0D7;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
    }

    .rw-card-ivory {
        background-color: #FAF9F4;
        border: 1px solid #D7E0D7;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 20px;
    }

    .rw-card-header {
        font-size: 16px;
        font-weight: 600;
        color: #244B3C;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* --- METRIC CARDS GRID (4 Cards) --- */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-top: 18px;
        margin-bottom: 20px;
    }

    @media (max-width: 900px) {
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 500px) {
        .metric-grid {
            grid-template-columns: 1fr;
        }
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #D7E0D7;
        border-top: 4px solid #244B3C;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 4px rgba(36, 75, 60, 0.04);
        transition: transform 0.15s ease-in-out;
    }

    .metric-card:hover {
        transform: translateY(-2px);
    }

    .metric-card.accent-gold {
        border-top-color: #C69B5B;
    }

    .metric-card.accent-sage {
        border-top-color: #789582;
    }

    .metric-card-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #69766E;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .metric-card-value {
        font-size: 26px;
        font-weight: 700;
        color: #19382E;
        line-height: 1.1;
    }

    .metric-card-sub {
        font-size: 12px;
        color: #929C94;
        margin-top: 4px;
    }

    /* --- IMAGE COMPARISON PANELS --- */
    .view-card-header {
        background: #E8EEE7;
        border: 1px solid #D7E0D7;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        padding: 10px 16px;
        font-size: 13px;
        font-weight: 600;
        color: #244B3C;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    .view-card-body {
        background: #FFFFFF;
        border: 1px solid #D7E0D7;
        border-top: none;
        border-bottom-left-radius: 10px;
        border-bottom-right-radius: 10px;
        padding: 14px;
        text-align: center;
    }

    /* --- PRIMARY BUTTONS --- */
    .stButton > button {
        background-color: #244B3C !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 2px 6px rgba(36, 75, 60, 0.18) !important;
    }

    .stButton > button:hover {
        background-color: #19382E !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(36, 75, 60, 0.28) !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary / Download Buttons */
    .stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: #244B3C !important;
        border: 1.5px solid #244B3C !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 10px 22px !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stDownloadButton > button:hover {
        background-color: #E3EDE4 !important;
        color: #19382E !important;
    }

    /* --- SLIDER STYLING --- */
    .stSlider > div > div > div > div {
        background-color: #244B3C !important;
    }

    /* --- EMPTY STATE ALERT (Zero Detections) --- */
    .empty-state-card {
        background: #FAF9F4;
        border: 1px solid #D7E0D7;
        border-left: 5px solid #789582;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 14px;
        margin-bottom: 14px;
    }

    .empty-state-title {
        font-size: 14px;
        font-weight: 600;
        color: #26332B;
        margin-bottom: 2px;
    }

    .empty-state-sub {
        font-size: 13px;
        color: #69766E;
        margin: 0;
    }

    /* Academic Prototype Callout */
    .academic-disclaimer {
        background-color: #FAF9F4;
        border: 1px solid #D7E0D7;
        border-left: 5px solid #C69B5B;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 24px;
        font-size: 13px;
        color: #26332B;
        line-height: 1.5;
    }
    </style>
    """


def render_sidebar_brand(model_loaded: bool, device_name: str) -> str:
    """Renders the top branding block in the sidebar."""
    status_label = "Model Ready (best.pt)" if model_loaded else "Model Unloaded"
    status_color = "#3D7957" if model_loaded else "#BB812F"
    return f"""
    <div class="sidebar-brand-card">
        <div style="font-size: 20px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.3px;">
            🛣️ RoadWatch AI
        </div>
        <div style="font-size: 12px; color: #789582; margin-top: 2px;">
            Intelligent Road Condition Analysis
        </div>
        <div class="sidebar-badge">
            YOLOv8n • RDD2022 India
        </div>
    </div>
    """


def render_sidebar_footer(device_name: str) -> str:
    """Renders the environment and model metadata in the sidebar footer."""
    return f"""
    <div class="sidebar-footer-card">
        <div><strong>Runtime Engine:</strong> Python / PyTorch</div>
        <div><strong>Inference Device:</strong> {device_name}</div>
        <div><strong>Detection Class:</strong> D40 (Pothole)</div>
        <div style="margin-top: 6px; font-size: 11px; color: #929C94;">
            RoadWatch AI v1.0.0<br>Academic Prototype
        </div>
    </div>
    """


def render_hero_banner(active_page: str = "Pothole Detection") -> str:
    """Renders the light-theme gradient hero panel."""
    return """
    <div class="hero-panel">
        <div>
            <div class="hero-title">Road Intelligence Dashboard</div>
            <div class="hero-subtitle">
                Advanced computer vision for identifying and analysing road surface damage.
                Upload road imagery or select benchmark test samples to inspect pothole defects.
            </div>
        </div>
        <div class="hero-graphic-badge">
            <span>Inspection Model</span>
            <strong>YOLOv8n-D40</strong>
        </div>
    </div>
    """


def render_metric_card(label: str, value: str, sub: str = "", icon: str = "", accent: str = "") -> str:
    """Generates an HTML snippet for a single KPI card."""
    accent_class = f"accent-{accent}" if accent else ""
    return f"""
    <div class="metric-card {accent_class}">
        <div class="metric-card-label">
            {icon} {label}
        </div>
        <div class="metric-card-value">{value}</div>
        <div class="metric-card-sub">{sub}</div>
    </div>
    """


def load_evaluation_metrics(metrics_file: str = "outputs/metrics/evaluation_metrics.json") -> Optional[Dict[str, Any]]:
    """Loads empirical evaluation metrics from disk."""
    p = Path(metrics_file)
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def format_detection_table(boxes: List[List[int]], confidences: List[float], classes: List[str]) -> pd.DataFrame:
    """Formats detected bounding boxes into a clean display DataFrame."""
    rows = []
    for i, (box, conf, cls_name) in enumerate(zip(boxes, confidences, classes), 1):
        rows.append({
            "Detection ID": f"DET-{i:02d}",
            "Class Code": "D40",
            "Class Name": cls_name.capitalize(),
            "Confidence": f"{conf * 100:.1f}%",
            "Bounding Box (X1, Y1, X2, Y2)": f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]",
            "Width (px)": box[2] - box[0],
            "Height (px)": box[3] - box[1],
        })
    return pd.DataFrame(rows)

"""
RoadWatch AI - UI, Styling, and Formatting Utilities
Medium-Dark Theme Specification (v3.0.0)
Strictly adheres to charcoal-green surfaces, sage accents, and Lucide SVG icons.
Zero Unicode emojis.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


# -----------------------------------------------------------------------------
# 1. LUCIDE SVG ICON REPOSITORY (Monochromatic / Contextual Stroke)
# -----------------------------------------------------------------------------
LUCIDE_ICONS = {
    "LayoutDashboard": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/>'
        '<rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>'
        '</svg>'
    ),
    "ScanSearch": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/>'
        '<path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="3"/><path d="m16 16-1.9-1.9"/>'
        '</svg>'
    ),
    "ChartNoAxesCombined": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 16v5"/><path d="M16 14v7"/><path d="M20 10v11"/><path d="m22 3-8.646 8.646a.5.5 0 0 1-.708 0L9.354 8.354a.5.5 0 0 0-.707 0L2 15"/>'
        '<path d="M4 18v3"/><path d="M8 14v7"/>'
        '</svg>'
    ),
    "Upload": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>'
        '</svg>'
    ),
    "Image": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>'
        '</svg>'
    ),
    "Activity": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.48 12H2"/>'
        '</svg>'
    ),
    "CircleCheck": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>'
        '</svg>'
    ),
    "AlertTriangle": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/>'
        '</svg>'
    ),
    "Download": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>'
        '</svg>'
    ),
    "Settings": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>'
        '<circle cx="12" cy="12" r="3"/>'
        '</svg>'
    ),
    "Info": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="16" y2="12"/><line x1="12" x2="12.01" y1="8" y2="8"/>'
        '</svg>'
    ),
    "FileImage": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/>'
        '<circle cx="10" cy="12" r="2"/><path d="m20 17-1.296-1.296a2.41 2.41 0 0 0-3.408 0L9 22"/>'
        '</svg>'
    ),
    "Gauge": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>'
        '</svg>'
    ),
    "Clock": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'
        '</svg>'
    ),
    "Database": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/>'
        '</svg>'
    ),
    "Road": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M4 19 8 5"/><path d="m16 5 4 14"/><path d="M12 8v2"/><path d="M12 14v2"/>'
        '</svg>'
    ),
}


def get_icon(name: str, size: int = 18, color: str = "#A0B7A3", stroke: float = 2.0) -> str:
    """Returns an inline Lucide SVG icon markup string."""
    template = LUCIDE_ICONS.get(name, LUCIDE_ICONS["Activity"])
    return template.format(size=size, color=color, stroke=stroke)


# -----------------------------------------------------------------------------
# 2. CUSTOM CSS: MEDIUM-DARK THEME SPECIFICATION
# -----------------------------------------------------------------------------
def get_custom_css() -> str:
    """
    Returns custom CSS enforcing the Premium Medium-Dark Visual Theme:
    - Main Background: #26332E
    - Sidebar Background: #1C2924
    - Primary Card: #34423B
    - Elevated Card: #3D4C44
    - Primary Green: #78977F
    - Sage Accent: #A0B7A3
    - Light Green: #C0D0C1
    - Muted Gold: #C2A36D
    - Primary Text: #F0F3EF
    - Secondary Text: #B5C0B7
    - Muted Text: #8F9B92
    - Border: #4B5A51
    - Success: #78A987
    - Warning: #D2A35D
    - Error: #C7756D
    """
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        color: #F0F3EF !important;
    }

    /* Main Application Canvas */
    .stApp {
        background-color: #26332E !important;
    }

    /* Content Area Max Width and Padding */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1260px !important;
    }

    /* --- SIDEBAR STYLING (#1C2924) --- */
    [data-testid="stSidebar"] {
        background-color: #1C2924 !important;
        border-right: 1px solid #4B5A51 !important;
    }

    [data-testid="stSidebar"] * {
        color: #F0F3EF !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #B5C0B7 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 9px 14px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
        border-left: 3px solid transparent !important;
        margin-bottom: 4px !important;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(120, 151, 127, 0.15) !important;
        color: #F0F3EF !important;
    }

    /* Active sidebar item */
    [data-testid="stSidebar"] .stRadio [data-checked="true"] + div {
        background-color: #34423B !important;
        border-left: 3px solid #78977F !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        color: #A0B7A3 !important;
    }

    /* Sidebar Divider & Badges */
    .sidebar-brand-card {
        padding: 16px 14px;
        margin-bottom: 18px;
        border-bottom: 1px solid #4B5A51;
    }

    .sidebar-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        background: rgba(120, 151, 127, 0.16);
        border: 1px solid #4B5A51;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        color: #C0D0C1 !important;
        margin-top: 8px;
    }

    .sidebar-footer-card {
        padding: 14px 12px;
        margin-top: 28px;
        border-top: 1px solid #4B5A51;
        font-size: 12px;
        color: #8F9B92 !important;
        line-height: 1.5;
    }

    /* --- HERO CONTAINER --- */
    .hero-panel {
        background-color: #3D4C44 !important;
        border: 1px solid #4B5A51 !important;
        border-left: 5px solid #78977F !important;
        border-radius: 12px !important;
        padding: 22px 26px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15) !important;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
    }

    .hero-title {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #F0F3EF !important;
        margin-bottom: 6px !important;
        letter-spacing: -0.3px !important;
    }

    .hero-subtitle {
        font-size: 14px !important;
        color: #B5C0B7 !important;
        margin: 0 !important;
        line-height: 1.5 !important;
    }

    /* --- CARDS & PANELS --- */
    .rw-card {
        background-color: #34423B !important;
        border: 1px solid #4B5A51 !important;
        border-radius: 12px !important;
        padding: 20px 22px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;
    }

    .rw-card-elevated {
        background-color: #3D4C44 !important;
        border: 1px solid #4B5A51 !important;
        border-radius: 12px !important;
        padding: 20px 22px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.16) !important;
    }

    .rw-card-header {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #A0B7A3 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
        margin-bottom: 16px !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* --- UPLOAD PANEL & STREAMLIT FILE UPLOADER --- */
    .upload-frame {
        border: 2px dashed #4B5A51;
        border-radius: 10px;
        padding: 26px 20px;
        text-align: center;
        background-color: #3D4C44;
        transition: border-color 0.2s ease-in-out;
    }

    .upload-frame:hover {
        border-color: #A0B7A3;
    }

    [data-testid="stFileUploader"] {
        background-color: #3D4C44 !important;
        border: 2px dashed #4B5A51 !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        margin-bottom: 16px !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #78977F !important;
        background-color: rgba(61, 76, 68, 0.9) !important;
    }

    section[data-testid="stFileUploaderDropzone"] {
        background-color: transparent !important;
        border: none !important;
        padding: 6px !important;
    }

    section[data-testid="stFileUploaderDropzone"] button {
        background-color: #78977F !important;
        color: #F0F3EF !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 8px 18px !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease !important;
    }

    section[data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #63826A !important;
    }

    section[data-testid="stFileUploaderDropzone"] span {
        color: #B5C0B7 !important;
    }

    section[data-testid="stFileUploaderDropzone"] small {
        color: #8F9B92 !important;
    }

    /* --- METRIC SUMMARY CARDS --- */
    .metric-card {
        background-color: #34423B !important;
        border: 1px solid #4B5A51 !important;
        border-radius: 10px !important;
        padding: 16px 18px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1) !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }

    .metric-card-label {
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.7px !important;
        color: #8F9B92 !important;
        margin-bottom: 6px !important;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .metric-card-value {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #F0F3EF !important;
        line-height: 1.15 !important;
        letter-spacing: -0.5px !important;
    }

    .metric-card-sub {
        font-size: 12px !important;
        color: #B5C0B7 !important;
        margin-top: 6px !important;
    }

    /* --- PRIMARY BUTTONS --- */
    .stButton > button {
        background-color: #78977F !important;
        color: #F0F3EF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 11px 24px !important;
        letter-spacing: 0.3px !important;
        transition: background-color 0.15s ease-in-out, box-shadow 0.15s ease !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.16) !important;
    }

    .stButton > button:hover {
        background-color: #637E69 !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.22) !important;
    }

    /* --- DOWNLOAD BUTTONS --- */
    .stDownloadButton > button {
        background-color: #34423B !important;
        color: #A0B7A3 !important;
        border: 1.5px solid #78977F !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 9px 20px !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stDownloadButton > button:hover {
        background-color: #3D4C44 !important;
        color: #F0F3EF !important;
        border-color: #A0B7A3 !important;
    }

    /* --- SLIDER STYLING --- */
    .stSlider label {
        font-weight: 600 !important;
        color: #F0F3EF !important;
        font-size: 13px !important;
    }

    .stSlider [data-baseweb="slider"] {
        margin-top: 8px !important;
    }

    /* --- DATA TABLES --- */
    .dataframe {
        border: 1px solid #4B5A51 !important;
        border-radius: 8px !important;
        background-color: #34423B !important;
        color: #F0F3EF !important;
    }

    .dataframe th {
        background-color: #3D4C44 !important;
        color: #A0B7A3 !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #4B5A51 !important;
    }

    .dataframe td {
        color: #F0F3EF !important;
        border-bottom: 1px solid #4B5A51 !important;
    }

    /* Notice / Disclaimer container */
    .academic-disclaimer {
        background-color: #34423B;
        border: 1px solid #4B5A51;
        border-left: 4px solid #C2A36D;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 13px;
        color: #B5C0B7;
        line-height: 1.55;
        margin-top: 24px;
    }

    /* Status indicator dots */
    .dot-ready {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #78A987;
        display: inline-block;
    }

    .dot-alert {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #D2A35D;
        display: inline-block;
    }
    </style>
    """


# -----------------------------------------------------------------------------
# 3. HTML RENDERERS FOR SIDEBAR, HERO, AND METRIC CARDS
# -----------------------------------------------------------------------------
def render_sidebar_brand(model_loaded: bool, device_name: str) -> str:
    """Renders the top branding and status indicator in the sidebar without emojis."""
    status_dot = '<span class="dot-ready"></span>' if model_loaded else '<span class="dot-alert"></span>'
    status_text = "YOLOv8n Active" if model_loaded else "Model Unloaded"
    road_icon = get_icon("Road", size=24, color="#78977F")

    return f"""
    <div class="sidebar-brand-card">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: rgba(120, 151, 127, 0.16); padding: 8px; border-radius: 8px; border: 1px solid #4B5A51; display: flex;">
                {road_icon}
            </div>
            <div>
                <div style="font-size: 18px; font-weight: 700; color: #F0F3EF; letter-spacing: -0.3px;">RoadWatch AI</div>
                <div style="font-size: 11px; color: #8F9B92; text-transform: uppercase; letter-spacing: 0.5px;">Infrastructure Vision</div>
            </div>
        </div>
        <div class="sidebar-badge">
            {status_dot}
            <span>{status_text}</span>
            <span style="color: #4B5A51;">|</span>
            <span style="color: #A0B7A3;">{device_name}</span>
        </div>
    </div>
    """


def render_sidebar_footer(device_name: str) -> str:
    """Renders the lower sidebar metadata box without emojis."""
    cpu_icon = get_icon("Gauge", size=14, color="#8F9B92")
    db_icon = get_icon("Database", size=14, color="#8F9B92")

    return f"""
    <div class="sidebar-footer-card">
        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
            {cpu_icon}
            <span>Compute: <strong style="color: #C0D0C1;">{device_name}</strong></span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 6px;">
            {db_icon}
            <span>Dataset: <strong style="color: #C0D0C1;">RDD2022 India</strong></span>
        </div>
        <div style="font-size: 11px; color: #8F9B92; margin-top: 8px;">
            Academic Prototype v2.0
        </div>
    </div>
    """


def render_hero_banner(title: str = "Road Intelligence Dashboard", subtitle: str = "Advanced computer vision for road surface damage detection.", icon_name: str = "ScanSearch") -> str:
    """Renders the medium-dark hero section banner with a Lucide icon."""
    icon_svg = get_icon(icon_name, size=26, color="#A0B7A3")
    return f"""
    <div class="hero-panel">
        <div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        <div style="background: rgba(38, 51, 46, 0.4); border: 1px solid #4B5A51; padding: 10px; border-radius: 10px; display: flex; align-items: center;">
            {icon_svg}
        </div>
    </div>
    """


def render_metric_card(label: str, value: str, subtext: str = "", icon_name: str = "Activity") -> str:
    """Renders an elevated metric summary card with a monochromatic Lucide icon."""
    icon_svg = get_icon(icon_name, size=15, color="#8F9B92")
    return f"""
    <div class="metric-card">
        <div>
            <div class="metric-card-label">
                {icon_svg}
                <span>{label}</span>
            </div>
            <div class="metric-card-value">{value}</div>
        </div>
        {f'<div class="metric-card-sub">{subtext}</div>' if subtext else ''}
    </div>
    """


def format_detection_table(boxes: List[List[int]], confidences: List[float], classes: List[str]) -> pd.DataFrame:
    """Formats detected bounding box coordinates into a clean DataFrame."""
    rows = []
    for i, (b, c, cls_name) in enumerate(zip(boxes, confidences, classes), 1):
        rows.append({
            "Detection #": i,
            "Class": cls_name,
            "Confidence": f"{c * 100:.1f}%",
            "Bounding Box [xmin, ymin, xmax, ymax]": f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]",
            "Box Width (px)": b[2] - b[0],
            "Box Height (px)": b[3] - b[1],
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Detection #", "Class", "Confidence", "Bounding Box [xmin, ymin, xmax, ymax]", "Box Width (px)", "Box Height (px)"
        ])

    return pd.DataFrame(rows)


def load_evaluation_metrics(metrics_path: str = "outputs/metrics/evaluation_metrics.json") -> Optional[Dict[str, Any]]:
    """Loads empirical evaluation metrics from disk if present."""
    p = Path(metrics_path)
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

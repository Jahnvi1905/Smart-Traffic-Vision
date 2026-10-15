"""Smart Traffic Vision - AI-Powered Traffic Monitoring & Analytics.

Streamlit Web Application.
Provides real-time vehicle detection, ByteTrack tracking, automated traffic counting,
license plate OCR, and safety violation auditing (Helmet, Wrong-Way, Virtual Stop-Line).
"""

from datetime import datetime
import io
import os
from pathlib import Path
import sys
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from counting.vehicle_counter import VehicleCounter
from database.database_manager import DatabaseManager
from detection.vehicle_detector import VehicleDetector
from ocr.plate_detector import PlateDetector
from ocr.plate_ocr import PlateOCR
from violations.violation_manager import TrafficViolationManager

# ── Paths ──────────────────────────────────────────────────────────────────────
VEHICLE_MODEL_PATH = PROJECT_ROOT / "models" / "vehicle_detector.pt"
BASE_YOLO_PATH     = PROJECT_ROOT / "yolo11n.pt"
PLATE_MODEL_PATH   = PROJECT_ROOT / "models" / "plate_detector.pt"
HELMET_MODEL_PATH  = PROJECT_ROOT / "models" / "helmet_detector.pt"
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data" / "input" / "sample.mp4"
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "input" / "sample.jpg"
DEFAULT_PLATE_IMG  = PROJECT_ROOT / "data" / "input" / "plate_test.jpg"
DB_PATH            = PROJECT_ROOT / "data" / "traffic_violations.db"
OUTPUT_DIR         = PROJECT_ROOT / "data" / "output"
VIOLATIONS_DIR     = PROJECT_ROOT / "data" / "output" / "violations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIOLATIONS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXT = {".jpg", ".jpeg", ".png"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Traffic Vision",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Defaults ─────────────────────────────────────────────────────
SESSION_DEFAULTS = {
    "page": "Home",
    "dark": False,
    "last_counts": {},
    "total_counted": 0,
    "image_analysis_result": None,
    "video_analysis_result": None,
    "plate_scan_results": None,
    "plate_scan_annotated": None,
    "uploaded_image_cache": None,
    "uploaded_image_name": None,
    "uploaded_video_cache": None,
    "uploaded_video_name": None,
}

for k, v in SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

dark = st.session_state.dark

# ── Palette & Design Tokens ────────────────────────────────────────────────────
if dark:
    BG           = "#0B1120"   # Deep Navy background
    SURFACE      = "#1E293B"   # Slate-800 card surface
    SURFACE_HOV  = "#283548"   # Slate-700 hover
    BORDER       = "#334155"   # Slate-700 border
    BORDER_LIGHT = "#475569"   # Slate-600 border
    TEXT         = "#F8FAFC"   # White text
    MUTED        = "#94A3B8"   # Cool Gray muted text
    ACCENT       = "#3B82F6"   # Primary Blue
    ACCENT_HOVER = "#2563EB"
    SBARBG       = "#070D18"   # Sidebar Dark Navy
    SBARBDR      = "#1E293B"
    CARD_SH      = "0 2px 8px rgba(0, 0, 0, 0.4)"
    TAG_OK_BG    = "#064E3B"; TAG_OK_FG = "#6EE7B7"; TAG_OK_BD = "#059669"
    TAG_WARN_BG  = "#451A03"; TAG_WARN_FG = "#FCD34D"; TAG_WARN_BD = "#D97706"
    TAG_DANG_BG  = "#450A0A"; TAG_DANG_FG = "#FCA5A5"; TAG_DANG_BD = "#DC2626"
    TAG_INFO_BG  = "#1E3A5F"; TAG_INFO_FG = "#93C5FD"; TAG_INFO_BD = "#3B82F6"
    PLATE_BG     = "#0F172A"; PLATE_BORDER = "#F59E0B"; PLATE_FG = "#F8FAFC"
else:
    BG           = "#F8FAFC"   # Clean Light Slate
    SURFACE      = "#FFFFFF"   # White surface
    SURFACE_HOV  = "#F1F5F9"   # Slate-100 hover
    BORDER       = "#E2E8F0"   # Slate-200 border
    BORDER_LIGHT = "#CBD5E1"   # Slate-300 border
    TEXT         = "#0F172A"   # Deep Slate text
    MUTED        = "#475569"   # Slate-600 legible secondary text
    ACCENT       = "#2563EB"   # Royal Blue
    ACCENT_HOVER = "#1D4ED8"
    SBARBG       = "#FFFFFF"   # Sidebar White
    SBARBDR      = "#E2E8F0"
    CARD_SH      = "0 1px 3px rgba(15, 23, 42, 0.08)"
    TAG_OK_BG    = "#DCFCE7"; TAG_OK_FG = "#15803D"; TAG_OK_BD = "#86EFAC"
    TAG_WARN_BG  = "#FEF9C3"; TAG_WARN_FG = "#92400E"; TAG_WARN_BD = "#FDE68A"
    TAG_DANG_BG  = "#FEE2E2"; TAG_DANG_FG = "#991B1B"; TAG_DANG_BD = "#FCA5A5"
    TAG_INFO_BG  = "#EFF6FF"; TAG_INFO_FG = "#1D4ED8"; TAG_INFO_BD = "#BFDBFE"
    PLATE_BG     = "#FFFBEB"; PLATE_BORDER = "#F59E0B"; PLATE_FG = "#0F172A"

# ── Inject Robust Theme CSS ────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
    --background-color: {BG};
    --secondary-background-color: {SURFACE};
    --text-color: {TEXT};
    --primary-color: {ACCENT};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}}

html, body, .stApp {{
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-family: 'Inter', sans-serif !important;
}}

/* Header & Controls */
header[data-testid="stHeader"] {{
    background: transparent !important;
    z-index: 99 !important;
}}
#MainMenu, footer {{ visibility: hidden !important; }}
.stDeployButton {{ display: none !important; }}

/* Ensure sidebar toggle button is ALWAYS visible and styled */
[data-testid="stSidebarCollapsedControl"],
button[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{
    display: flex !important;
    visibility: visible !important;
    position: fixed !important;
    top: 10px !important;
    left: 10px !important;
    z-index: 100000 !important;
    background: {SURFACE} !important;
    border: 1px solid {BORDER_LIGHT} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
    box-shadow: {CARD_SH} !important;
}}

/* Main block padding */
.block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1220px;
}}

/* Force sidebar visibility on normal laptop screens */
@media (min-width: 900px) {{
    section[data-testid="stSidebar"] {{
        transform: none !important;
        margin-left: 0 !important;
        display: block !important;
        visibility: visible !important;
        min-width: 250px !important;
        max-width: 275px !important;
    }}
}}

section[data-testid="stSidebar"] {{
    background-color: {SBARBG} !important;
    border-right: 1px solid {SBARBDR} !important;
    padding-top: 0.8rem !important;
}}
section[data-testid="stSidebar"] * {{
    color: {TEXT};
}}

/* Typography */
h1, h2, h3, h4, p, span, label, div {{
    color: {TEXT};
}}
.stMarkdown p {{
    color: {TEXT} !important;
    line-height: 1.5;
}}
.stCaption, caption {{
    color: {MUTED} !important;
}}

/* Prevent blue selection highlight on radio labels */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] span,
label, p, span {{
    user-select: none;
    -webkit-user-select: none;
}}
div[data-testid="stRadio"] label:focus,
div[data-testid="stRadio"] label:active {{
    outline: none !important;
    background-color: transparent !important;
}}

/* Radio Buttons Styling */
div[data-testid="stRadio"] > div[role="radiogroup"] {{
    gap: 10px !important;
}}
div[data-testid="stRadio"] label {{
    background: transparent !important;
    padding: 3px 6px !important;
    border-radius: 6px !important;
}}
div[data-testid="stRadio"] label:hover {{
    background: {SURFACE_HOV} !important;
}}
div[data-testid="stRadio"] label span {{
    color: {TEXT} !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
}}
div[data-testid="stRadio"] div[role="radio"] {{
    border-color: {BORDER_LIGHT} !important;
    background-color: {SURFACE} !important;
}}
div[data-testid="stRadio"] div[role="radio"][aria-checked="true"] {{
    border-color: {ACCENT} !important;
    background-color: {ACCENT} !important;
}}

/* Header Banner */
.page-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 14px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 18px;
}}
.page-title {{
    font-size: 21px;
    font-weight: 700;
    color: {TEXT};
    margin: 0;
    letter-spacing: -0.01em;
}}
.page-subtitle {{
    font-size: 13px;
    color: {MUTED};
    margin: 3px 0 0 0;
}}
.live-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11.5px;
    font-weight: 600;
    color: {TAG_OK_FG};
    background: {TAG_OK_BG};
    border: 1px solid {TAG_OK_BD};
    padding: 3px 10px;
    border-radius: 14px;
}}

/* SaaS Cards */
.saas-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: {CARD_SH};
}}
.saas-card-title {{
    font-size: 14px;
    font-weight: 600;
    color: {TEXT};
    margin-bottom: 10px;
}}

/* Metrics Grid */
.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin-bottom: 18px;
}}
.metric-box {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: {CARD_SH};
}}
.metric-box-label {{
    font-size: 11px;
    font-weight: 600;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}}
.metric-box-value {{
    font-size: 24px;
    font-weight: 700;
    color: {TEXT};
    line-height: 1.1;
}}
.metric-box-desc {{
    font-size: 11px;
    color: {MUTED};
    margin-top: 3px;
}}

/* Status Badges */
.badge {{
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}}
.badge-ok   {{ background: {TAG_OK_BG};   color: {TAG_OK_FG};   border: 1px solid {TAG_OK_BD}; }}
.badge-warn {{ background: {TAG_WARN_BG}; color: {TAG_WARN_FG}; border: 1px solid {TAG_WARN_BD}; }}
.badge-dang {{ background: {TAG_DANG_BG}; color: {TAG_DANG_FG}; border: 1px solid {TAG_DANG_BD}; }}
.badge-info {{ background: {TAG_INFO_BG}; color: {TAG_INFO_FG}; border: 1px solid {TAG_INFO_BD}; }}

/* License Plate Display */
.license-plate {{
    font-family: 'Courier New', monospace;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 2px;
    color: {PLATE_FG};
    background: {PLATE_BG};
    border: 2px solid {PLATE_BORDER};
    border-radius: 6px;
    padding: 6px 16px;
    display: inline-block;
}}

/* Empty State */
.empty-box {{
    text-align: center;
    padding: 32px 20px;
    background: {SURFACE};
    border: 1px dashed {BORDER_LIGHT};
    border-radius: 10px;
    color: {MUTED};
    margin: 12px 0;
}}
.empty-icon {{ font-size: 28px; margin-bottom: 6px; }}
.empty-title {{ font-size: 14.5px; font-weight: 600; color: {TEXT}; margin-bottom: 4px; }}
.empty-desc {{ font-size: 12px; color: {MUTED}; max-width: 440px; margin: 0 auto; line-height: 1.5; }}

/* Buttons */
.stButton > button {{
    background-color: {SURFACE} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER_LIGHT} !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 6px 14px !important;
}}
.stButton > button:hover {{
    background-color: {SURFACE_HOV} !important;
    border-color: {ACCENT} !important;
}}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {{
    background-color: {ACCENT} !important;
    border-color: {ACCENT} !important;
    color: #FFFFFF !important;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {{
    background-color: {ACCENT_HOVER} !important;
    border-color: {ACCENT_HOVER} !important;
}}

/* Download Button */
.stDownloadButton > button {{
    background-color: {SURFACE} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER_LIGHT} !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
}}

/* Expanders */
details[data-testid="stExpander"] {{
    background-color: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    margin-bottom: 10px !important;
}}
details[data-testid="stExpander"] summary {{
    background-color: {SURFACE} !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
}}
details[data-testid="stExpander"] summary span {{
    color: {TEXT} !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}}

/* File Uploader */
div[data-testid="stFileUploaderDropzone"] {{
    background: {SURFACE} !important;
    border: 2px dashed {BORDER_LIGHT} !important;
    border-radius: 8px !important;
    padding: 16px !important;
}}
div[data-testid="stFileUploaderDropzone"] span,
div[data-testid="stFileUploaderDropzone"] small {{
    color: {TEXT} !important;
}}
div[data-testid="stFileUploaderDropzone"] button {{
    background: {SURFACE} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER_LIGHT} !important;
}}

/* Selectbox & Inputs */
div[data-baseweb="select"] > div {{
    background-color: {SURFACE} !important;
    color: {TEXT} !important;
    border-color: {BORDER_LIGHT} !important;
}}
div[data-baseweb="select"] * {{ color: {TEXT} !important; }}
div[data-testid="stTextInput"] input {{
    background-color: {SURFACE} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER_LIGHT} !important;
}}

/* Progress & Alert */
.stProgress > div > div {{ background-color: {ACCENT} !important; }}
div[data-testid="stAlert"] {{ border-radius: 8px !important; }}
hr {{ border-color: {BORDER} !important; margin: 14px 0 !important; }}
</style>
""", unsafe_allow_html=True)


# ── Model & Database Loaders ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_database_manager():
    return DatabaseManager(db_path=str(DB_PATH))


@st.cache_resource(show_spinner=False)
def load_vehicle_detector():
    model_path = str(VEHICLE_MODEL_PATH) if VEHICLE_MODEL_PATH.exists() else str(BASE_YOLO_PATH)
    det = VehicleDetector(model_path=model_path, conf_threshold=0.35)
    det.load_model()
    return det


@st.cache_resource(show_spinner=False)
def load_plate_detector():
    if not PLATE_MODEL_PATH.exists():
        return None
    det = PlateDetector(model_path=str(PLATE_MODEL_PATH), conf_threshold=0.25)
    det.load_model()
    return det


@st.cache_resource(show_spinner=False)
def load_plate_ocr():
    ocr = PlateOCR(gpu=False)
    ocr.load_model()
    return ocr


db = get_database_manager()
vehicle_ok = VEHICLE_MODEL_PATH.exists()
plate_ok   = PLATE_MODEL_PATH.exists()
helmet_ok  = HELMET_MODEL_PATH.exists()

NAV_ITEMS = [
    ("Home",                 "🏠"),
    ("Analyze Traffic",      "📹"),
    ("Number Plate Scanner", "🔍"),
    ("Records & Reports",    "📋"),
]


def format_event_display(e: dict) -> dict:
    """
    Maps raw SQLite database event rows to clean, truthful user-facing fields.
    Ensures:
    - No vehicle counts under Number Plate
    - No event names or count counts under Vehicle Type
    - Confidence is shown only when genuinely representing model confidence
    - Dictionaries/JSON are parsed into readable summaries
    - Evidence filenames are displayed with friendly labels
    """
    raw_type = str(e.get("event_type") or "").strip()
    raw_vclass = str(e.get("vehicle_class") or "").strip()
    raw_plate = str(e.get("plate_text") or "").strip()
    raw_conf = e.get("confidence")
    raw_evidence = str(e.get("evidence_path") or "").strip()

    # 1. Human-Readable Event Type
    if raw_type == "PLATE_READ":
        ev_name = "Number Plate Scan"
    elif raw_type == "COUNT":
        ev_name = "Traffic Flow Count"
    elif raw_type == "NO_HELMET":
        ev_name = "Helmet Violation"
    elif raw_type == "WRONG_WAY":
        ev_name = "Wrong-Way Movement"
    elif raw_type == "RED_LIGHT":
        ev_name = "Stop-Line Violation (Virtual Policy)"
    else:
        ev_name = raw_type.replace("_", " ").title()

    # 2. Field Mappings per Event Type
    if raw_type == "COUNT":
        # Road counting events have no vehicle license plate
        plate_display = "—"
        vehicle_type_display = "Traffic Stream (All Classes)"
        confidence_display = "—"  # Stored float was threshold setting, not model confidence

        # Parse count breakdown dictionary into readable text
        summary_items = []
        if raw_plate.startswith("{") and raw_plate.endswith("}"):
            try:
                import ast
                cnts = ast.literal_eval(raw_plate)
                if isinstance(cnts, dict):
                    for c_name, c_cnt in cnts.items():
                        if c_cnt > 0:
                            clean_name = c_name.replace("-", " ").title()
                            summary_items.append(f"{c_cnt} {clean_name}{'s' if c_cnt > 1 else ''}")
            except Exception:
                pass

        if summary_items:
            flow_summary = ", ".join(summary_items)
        elif "Total: 0" in raw_vclass or raw_plate == "{}":
            flow_summary = "0 vehicles crossed line"
        elif raw_vclass:
            flow_summary = raw_vclass
        else:
            flow_summary = "No vehicle crossings registered"

        details_display = flow_summary

    elif raw_type == "PLATE_READ":
        vehicle_type_display = raw_vclass if (raw_vclass and raw_vclass != "None") else "Vehicle"
        plate_display = raw_plate if (raw_plate and raw_plate not in ("UNKNOWN", "None")) else "—"
        confidence_display = f"{float(raw_conf)*100:.0f}%" if (raw_conf is not None and float(raw_conf) > 0) else "—"
        details_display = f"Plate Text: {plate_display}"

    elif raw_type == "NO_HELMET":
        vehicle_type_display = raw_vclass.title() if (raw_vclass and raw_vclass != "None") else "Two-Wheeler / Motorcycle"
        plate_display = raw_plate if (raw_plate and raw_plate not in ("UNKNOWN", "None", "")) else "Not Identified"
        confidence_display = f"{float(raw_conf)*100:.0f}%" if (raw_conf is not None and float(raw_conf) > 0) else "—"
        details_display = "Motorcycle rider detected without protective helmet"

    elif raw_type == "WRONG_WAY":
        vehicle_type_display = raw_vclass.title() if (raw_vclass and raw_vclass != "None") else "Vehicle"
        plate_display = raw_plate if (raw_plate and raw_plate not in ("UNKNOWN", "None", "")) else "Not Identified"
        confidence_display = f"{float(raw_conf)*100:.0f}%" if (raw_conf is not None and float(raw_conf) > 0) else "—"
        details_display = "Vehicle moving against designated lane direction"

    elif raw_type == "RED_LIGHT":
        vehicle_type_display = raw_vclass.title() if (raw_vclass and raw_vclass != "None") else "Vehicle"
        plate_display = raw_plate if (raw_plate and raw_plate not in ("UNKNOWN", "None", "")) else "Not Identified"
        confidence_display = "Policy Trigger"
        details_display = "Vehicle crossed virtual stop-line during policy RED signal state"

    else:
        vehicle_type_display = raw_vclass if raw_vclass else "—"
        plate_display = raw_plate if raw_plate else "—"
        confidence_display = f"{float(raw_conf)*100:.0f}%" if (raw_conf is not None and float(raw_conf) > 0) else "—"
        details_display = f"{raw_vclass} {raw_plate}".strip() or "—"

    # Friendly evidence filename/description
    if raw_evidence and os.path.exists(raw_evidence):
        fname = Path(raw_evidence).name
        if fname.endswith(".mp4"):
            ev_label = f"Processed Video ({fname})"
        elif "helmet" in fname.lower():
            ev_label = f"Helmet Violation Crop ({fname})"
        elif "wrongway" in fname.lower():
            ev_label = f"Wrong-Way Snapshot ({fname})"
        elif "plate" in fname.lower():
            ev_label = f"Plate Snapshot ({fname})"
        else:
            ev_label = f"Evidence File ({fname})"
    elif raw_evidence:
        ev_label = f"Media Ref ({Path(raw_evidence).name})"
    else:
        ev_label = "No Evidence File Attached"

    return {
        "id": e.get("id"),
        "timestamp": e.get("timestamp", "--"),
        "event_type": ev_name,
        "raw_type": raw_type,
        "vehicle_type": vehicle_type_display,
        "plate_number": plate_display,
        "confidence": confidence_display,
        "review_status": (e.get("review_status") or "pending").title(),
        "details": details_display,
        "evidence_path": raw_evidence,
        "evidence_label": ev_label,
    }



# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 4px 4px 12px 4px;">
        <div style="font-size: 16px; font-weight: 700; color: {TEXT}; display: flex; align-items: center; gap: 8px;">
            <span>🚦</span> Smart Traffic Vision
        </div>
        <div style="font-size: 11px; color: {MUTED}; margin-top: 1px;">
            Traffic Intelligence &amp; Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Theme Switcher (Reliable callback toggle)
    st.markdown(
        f"<div style='font-size:11px; font-weight:600; color:{MUTED}; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;'>Theme Mode</div>",
        unsafe_allow_html=True,
    )

    def on_theme_toggle():
        st.session_state.dark = (st.session_state.theme_switch_control == "🌙 Dark Mode")

    st.radio(
        "Theme Mode Selection",
        options=["☀️ Light Mode", "🌙 Dark Mode"],
        index=1 if dark else 0,
        horizontal=True,
        label_visibility="collapsed",
        key="theme_switch_control",
        on_change=on_theme_toggle,
    )

    st.markdown(f"<hr style='border-color:{BORDER}; margin: 10px 0 12px 0;'/>", unsafe_allow_html=True)

    # 4 Destination Navigation Buttons
    st.markdown(
        f"<div style='font-size:11px; font-weight:600; color:{MUTED}; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;'>Navigation</div>",
        unsafe_allow_html=True,
    )

    current_page = st.session_state.page
    for label, icon in NAV_ITEMS:
        is_active = (current_page == label)
        if st.button(
            f"{icon}  {label}",
            key=f"nav_link_{label}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = label
            st.rerun()

    page = st.session_state.page

    st.markdown(f"<hr style='border-color:{BORDER}; margin: 14px 0 10px 0;'/>", unsafe_allow_html=True)

    # Collapsible System Diagnostics
    with st.expander("System Status & Diagnostics", expanded=False):
        st.caption(f"• Vehicle Model: {'Active (6-Class)' if vehicle_ok else 'Base COCO'}")
        st.caption(f"• Plate Model: {'Active' if plate_ok else 'Missing'}")
        st.caption(f"• Helmet Model: {'Active' if helmet_ok else 'Missing'}")
        st.caption(f"• EasyOCR Engine: Active")
        st.caption(f"• Device: {'GPU (CUDA)' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'CPU (Safe)'}")
        st.caption(f"• Python: {sys.version.split()[0]}")

    st.caption("Smart Traffic Vision · v2.5")


# ── Reusable Header ────────────────────────────────────────────────────────────
def render_header(title: str, subtitle: str):
    st.markdown(f"""
    <div class="page-header">
        <div>
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        <div class="live-pill">&#x25CF; System Active</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. HOME — SIMPLE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    render_header(
        "Traffic Monitoring Dashboard",
        "Automated vehicle detection, traffic flow counting, and license plate recognition.",
    )

    all_events = db.get_events(limit=1000)
    plate_events = [e for e in all_events if e.get("event_type") == "PLATE_READ"]
    violation_events = [
        e for e in all_events
        if e.get("event_type") in ("NO_HELMET", "WRONG_WAY", "RED_LIGHT")
    ]
    total_db = len(all_events)
    session_counted = st.session_state.total_counted

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-box">
            <div class="metric-box-label">Vehicles Counted</div>
            <div class="metric-box-value">{session_counted}</div>
            <div class="metric-box-desc">Current session road crossings</div>
        </div>
        <div class="metric-box">
            <div class="metric-box-label">Plates Read</div>
            <div class="metric-box-value">{len(plate_events)}</div>
            <div class="metric-box-desc">Recognized license plates</div>
        </div>
        <div class="metric-box">
            <div class="metric-box-label">Recorded Events</div>
            <div class="metric-box-value">{total_db}</div>
            <div class="metric-box-desc">Total entries in database</div>
        </div>
        <div class="metric-box">
            <div class="metric-box-label">Safety Violations</div>
            <div class="metric-box-value">{len(violation_events)}</div>
            <div class="metric-box-desc">Helmet, wrong-way &amp; stop-line</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_main, col_side = st.columns([1.5, 1], gap="medium")

    with col_main:
        if st.session_state.last_counts and sum(st.session_state.last_counts.values()) > 0:
            st.markdown(f"""
            <div class="saas-card">
                <div class="saas-card-title">Latest Traffic Breakdown by Vehicle Type</div>
            """, unsafe_allow_html=True)
            df_counts = (
                pd.DataFrame(list(st.session_state.last_counts.items()), columns=["Vehicle Class", "Count"])
                .sort_values("Count", ascending=False)
            )
            st.bar_chart(df_counts.set_index("Vehicle Class"), color=ACCENT, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        elif total_db > 0:
            st.markdown(f"""
            <div class="saas-card">
                <div class="saas-card-title">Recent Activity Log</div>
            """, unsafe_allow_html=True)
            recent_rows = []
            for ev in all_events[:6]:
                fev = format_event_display(ev)
                recent_rows.append({
                    "Date & Time": fev["timestamp"],
                    "Event": fev["event_type"],
                    "Vehicle Type": fev["vehicle_type"],
                    "Summary / Plate": fev["plate_number"] if fev["plate_number"] != "—" else fev["details"],
                    "Status": fev["review_status"],
                })
            st.dataframe(pd.DataFrame(recent_rows), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="empty-box">
                <div class="empty-icon">🚦</div>
                <div class="empty-title">Ready for Analysis</div>
                <div class="empty-desc">
                    No traffic records logged yet. Go to <b>Analyze Traffic</b> to upload video or photos, or <b>Number Plate Scanner</b> to read plates.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_side:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-card-title">Quick Actions</div>
            <div style="font-size: 13px; color: {MUTED}; margin-bottom: 12px;">
                Choose an action to start inspecting traffic media:
            </div>
        """, unsafe_allow_html=True)

        qa1, qa2 = st.columns(2)
        with qa1:
            if st.button("📹 Analyze Traffic", key="home_qa_analyze", use_container_width=True):
                st.session_state.page = "Analyze Traffic"
                st.rerun()
        with qa2:
            if st.button("🔍 Scan Plate", key="home_qa_scan", use_container_width=True):
                st.session_state.page = "Number Plate Scanner"
                st.rerun()

        st.markdown(f"""
        <hr style="border-color:{BORDER}; margin: 14px 0 10px 0;"/>
        <div class="saas-card-title" style="margin-bottom: 8px;">System Capabilities</div>
        <div style="display:flex; flex-direction:column; gap:7px; font-size:12.5px;">
            <div><span class="badge badge-ok">Active</span> 6-Class Vehicle Detection (YOLO11)</div>
            <div><span class="badge badge-ok">Active</span> Multi-Vehicle Tracking &amp; Road Line Counting</div>
            <div><span class="badge badge-ok">Active</span> Automatic Number Plate OCR</div>
            <div><span class="badge badge-ok">Active</span> Helmet Compliance &amp; Wrong-Way Audit</div>
            <div><span class="badge badge-info">Policy</span> Virtual Stop-Line Rule (Configurable Policy)</div>
        </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. ANALYZE TRAFFIC — COMPACT, INTUITIVE WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analyze Traffic":
    render_header(
        "Analyze Traffic",
        "Upload a photo or video to detect vehicles and view the results.",
    )

    # Balanced 2-column layout so controls and preview are visible together without scrolling
    col_input, col_view = st.columns([1.1, 1], gap="large")

    input_path = None
    input_is_img = False
    preview_img = None
    video_dims = (1920, 1080)
    selected_name = "None"
    file_info_text = ""

    with col_input:
        st.markdown(f"<div style='font-size:13px; font-weight:600; color:{TEXT}; margin-bottom:4px;'>Analysis Choice</div>", unsafe_allow_html=True)
        media_type = st.radio(
            "Analysis Mode Selection",
            options=["📷 Photo Analysis", "🎥 Video Analysis"],
            horizontal=True,
            label_visibility="collapsed",
            key="at_media_mode",
        )

        # ── Photo Mode ──
        if "Photo" in media_type:
            input_is_img = True
            p_src = st.radio(
                "Photo Source",
                options=["Upload your photo", "Try sample photo"],
                horizontal=True,
                key="at_photo_source",
            )

            if p_src == "Try sample photo":
                if DEFAULT_IMAGE_PATH.exists():
                    input_path = str(DEFAULT_IMAGE_PATH)
                    preview_img = cv2.imread(input_path)
                    h, w = preview_img.shape[:2]
                    selected_name = DEFAULT_IMAGE_PATH.name
                    file_info_text = f"Sample Photo · {w}×{h}px"
                else:
                    st.warning("Sample image not found in `data/input/sample.jpg`.")
            else:
                up_img = st.file_uploader(
                    "Choose an image (JPG, JPEG, PNG)",
                    type=["jpg", "jpeg", "png"],
                    key="at_uploader_img",
                )
                if up_img is not None:
                    val = up_img.getvalue()
                    data = np.asarray(bytearray(val), dtype=np.uint8)
                    preview_img = cv2.imdecode(data, cv2.IMREAD_COLOR)

                    if (
                        "uploaded_image_cache" not in st.session_state
                        or st.session_state.get("uploaded_image_name") != up_img.name
                    ):
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up_img.name).suffix.lower())
                        tmp.write(val)
                        tmp.close()
                        st.session_state.uploaded_image_cache = tmp.name
                        st.session_state.uploaded_image_name = up_img.name

                    input_path = st.session_state.uploaded_image_cache
                    h, w = preview_img.shape[:2]
                    selected_name = up_img.name
                    file_info_text = f"Uploaded Photo · {w}×{h}px · {len(val)/1024:.1f} KB"

        # ── Video Mode ──
        else:
            input_is_img = False
            v_src = st.radio(
                "Video Source",
                options=["Upload your video", "Try sample video"],
                horizontal=True,
                key="at_video_source",
            )

            first_frame = None
            if v_src == "Try sample video":
                if DEFAULT_VIDEO_PATH.exists():
                    input_path = str(DEFAULT_VIDEO_PATH)
                    cap = cv2.VideoCapture(input_path)
                    if cap.isOpened():
                        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        v_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        video_dims = (vw, vh)
                        ret, first_frame = cap.read()
                        cap.release()
                    else:
                        vw, vh, v_fps, v_frames = 1080, 1920, 30.0, 484

                    sz_mb = DEFAULT_VIDEO_PATH.stat().st_size / (1024 * 1024)
                    selected_name = DEFAULT_VIDEO_PATH.name
                    file_info_text = f"Sample Video · {vw}×{vh}px · {v_fps:.0f} FPS · {sz_mb:.1f} MB"
                else:
                    st.warning("Sample video not found in `data/input/sample.mp4`.")
            else:
                up_vid = st.file_uploader(
                    "Choose a video (MP4, AVI, MOV)",
                    type=["mp4", "avi", "mov", "mkv"],
                    key="at_uploader_vid",
                )
                if up_vid is not None:
                    val = up_vid.getvalue()
                    if (
                        "uploaded_video_cache" not in st.session_state
                        or st.session_state.get("uploaded_video_name") != up_vid.name
                    ):
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tmp.write(val)
                        tmp.close()
                        st.session_state.uploaded_video_cache = tmp.name
                        st.session_state.uploaded_video_name = up_vid.name

                    input_path = st.session_state.uploaded_video_cache
                    cap = cv2.VideoCapture(input_path)
                    if cap.isOpened():
                        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        v_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        video_dims = (vw, vh)
                        ret, first_frame = cap.read()
                        cap.release()
                    else:
                        vw, vh, v_fps, v_frames = 1280, 720, 30.0, 100

                    selected_name = up_vid.name
                    file_info_text = f"Uploaded Video · {vw}×{vh}px · {v_fps:.0f} FPS · {v_frames} frames"

        # Collapsible Settings
        with st.expander("⚙️ Optional Settings", expanded=False):
            sensitivity = st.selectbox(
                "Detection Sensitivity",
                options=["Standard (Balanced)", "High (More Detections)", "Strict (Fewer False Positives)"],
                index=0,
                help="Confidence threshold for detecting vehicles.",
            )
            conf_map = {
                "Standard (Balanced)": 0.35,
                "High (More Detections)": 0.25,
                "Strict (Fewer False Positives)": 0.50,
            }
            conf_thresh = conf_map[sensitivity]

            if not input_is_img:
                proc_length = st.selectbox(
                    "Video Processing Duration",
                    options=["Quick Preview (~2.5s / 75 frames)", "Short Segment (~5s / 150 frames)", "Extended Segment (~10s / 300 frames)"],
                    index=0,
                )
                frames_map = {
                    "Quick Preview (~2.5s / 75 frames)": 75,
                    "Short Segment (~5s / 150 frames)": 150,
                    "Extended Segment (~10s / 300 frames)": 300,
                }
                max_frames = frames_map[proc_length]

                enable_tracking = st.toggle("Vehicle Tracking", value=True)
                enable_counter = st.toggle("Virtual Counting Line", value=True)
                enable_helmet = st.toggle("Helmet Compliance Check", value=helmet_ok)
                enable_wrongway = st.toggle("Wrong-Way Traffic Check", value=False)
            else:
                enable_tracking = False
                enable_counter = False
                enable_helmet = False
                enable_wrongway = False
                max_frames = 1

        if "conf_thresh" not in locals():
            conf_thresh = 0.35
        if "max_frames" not in locals():
            max_frames = 75
        if "enable_tracking" not in locals():
            enable_tracking = True
        if "enable_counter" not in locals():
            enable_counter = True
        if "enable_helmet" not in locals():
            enable_helmet = helmet_ok
        if "enable_wrongway" not in locals():
            enable_wrongway = False

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        analyze_btn = st.button(
            "▶️  Analyze Now",
            type="primary",
            disabled=(input_path is None),
            use_container_width=True,
            key="btn_run_analysis",
        )

    # ── Right Column: Media Preview ──
    with col_view:
        st.markdown(f"""
        <div class="saas-card" style="min-height: 280px;">
            <div class="saas-card-title">Media Preview</div>
        """, unsafe_allow_html=True)

        if input_path is not None:
            st.markdown(f"""
            <div style="font-size:12.5px; margin-bottom:10px; color:{TEXT};">
                <b>{selected_name}</b> <span style="color:{MUTED}; margin-left:6px;">{file_info_text}</span>
            </div>
            """, unsafe_allow_html=True)

            if input_is_img and preview_img is not None:
                # Resize preview image neatly to prevent oversized vertical stretching
                ph, pw = preview_img.shape[:2]
                target_w = 460
                scale = target_w / float(pw)
                resized_preview = cv2.resize(preview_img, (target_w, int(ph * scale)))
                st.image(cv2.cvtColor(resized_preview, cv2.COLOR_BGR2RGB), use_container_width=False, width=420)

            elif not input_is_img and first_frame is not None:
                f_preview = first_frame.copy()
                py = int(video_dims[1] * 0.55)
                cv2.line(f_preview, (0, py), (video_dims[0], py), (0, 220, 255), 4)
                cv2.putText(f_preview, "Counting Line (55%)", (25, py - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
                f_w = 460
                f_scale = f_w / float(video_dims[0])
                resized_frame = cv2.resize(f_preview, (f_w, int(video_dims[1] * f_scale)))
                st.image(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), use_container_width=False, width=420)
        else:
            st.markdown(f"""
            <div class="empty-box" style="padding: 40px 10px; border:none;">
                <div class="empty-icon">📁</div>
                <div class="empty-title">No Media Selected</div>
                <div class="empty-desc">Choose a sample file or upload an image/video to view the preview.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Processing Execution ──
    if analyze_btn and input_path is not None:
        detector = load_vehicle_detector()
        detector.conf_threshold = conf_thresh
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        if input_is_img and preview_img is not None:
            with st.spinner("Analyzing photo with YOLO11…"):
                try:
                    results = detector.model(
                        preview_img,
                        conf=conf_thresh,
                        classes=detector.target_class_ids,
                        verbose=False,
                    )
                    boxes = results[0].boxes if results else []
                    annotated = detector.draw_detections(preview_img.copy(), boxes)
                except Exception as exc:
                    st.error(f"Image analysis error: {exc}")
                    boxes = []
                    annotated = preview_img.copy()

            records = []
            img_counts = {}
            for b in boxes:
                cid = int(b.cls[0].item())
                cname = detector.target_classes.get(cid, str(cid)).title()
                cconf = float(b.conf[0].item())
                records.append({"Vehicle Type": cname, "Confidence": f"{cconf*100:.1f}%"})
                img_counts[cname.lower()] = img_counts.get(cname.lower(), 0) + 1

            st.session_state.last_counts = img_counts
            st.session_state.total_counted = len(records)
            st.session_state.image_analysis_result = {
                "annotated": annotated,
                "records": records,
                "counts": img_counts,
                "total": len(records),
                "timestamp": ts,
            }

        elif not input_is_img:
            out_path = str(OUTPUT_DIR / f"processed_{ts}.mp4")
            vw, vh = video_dims
            calc_y = int(vh * 0.55)

            counter = None
            if enable_counter:
                counter = VehicleCounter(
                    line_start=(0, calc_y),
                    line_end=(vw, calc_y),
                    target_classes=detector.target_classes,
                )

            violation_mgr = None
            if enable_helmet or enable_wrongway:
                violation_mgr = TrafficViolationManager(
                    database_manager=db,
                    evidence_dir=str(VIOLATIONS_DIR),
                    helmet_model_path=str(HELMET_MODEL_PATH),
                    plate_model_path=str(PLATE_MODEL_PATH),
                    enable_helmet=enable_helmet,
                    enable_wrong_way=enable_wrongway,
                    enable_plate_assoc=True,
                    expected_direction="down",
                )

            prog = st.progress(0)
            status_text = st.empty()
            t0 = time.time()

            def _vid_progress(cur, tot, frm, _ctr):
                pct = int((cur / tot) * 100) if tot > 0 else 0
                prog.progress(min(pct, 100))
                status_text.caption(f"Processing frame {cur} of {tot} ({pct}%)")

            with st.spinner("Processing video frames and tracking vehicles…"):
                ok = detector.process_video(
                    input_path=input_path,
                    output_path=out_path,
                    enable_tracking=enable_tracking,
                    tracker="bytetrack.yaml",
                    counter=counter,
                    violation_manager=violation_mgr,
                    max_frames=max_frames,
                    frame_skip=2,
                    progress_callback=_vid_progress,
                )

            elapsed = time.time() - t0
            prog.progress(100)
            status_text.empty()

            if ok and os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                with open(out_path, "rb") as vf:
                    v_bytes = vf.read()

                total_counted = counter.total_count if counter else 0
                st.session_state.last_counts = counter.class_counts if counter else {}
                st.session_state.total_counted = total_counted

                if counter:
                    db.insert_event(
                        event_type="COUNT",
                        vehicle_class=f"Total: {total_counted}",
                        confidence=conf_thresh,
                        plate_text=str(dict(counter.class_counts)),
                        evidence_path=out_path,
                        review_status="approved",
                    )

                st.session_state.video_analysis_result = {
                    "out_path": out_path,
                    "video_bytes": v_bytes,
                    "total_counted": total_counted,
                    "class_counts": counter.class_counts if counter else {},
                    "elapsed": elapsed,
                    "max_frames": max_frames,
                    "timestamp": ts,
                    "violations_logged": len(violation_mgr.violations_logged) if violation_mgr else 0,
                }
            else:
                st.error("Video processing could not finalize. Try reducing the video processing duration.")

    # ── Display Analysis Results ──
    if input_is_img and st.session_state.get("image_analysis_result"):
        res = st.session_state.image_analysis_result
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size: 15px; font-weight: 700; color: {TEXT}; margin-bottom: 12px;">
            Results: {res['total']} Vehicles Detected
        </div>
        """, unsafe_allow_html=True)

        r_img_col, r_data_col = st.columns([1.2, 1], gap="medium")
        with r_img_col:
            st.markdown(f'<div class="saas-card"><div class="saas-card-title">Annotated Image</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB), use_container_width=True)
            _, img_buf = cv2.imencode(".jpg", res["annotated"])
            st.download_button(
                label="⬇️ Download Annotated Photo",
                data=img_buf.tobytes(),
                file_name=f"traffic_detected_{res['timestamp']}.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with r_data_col:
            st.markdown(f'<div class="saas-card"><div class="saas-card-title">Detection Breakdown</div>', unsafe_allow_html=True)
            if res["records"]:
                df_img = pd.DataFrame(res["records"])
                st.dataframe(df_img, use_container_width=True, hide_index=True)
                counts_series = df_img["Vehicle Type"].value_counts().reset_index()
                counts_series.columns = ["Vehicle Type", "Count"]
                st.bar_chart(counts_series.set_index("Vehicle Type"), color=ACCENT, use_container_width=True)
            else:
                st.info("No vehicles detected at this sensitivity. Try setting Detection Sensitivity to 'High'.")
            st.markdown("</div>", unsafe_allow_html=True)

    elif not input_is_img and st.session_state.get("video_analysis_result"):
        vres = st.session_state.video_analysis_result
        st.markdown("<hr/>", unsafe_allow_html=True)

        if vres.get("violations_logged", 0) > 0:
            st.warning(f"⚠️ {vres['violations_logged']} safety violation(s) flagged and recorded to Records & Reports.")

        st.markdown(f"""
        <div style="font-size: 15px; font-weight: 700; color: {TEXT}; margin-bottom: 12px;">
            Results: {vres['total_counted']} Vehicles Counted Across Virtual Line
        </div>
        """, unsafe_allow_html=True)

        v_player_col, v_chart_col = st.columns([1.2, 1], gap="medium")
        with v_player_col:
            st.markdown(f'<div class="saas-card"><div class="saas-card-title">Processed Video Playback</div>', unsafe_allow_html=True)
            st.video(vres["out_path"])
            st.download_button(
                label="⬇️ Download Processed Video",
                data=vres["video_bytes"],
                file_name=f"traffic_counted_{vres['timestamp']}.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with v_chart_col:
            st.markdown(f'<div class="saas-card"><div class="saas-card-title">Vehicle Category Counts</div>', unsafe_allow_html=True)
            counts = vres.get("class_counts", {})
            if counts and sum(counts.values()) > 0:
                df_vcounts = (
                    pd.DataFrame(list(counts.items()), columns=["Category", "Count"])
                    .sort_values("Count", ascending=False)
                )
                st.dataframe(df_vcounts, use_container_width=True, hide_index=True)
                st.bar_chart(df_vcounts.set_index("Category"), color=ACCENT, use_container_width=True)
            else:
                st.markdown(f"""
                <div style="font-size: 13px; color: {MUTED}; line-height: 1.5;">
                    ℹ️ <b>0 vehicles crossed the counting line</b> during this short preview segment.<br>
                    To record line crossings, select <b>Extended Segment</b> in Optional Settings.
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. NUMBER PLATE SCANNER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Number Plate Scanner":
    render_header(
        "Number Plate Scanner",
        "Detect license plates and extract registration numbers using AI OCR.",
    )

    c_sel, c_view = st.columns([1.1, 1], gap="large")

    plate_input_img = None
    with c_sel:
        plate_src = st.radio(
            "Plate Photo Source",
            options=["Try sample vehicle plate", "Upload vehicle photo"],
            horizontal=True,
            key="anpr_source_choice",
        )

        if plate_src == "Try sample vehicle plate":
            if DEFAULT_PLATE_IMG.exists():
                plate_input_img = cv2.imread(str(DEFAULT_PLATE_IMG))
            else:
                st.warning("Sample plate photo not found in `data/input/plate_test.jpg`.")
        else:
            up_plate = st.file_uploader(
                "Upload vehicle photo (JPG, PNG)",
                type=["jpg", "jpeg", "png"],
                key="anpr_uploader",
            )
            if up_plate is not None:
                val = up_plate.getvalue()
                plate_input_img = cv2.imdecode(np.asarray(bytearray(val), dtype=np.uint8), cv2.IMREAD_COLOR)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        scan_btn = st.button(
            "🔍  Scan Plate",
            type="primary",
            disabled=(plate_input_img is None),
            use_container_width=True,
            key="btn_scan_plate",
        )

    with c_view:
        st.markdown(f'<div class="saas-card" style="min-height: 200px;"><div class="saas-card-title">Vehicle Photo Preview</div>', unsafe_allow_html=True)
        if plate_input_img is not None:
            # Display compact framed preview
            th, tw = plate_input_img.shape[:2]
            scale = 440.0 / float(tw)
            p_thumb = cv2.resize(plate_input_img, (440, int(th * scale)))
            st.image(cv2.cvtColor(p_thumb, cv2.COLOR_BGR2RGB), use_container_width=False, width=400)
        else:
            st.markdown(f"""
            <div class="empty-box" style="padding: 30px 10px; border:none;">
                <div class="empty-icon">🚗</div>
                <div class="empty-title">No Plate Photo Selected</div>
                <div class="empty-desc">Choose a sample image or upload a vehicle photo to begin.</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if scan_btn and plate_input_img is not None:
        plate_det = load_plate_detector()
        plate_ocr = load_plate_ocr()

        if plate_det is None:
            st.error("License plate model weights (`models/plate_detector.pt`) are missing.")
        else:
            with st.spinner("Detecting license plate & reading registration text…"):
                detections = plate_det.detect_plates(plate_input_img)

            if not detections:
                st.session_state.plate_scan_results = []
                st.warning("No license plate found in this image. Ensure the plate is clearly visible.")
            else:
                annotated = plate_det.draw_detections(plate_input_img, detections)
                crops = plate_det.crop_plates(plate_input_img, detections)
                scan_items = []
                for idx, (det, crop) in enumerate(zip(detections, crops), 1):
                    text, conf = plate_ocr.read_plate(crop) if crop is not None else ("", 0.0)
                    clean = text.strip() if text else "UNREADABLE"
                    crop_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    crop_path = OUTPUT_DIR / f"plate_{crop_ts}_{idx}.jpg"
                    if crop is not None:
                        cv2.imwrite(str(crop_path), crop)
                    scan_items.append({
                        "idx": idx,
                        "det": det,
                        "crop_path": str(crop_path),
                        "text": clean,
                        "conf": conf,
                        "saved_id": None,
                    })
                st.session_state.plate_scan_annotated = annotated
                st.session_state.plate_scan_results = scan_items

    # Render Scan Results
    if st.session_state.get("plate_scan_results"):
        results_list = st.session_state.plate_scan_results
        annotated_img = st.session_state.get("plate_scan_annotated")

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.success(f"Identified {len(results_list)} license plate(s) successfully!")

        for item in results_list:
            idx = item["idx"]
            det = item["det"]
            crop_path = item["crop_path"]
            clean = item["text"]
            conf = item["conf"]

            p_col1, p_col2, p_col3 = st.columns([1.2, 0.8, 1], gap="medium")
            with p_col1:
                st.markdown(f'<div class="saas-card"><div class="saas-card-title">Localization View</div>', unsafe_allow_html=True)
                if annotated_img is not None:
                    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with p_col2:
                st.markdown(f'<div class="saas-card"><div class="saas-card-title">Cropped Plate</div>', unsafe_allow_html=True)
                if os.path.exists(crop_path):
                    st.image(crop_path, use_container_width=True)
                else:
                    st.caption("Crop unavailable")
                st.markdown("</div>", unsafe_allow_html=True)

            with p_col3:
                st.markdown(f"""
                <div class="saas-card">
                    <div class="saas-card-title">Recognized Registration</div>
                    <div style="margin: 6px 0 12px 0;">
                        <span class="license-plate">{clean}</span>
                    </div>
                    <div style="font-size: 12.5px; color: {MUTED}; line-height: 1.8;">
                        <b style="color:{TEXT};">Detection Confidence:</b> {det['conf']*100:.1f}%<br>
                        <b style="color:{TEXT};">OCR Confidence:</b> {conf*100:.1f}%
                    </div>
                """, unsafe_allow_html=True)

                if item.get("saved_id"):
                    st.success(f"✅ Saved to Records (Event #{item['saved_id']})")
                else:
                    if st.button(f"💾 Save to Records", key=f"btn_save_plate_{idx}", use_container_width=True):
                        eid = db.insert_event(
                            event_type="PLATE_READ",
                            confidence=round(conf, 2),
                            plate_text=clean,
                            evidence_path=crop_path,
                            review_status="approved" if conf > 0.40 else "pending",
                        )
                        item["saved_id"] = eid
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. RECORDS & REPORTS — AUDIT LOG & EASY VIOLATIONS ACCESS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Records & Reports":
    render_header(
        "Records & Reports",
        "Searchable audit trail of traffic flow, number plate scans, and safety violations.",
    )

    # ── TWO DISTINCT TABS: ALL EVENTS vs VIOLATIONS ──
    tab_all, tab_viol = st.tabs(["📋 All Activity Records", "⚠️ Safety Violations"])

    # ── TAB 1: ALL ACTIVITY RECORDS ──
    with tab_all:
        f1, f2, f3 = st.columns([1.5, 1, 1], gap="medium")
        with f1:
            search_query = st.text_input(
                "Search all records",
                placeholder="Search by plate, vehicle type, or event…",
                label_visibility="collapsed",
                key="rr_search_all",
            )
        with f2:
            type_filter = st.selectbox(
                "Filter Event Type",
                options=["All Event Types", "Traffic Counts Only", "Plate Scans Only", "Safety Violations Only"],
                key="rr_type_filter_all",
            )
        with f3:
            status_filter = st.selectbox(
                "Filter Status",
                options=["All Statuses", "Pending", "Approved", "Rejected"],
                key="rr_status_filter_all",
            )

        status_arg = None if status_filter == "All Statuses" else status_filter.lower()
        raw_events = db.get_events(review_status=status_arg, limit=500)

        if type_filter == "Safety Violations Only":
            events = [e for e in raw_events if e.get("event_type") in ("NO_HELMET", "WRONG_WAY", "RED_LIGHT")]
        elif type_filter == "Plate Scans Only":
            events = [e for e in raw_events if e.get("event_type") == "PLATE_READ"]
        elif type_filter == "Traffic Counts Only":
            events = [e for e in raw_events if e.get("event_type") == "COUNT"]
        else:
            events = raw_events

        if search_query:
            sq = search_query.strip().lower()
            events = [
                e for e in events
                if sq in str(e.get("plate_text", "")).lower()
                or sq in str(e.get("vehicle_class", "")).lower()
                or sq in str(e.get("event_type", "")).lower()
                or sq in str(e.get("timestamp", "")).lower()
            ]

        if not events:
            st.markdown(f"""
            <div class="empty-box">
                <div class="empty-icon">📋</div>
                <div class="empty-title">No Records Found</div>
                <div class="empty-desc">No events match the selected filters. Process a video or photo to create new audit records.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            pending_cnt  = sum(1 for e in events if e.get("review_status") == "pending")
            approved_cnt = sum(1 for e in events if e.get("review_status") == "approved")
            rejected_cnt = sum(1 for e in events if e.get("review_status") == "rejected")

            st.markdown(f"""
            <div class="metrics-grid" style="margin: 12px 0;">
                <div class="metric-box">
                    <div class="metric-box-label">Filtered Records</div>
                    <div class="metric-box-value">{len(events)}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-label">Pending Review</div>
                    <div class="metric-box-value" style="color: {TAG_WARN_FG};">{pending_cnt}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-label">Approved</div>
                    <div class="metric-box-value" style="color: {TAG_OK_FG};">{approved_cnt}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-label">Rejected</div>
                    <div class="metric-box-value" style="color: {TAG_DANG_FG};">{rejected_cnt}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            table_rows = []
            for e in events:
                fev = format_event_display(e)
                table_rows.append({
                    "Record #": fev["id"],
                    "Date & Time": fev["timestamp"],
                    "Event Type": fev["event_type"],
                    "Vehicle Type": fev["vehicle_type"],
                    "Number Plate": fev["plate_number"],
                    "Confidence": fev["confidence"],
                    "Summary / Details": fev["details"],
                    "Review Status": fev["review_status"],
                })

            df_display = pd.DataFrame(table_rows)
            st.dataframe(
                df_display[["Date & Time", "Event Type", "Vehicle Type", "Number Plate", "Confidence", "Summary / Details", "Review Status"]],
                use_container_width=True,
                hide_index=True,
            )

            # CSV Export
            csv_data = df_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️  Export Records to CSV",
                data=csv_data,
                file_name=f"traffic_audit_log_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="rr_download_all_csv",
            )

            # Evidence Review Panel
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 15px; font-weight: 700; color: {TEXT}; margin-bottom: 10px;">Evidence &amp; Event Review</div>', unsafe_allow_html=True)

            selected_id = st.selectbox(
                "Select record to review",
                options=[e["id"] for e in events],
                format_func=lambda x: (
                    f"Record #{x} — {next((r['Event Type'] for r in table_rows if r['Record #']==x), 'Event')} "
                    f"({next((r['Date & Time'] for r in table_rows if r['Record #']==x), '')})"
                ),
                key="rr_select_all_event",
            )

            if selected_id:
                sel_ev = db.get_event_by_id(selected_id)
                if sel_ev:
                    fsel = format_event_display(sel_ev)
                    det_col, ev_col = st.columns([1, 1.2], gap="medium")
                    with det_col:
                        st.markdown(f"""
                        <div class="saas-card">
                            <div class="saas-card-title">Record #{fsel['id']} Overview</div>
                            <table style="width:100%; font-size:13px; line-height:2.1; border-collapse:collapse;">
                                <tr><td style="color:{MUTED}; width:38%;">Date &amp; Time</td><td><b>{fsel['timestamp']}</b></td></tr>
                                <tr><td style="color:{MUTED};">Event Type</td><td><b>{fsel['event_type']}</b></td></tr>
                                <tr><td style="color:{MUTED};">Vehicle Info</td><td><b>{fsel['vehicle_type']}</b></td></tr>
                                <tr><td style="color:{MUTED};">Number Plate</td><td><code>{fsel['plate_number']}</code></td></tr>
                                <tr><td style="color:{MUTED};">Summary / Details</td><td>{fsel['details']}</td></tr>
                                <tr><td style="color:{MUTED};">Confidence</td><td>{fsel['confidence']}</td></tr>
                                <tr><td style="color:{MUTED};">Evidence File</td><td><small>{fsel['evidence_label']}</small></td></tr>
                                <tr><td style="color:{MUTED};">Review Status</td><td><b>{fsel['review_status']}</b></td></tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)

                        b_app, b_rej, b_del = st.columns(3)
                        with b_app:
                            if st.button("Approve", key="btn_app_all", use_container_width=True):
                                db.update_review_status(selected_id, "approved")
                                st.rerun()
                        with b_rej:
                            if st.button("Reject", key="btn_rej_all", use_container_width=True):
                                db.update_review_status(selected_id, "rejected")
                                st.rerun()
                        with b_del:
                            if st.button("Delete", key="btn_del_all", use_container_width=True):
                                db.delete_event(selected_id)
                                st.rerun()

                    with ev_col:
                        st.markdown(f'<div class="saas-card"><div class="saas-card-title">Attached Evidence Media</div>', unsafe_allow_html=True)
                        ev_path = sel_ev.get("evidence_path") or ""
                        if ev_path and os.path.exists(ev_path):
                            low = ev_path.lower()
                            if low.endswith((".jpg", ".jpeg", ".png")):
                                st.image(ev_path, caption=f"Evidence Snapshot: {Path(ev_path).name}", use_container_width=True)
                            elif low.endswith((".mp4", ".avi", ".mov")):
                                st.video(ev_path)
                        else:
                            st.markdown(f"""
                            <div style="text-align:center; padding:30px; color:{MUTED}; font-size:13px;">
                                📁 No evidence file available for this record.
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2: SAFETY VIOLATIONS SPECIFIC TAB ──
    with tab_viol:
        st.markdown(f"""
        <div class="saas-card" style="border-left: 4px solid {ACCENT};">
            <div style="font-size: 13.5px; font-weight: 600; color: {TEXT}; margin-bottom: 4px;">
                ⚠️ Safety Violation Enforcement Log
            </div>
            <div style="font-size: 12px; color: {MUTED}; line-height: 1.5;">
                This section monitors automated safety violations including <b>Helmet Compliance</b> (two-wheeler riders),
                <b>Wrong-Way Movement</b> (opposing designated traffic flow), and <b>Stop-Line Crossings</b>.<br>
                <i>Notice: Red-light violation monitoring operates on a policy-based virtual stop-line check (simulated signal phase), not automatic optical camera recognition of physical traffic light fixtures.</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

        vf1, vf2 = st.columns([1.2, 1], gap="medium")
        with vf1:
            viol_filter = st.selectbox(
                "Filter Violation Category",
                options=["All Violations", "Helmet Violations (NO_HELMET)", "Wrong-Way Movement (WRONG_WAY)", "Virtual Stop-Line (RED_LIGHT)"],
                key="rr_viol_type_filter",
            )
        with vf2:
            viol_status_filter = st.selectbox(
                "Filter Violation Status",
                options=["All Statuses", "Pending", "Approved", "Rejected"],
                key="rr_viol_status_filter",
            )

        # Query all genuine violation records
        all_viols = db.get_events(limit=500)
        viols = [
            e for e in all_viols
            if e.get("event_type") in ("NO_HELMET", "WRONG_WAY", "RED_LIGHT")
        ]

        if viol_filter == "Helmet Violations (NO_HELMET)":
            viols = [e for e in viols if e.get("event_type") == "NO_HELMET"]
        elif viol_filter == "Wrong-Way Movement (WRONG_WAY)":
            viols = [e for e in viols if e.get("event_type") == "WRONG_WAY"]
        elif viol_filter == "Virtual Stop-Line (RED_LIGHT)":
            viols = [e for e in viols if e.get("event_type") == "RED_LIGHT"]

        if viol_status_filter != "All Statuses":
            viols = [e for e in viols if (e.get("review_status") or "pending").lower() == viol_status_filter.lower()]

        if not viols:
            st.markdown(f"""
            <div class="empty-box" style="padding: 36px 24px;">
                <div class="empty-icon">🛡️</div>
                <div class="empty-title">No Safety Violations Logged</div>
                <div class="empty-desc" style="max-width: 520px; line-height: 1.6;">
                    The database currently contains no violation records.<br><br>
                    <b>How to capture genuine violations:</b><br>
                    1. Go to <b>Analyze Traffic</b> and select <b>Upload Video</b>.<br>
                    2. Expand <b>Optional Settings</b> and enable <b>Helmet Compliance Check</b> (for motorcycle riders) or <b>Wrong-Way Traffic Check</b>.<br>
                    3. Click <b>Analyze Now</b>. Any detected violations will be automatically logged here with evidence snapshots.<br><br>
                    <i>Notice: Red-light violation monitoring operates on a policy-based virtual stop-line check (simulated signal phase), not automatic optical camera recognition of physical traffic light fixtures.</i>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            v_pending  = sum(1 for e in viols if e.get("review_status") == "pending")
            v_approved = sum(1 for e in viols if e.get("review_status") == "approved")
            v_rejected = sum(1 for e in viols if e.get("review_status") == "rejected")

            st.markdown(f"""
            <div class="metrics-grid" style="margin: 12px 0;">
                <div class="metric-box">
                    <div class="metric-box-label">Total Violations</div>
                    <div class="metric-box-value" style="color: {TAG_DANG_FG};">{len(viols)}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-label">Pending Review</div>
                    <div class="metric-box-value" style="color: {TAG_WARN_FG};">{v_pending}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-label">Approved</div>
                    <div class="metric-box-value" style="color: {TAG_OK_FG};">{v_approved}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-label">Rejected</div>
                    <div class="metric-box-value">{v_rejected}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            v_rows = []
            for e in viols:
                fvev = format_event_display(e)
                v_rows.append({
                    "Violation #": fvev["id"],
                    "Date & Time": fvev["timestamp"],
                    "Violation Rule": fvev["event_type"],
                    "Vehicle Type": fvev["vehicle_type"],
                    "Number Plate": fvev["plate_number"],
                    "Confidence": fvev["confidence"],
                    "Review Status": fvev["review_status"],
                })

            df_viols = pd.DataFrame(v_rows)
            st.dataframe(
                df_viols[["Date & Time", "Violation Rule", "Vehicle Type", "Number Plate", "Confidence", "Review Status"]],
                use_container_width=True,
                hide_index=True,
            )

            # Export Violations CSV
            v_csv = df_viols.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️  Export Violations to CSV",
                data=v_csv,
                file_name=f"traffic_violations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="rr_download_viols_csv",
            )

            # View Evidence Action for Selected Violation
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 15px; font-weight: 700; color: {TEXT}; margin-bottom: 10px;">Violation Evidence &amp; Adjudication</div>', unsafe_allow_html=True)

            sel_viol_id = st.selectbox(
                "Select violation to view evidence",
                options=[e["id"] for e in viols],
                format_func=lambda x: (
                    f"Violation #{x} — {next((r['Violation Rule'] for r in v_rows if r['Violation #']==x), 'Violation')} "
                    f"({next((r['Date & Time'] for r in v_rows if r['Violation #']==x), '')})"
                ),
                key="rr_select_viol_item",
            )

            if sel_viol_id:
                sel_v_ev = db.get_event_by_id(sel_viol_id)
                if sel_v_ev:
                    fvsel = format_event_display(sel_v_ev)
                    v_col1, v_col2 = st.columns([1, 1.2], gap="medium")
                    with v_col1:
                        st.markdown(f"""
                        <div class="saas-card">
                            <div class="saas-card-title">Violation #{fvsel['id']} Overview</div>
                            <table style="width:100%; font-size:13px; line-height:2.2; border-collapse:collapse;">
                                <tr><td style="color:{MUTED}; width:40%;">Date &amp; Time</td><td><b>{fvsel['timestamp']}</b></td></tr>
                                <tr><td style="color:{MUTED};">Violation Rule</td><td><b>{fvsel['event_type']}</b></td></tr>
                                <tr><td style="color:{MUTED};">Vehicle Type</td><td><b>{fvsel['vehicle_type']}</b></td></tr>
                                <tr><td style="color:{MUTED};">Number Plate</td><td><code>{fvsel['plate_number']}</code></td></tr>
                                <tr><td style="color:{MUTED};">Summary / Details</td><td>{fvsel['details']}</td></tr>
                                <tr><td style="color:{MUTED};">Confidence</td><td>{fvsel['confidence']}</td></tr>
                                <tr><td style="color:{MUTED};">Evidence File</td><td><small>{fvsel['evidence_label']}</small></td></tr>
                                <tr><td style="color:{MUTED};">Review Status</td><td><b>{fvsel['review_status']}</b></td></tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)

                        va, vr, vd = st.columns(3)
                        with va:
                            if st.button("Approve", key="btn_app_viol", use_container_width=True):
                                db.update_review_status(sel_viol_id, "approved")
                                st.rerun()
                        with vr:
                            if st.button("Reject", key="btn_rej_viol", use_container_width=True):
                                db.update_review_status(sel_viol_id, "rejected")
                                st.rerun()
                        with vd:
                            if st.button("Delete", key="btn_del_viol", use_container_width=True):
                                db.delete_event(sel_viol_id)
                                st.rerun()

                    with v_col2:
                        st.markdown(f'<div class="saas-card"><div class="saas-card-title">Evidence Media / Crop Snapshot</div>', unsafe_allow_html=True)
                        v_ev_path = sel_v_ev.get("evidence_path") or ""
                        if v_ev_path and os.path.exists(v_ev_path):
                            if v_ev_path.lower().endswith((".jpg", ".jpeg", ".png")):
                                st.image(v_ev_path, caption=f"Evidence Snapshot: {Path(v_ev_path).name}", use_container_width=True)
                            elif v_ev_path.lower().endswith((".mp4", ".avi", ".mov")):
                                st.video(v_ev_path)
                        else:
                            st.markdown(f"""
                            <div style="text-align:center; padding:30px; color:{MUTED}; font-size:13px;">
                                📁 Evidence file not found on disk.
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)


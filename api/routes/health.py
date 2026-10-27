import os
import sys
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Depends

from api.schemas import HealthResponse, DiagnosticsResponse
from api.dependencies import get_db
from database.database_manager import DatabaseManager

router = APIRouter(prefix="/api", tags=["Health & Diagnostics"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """
    Returns general backend operational status and timestamp.
    """
    return HealthResponse(
        status="healthy",
        app="Smart Traffic Vision API",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@router.get("/diagnostics", response_model=DiagnosticsResponse)
def get_diagnostics(db: DatabaseManager = Depends(get_db)) -> DiagnosticsResponse:
    """
    Performs comprehensive diagnostic checks:
    - Model file availability and sizes
    - SQLite database connectivity and event counts
    - Required input/output directory permissions
    - OpenH264 codec library status
    - Python and PyTorch execution environment
    """
    # 1. Model inspections
    models_to_check = {
        "vehicle_detector": "models/vehicle_detector.pt",
        "plate_detector": "models/plate_detector.pt",
        "helmet_detector": "models/helmet_detector.pt",
        "base_yolo": "yolo11n.pt",
    }
    models_info = {}
    for name, path in models_to_check.items():
        exists = os.path.exists(path)
        size_mb = round(os.path.getsize(path) / (1024 * 1024), 2) if exists else 0.0
        models_info[name] = {
            "path": path,
            "exists": exists,
            "size_mb": size_mb,
        }

    # 2. Database checks
    db_connected = False
    table_exists = False
    total_events = 0
    total_violations = 0
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events';")
            table_exists = cursor.fetchone() is not None
            db_connected = True
        total_events = db.count_events()
        total_violations = db.count_events(exclude_event_types=["COUNT", "PLATE_READ"])
    except Exception as e:
        print(f"[DIAGNOSTICS] DB check error: {e}")

    db_info = {
        "db_path": db.db_path,
        "exists": os.path.exists(db.db_path),
        "connected": db_connected,
        "table_exists": table_exists,
        "total_events": total_events,
        "total_violations": total_violations,
    }

    # 3. Directory checks
    dirs_to_check = [
        "data/input",
        "data/output",
        "data/output/violations",
    ]
    dirs_info = {}
    for d in dirs_to_check:
        Path(d).mkdir(parents=True, exist_ok=True)
        exists = os.path.exists(d)
        writable = os.access(d, os.W_OK) if exists else False
        dirs_info[d] = {
            "exists": exists,
            "writable": writable,
        }

    # 4. OpenH264 DLL check
    openh264_path = "openh264-2.5.0-win64.dll"
    openh264_exists = os.path.exists(openh264_path)
    openh264_size_kb = round(os.path.getsize(openh264_path) / 1024, 1) if openh264_exists else 0.0
    openh264_info = {
        "path": openh264_path,
        "exists": openh264_exists,
        "size_kb": openh264_size_kb,
    }

    # 5. Environment details
    torch_version = None
    cuda_available = False
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    env_info = {
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "torch_version": torch_version,
        "cuda_available": cuda_available,
    }

    return DiagnosticsResponse(
        status="ok" if db_connected else "degraded",
        models=models_info,
        database=db_info,
        directories=dirs_info,
        openh264=openh264_info,
        environment=env_info,
    )

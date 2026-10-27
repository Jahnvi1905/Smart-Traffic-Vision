import os
from pathlib import Path
from typing import Optional

from database.database_manager import DatabaseManager, DEFAULT_DB_PATH
from detection.vehicle_detector import VehicleDetector
from ocr.plate_detector import PlateDetector
from ocr.plate_ocr import PlateOCR

# Global cached model instances
_vehicle_detector: Optional[VehicleDetector] = None
_plate_detector: Optional[PlateDetector] = None
_plate_ocr: Optional[PlateOCR] = None


def get_db() -> DatabaseManager:
    """
    Dependency provider for DatabaseManager.
    Can be easily overridden in tests using app.dependency_overrides[get_db].
    """
    return DatabaseManager(db_path=DEFAULT_DB_PATH)


def get_vehicle_detector() -> VehicleDetector:
    """
    Returns a cached VehicleDetector instance.
    Loads models/vehicle_detector.pt if present, falling back to base yolo11n.pt.
    """
    global _vehicle_detector
    if _vehicle_detector is None:
        custom_weights = "models/vehicle_detector.pt"
        base_weights = "yolo11n.pt"
        chosen_path = custom_weights if os.path.exists(custom_weights) else base_weights
        _vehicle_detector = VehicleDetector(model_path=chosen_path, conf_threshold=0.35)
        _vehicle_detector.load_model()
    return _vehicle_detector


def get_plate_detector() -> Optional[PlateDetector]:
    """
    Returns a cached PlateDetector instance, or None if weights are not found.
    """
    global _plate_detector
    if _plate_detector is None:
        model_path = "models/plate_detector.pt"
        if os.path.exists(model_path):
            _plate_detector = PlateDetector(model_path=model_path, conf_threshold=0.25)
            _plate_detector.load_model()
    return _plate_detector


def get_plate_ocr() -> PlateOCR:
    """
    Returns a cached PlateOCR reader instance.
    """
    global _plate_ocr
    if _plate_ocr is None:
        _plate_ocr = PlateOCR(gpu=False)
        _plate_ocr.load_model()
    return _plate_ocr

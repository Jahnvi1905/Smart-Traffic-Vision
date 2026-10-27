from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ── Health & Diagnostics ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "healthy"
    app: str = "Smart Traffic Vision API"
    version: str = "1.0.0"
    timestamp: str


class DiagnosticsResponse(BaseModel):
    status: str
    models: Dict[str, Any]
    database: Dict[str, Any]
    directories: Dict[str, Any]
    openh264: Dict[str, Any]
    environment: Dict[str, Any]


# ── Overview Stats ─────────────────────────────────────────────────────────────

class RecentEventItem(BaseModel):
    id: int
    timestamp: str
    track_id: Optional[int] = None
    vehicle_class: Optional[str] = None
    event_type: str
    confidence: Optional[float] = None
    plate_text: Optional[str] = None
    evidence_url: Optional[str] = None
    review_status: str = "pending"
    count_details: Optional[str] = None
    total_count: Optional[int] = None


class OverviewStatsResponse(BaseModel):
    total_vehicles: int = Field(0, description="Cumulative vehicle crossings counted across all sessions")
    total_violations: int = Field(0, description="Total safety violation incidents (excludes COUNT events)")
    violations_by_type: Dict[str, int] = Field(default_factory=dict)
    category_breakdown: Dict[str, int] = Field(default_factory=dict)
    recent_events: List[RecentEventItem] = Field(default_factory=list)


# ── Violations Management ──────────────────────────────────────────────────────

class ViolationItem(BaseModel):
    id: int
    timestamp: str
    track_id: Optional[int] = None
    vehicle_class: Optional[str] = None
    event_type: str
    confidence: Optional[float] = None
    plate_text: Optional[str] = None
    evidence_url: Optional[str] = None
    review_status: str
    count_details: Optional[str] = None
    total_count: Optional[int] = None


class ViolationsListResponse(BaseModel):
    total: int
    page: int
    limit: int
    pages: int
    items: List[ViolationItem]


class ReviewStatusUpdate(BaseModel):
    status: str = Field(..., description="Target review status: 'pending', 'approved', or 'rejected'")


class ReviewStatusResponse(BaseModel):
    id: int
    review_status: str
    updated: bool


# ── Video & Telemetry ──────────────────────────────────────────────────────────

class TelemetryItem(BaseModel):
    frame_idx: int
    timestamp_s: float
    cumulative_total: int
    cumulative_classes: Dict[str, int]
    visible_total: int
    visible_classes: Dict[str, int]
    newly_counted: List[int]


class AnalyzeVideoResponse(BaseModel):
    status: str = "success"
    video_url: str
    total_counted: int
    class_counts: Dict[str, int]
    violations_logged: int
    telemetry: List[TelemetryItem]
    frames_processed: int


# ── Image Analysis ─────────────────────────────────────────────────────────────

class DetectionBoxItem(BaseModel):
    class_name: str
    label: str
    confidence: float
    box: List[int]  # [x1, y1, x2, y2]


class AnalyzeImageResponse(BaseModel):
    status: str = "success"
    annotated_image_url: str
    total_detected: int
    class_counts: Dict[str, int]
    detections: List[DetectionBoxItem]


# ── Plate Scanner ──────────────────────────────────────────────────────────────

class PlateItem(BaseModel):
    idx: int
    plate_text: str
    raw_text: Optional[str] = None
    confidence: float
    is_low_confidence: bool = False
    confidence_status: str = "medium"
    crop_url: Optional[str] = None
    box: List[int]  # [x1, y1, x2, y2]


class PlateScanResponse(BaseModel):
    status: str = "success"
    annotated_image_url: Optional[str] = None
    total_plates: int
    plates: List[PlateItem]

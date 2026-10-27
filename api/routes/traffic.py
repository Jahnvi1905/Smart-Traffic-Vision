import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from api.schemas import (
    AnalyzeImageResponse,
    AnalyzeVideoResponse,
    DetectionBoxItem,
    TelemetryItem,
)
from api.dependencies import get_db, get_vehicle_detector
from api.utils import (
    generate_safe_filename,
    save_upload_stream,
    to_media_url,
    validate_file_extension,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
)
from counting.vehicle_counter import VehicleCounter
from database.database_manager import DatabaseManager
from detection.vehicle_detector import VehicleDetector
from violations.violation_manager import TrafficViolationManager

router = APIRouter(prefix="/api/traffic", tags=["Traffic Vision Analysis"])

DEFAULT_SAMPLE_VIDEO = Path("data/input/sample.mp4")
OUTPUT_DIR = Path("data/output")
INPUT_DIR = Path("data/input")
VIOLATIONS_DIR = Path("data/output/violations")
HELMET_MODEL_PATH = Path("models/helmet_detector.pt")
PLATE_MODEL_PATH = Path("models/plate_detector.pt")


@router.post("/analyze-image", response_model=AnalyzeImageResponse)
async def analyze_image(
    file: UploadFile = File(..., description="Vehicle photo for classification and bounding box detection"),
    conf_threshold: float = Form(0.35, ge=0.1, le=1.0),
    detector: VehicleDetector = Depends(get_vehicle_detector)
) -> AnalyzeImageResponse:
    """
    Analyzes a single traffic snapshot using YOLO vehicle detection.
    Returns localized bounding boxes, classifications, confidence scores, and an annotated image.
    """
    validate_file_extension(file.filename, ALLOWED_IMAGE_EXTENSIONS)
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None or image_bgr.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image. Please ensure the file is a valid image format."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector.conf_threshold = conf_threshold

    # Run YOLO inference
    try:
        results = detector.model(
            image_bgr,
            conf=conf_threshold,
            classes=detector.target_class_ids,
            verbose=False
        )
        boxes = results[0].boxes if (results and len(results) > 0) else []
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during image analysis: {str(exc)}"
        )

    annotated = detector.draw_detections(image_bgr.copy(), boxes)
    filename = generate_safe_filename("detected", "jpg")
    out_path = OUTPUT_DIR / filename
    cv2.imwrite(str(out_path), annotated)

    detections = []
    class_counts = {}
    for b in boxes:
        cid = int(b.cls[0].item())
        cname = detector.target_classes.get(cid, str(cid)).lower()
        conf = float(b.conf[0].item())
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

        detections.append(
            DetectionBoxItem(
                class_name=cname,
                label=cname.title(),
                confidence=round(conf, 3),
                box=[x1, y1, x2, y2]
            )
        )
        class_counts[cname] = class_counts.get(cname, 0) + 1

    return AnalyzeImageResponse(
        status="success",
        annotated_image_url=to_media_url(str(out_path)),
        total_detected=len(detections),
        class_counts=class_counts,
        detections=detections
    )


@router.post("/analyze-video", response_model=AnalyzeVideoResponse)
async def analyze_video(
    file: Optional[UploadFile] = File(None),
    use_sample: bool = Form(False),
    max_frames: Optional[int] = Form(None, ge=1, le=5000),
    frame_skip: int = Form(1, ge=1, le=10),
    enable_counter: bool = Form(True),
    enable_helmet: bool = Form(True),
    enable_wrong_way: bool = Form(True),
    enable_red_light: bool = Form(False),
    expected_direction: str = Form("down"),
    signal_state: str = Form("GREEN"),
    conf_threshold: float = Form(0.35, ge=0.1, le=1.0),
    detector: VehicleDetector = Depends(get_vehicle_detector),
    db: DatabaseManager = Depends(get_db)
) -> AnalyzeVideoResponse:
    """
    Processes video streams through the end-to-end computer vision pipeline:
    - ByteTrack multi-object vehicle tracking
    - Dynamic virtual line counting with class aggregation
    - Safety violation auditing (helmet, wrong-way, virtual stop-line)
    - Per-frame telemetry recording for playback synchronization
    """
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIOLATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Determine input video source
    if use_sample:
        if not DEFAULT_SAMPLE_VIDEO.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Benchmark sample video not found at '{DEFAULT_SAMPLE_VIDEO}'."
            )
        input_video_path = str(DEFAULT_SAMPLE_VIDEO)
        cleanup_input = False
    elif file is not None:
        ext = validate_file_extension(file.filename, ALLOWED_VIDEO_EXTENSIONS)
        safe_in_name = generate_safe_filename("upload", ext)
        dest_input = INPUT_DIR / safe_in_name
        await save_upload_stream(file, str(dest_input))
        input_video_path = str(dest_input)
        cleanup_input = False
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either upload a video 'file' or set 'use_sample=true'."
        )

    # 2. Inspect video stream dimensions
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not open video file. Ensure file is not corrupted."
        )
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if width <= 0 or height <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video dimensions encountered."
        )

    # 3. Setup output destination
    safe_out_name = generate_safe_filename("processed", "mp4")
    output_video_path = str(OUTPUT_DIR / safe_out_name)

    # 4. Configure counting line and violation suites
    detector.conf_threshold = conf_threshold
    calc_y = int(height * 0.55)

    counter = None
    if enable_counter:
        counter = VehicleCounter(
            line_start=(0, calc_y),
            line_end=(width, calc_y),
            target_classes=detector.target_classes
        )

    violation_manager = None
    if enable_helmet or enable_wrong_way or enable_red_light:
        violation_manager = TrafficViolationManager(
            database_manager=db,
            evidence_dir=str(VIOLATIONS_DIR),
            helmet_model_path=str(HELMET_MODEL_PATH),
            plate_model_path=str(PLATE_MODEL_PATH),
            enable_helmet=enable_helmet,
            enable_wrong_way=enable_wrong_way,
            enable_red_light=enable_red_light,
            enable_plate_assoc=True,
            expected_direction=expected_direction,
            signal_state=signal_state,
        )

    # 5. Execute processing pipeline
    try:
        ok = detector.process_video(
            input_path=input_video_path,
            output_path=output_video_path,
            enable_tracking=True,
            tracker="bytetrack.yaml",
            counter=counter,
            violation_manager=violation_manager,
            max_frames=max_frames,
            frame_skip=frame_skip,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video processing failed: {str(exc)}"
        )

    if not ok or not os.path.exists(output_video_path) or os.path.getsize(output_video_path) < 100:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Video processing failed to produce a valid output stream."
        )

    # 6. Record COUNT event if counter active
    total_counted = counter.total_count if counter else 0
    class_counts = counter.class_counts if counter else {}
    if counter is not None:
        db.insert_event(
            event_type="COUNT",
            vehicle_class="Traffic Stream",
            confidence=conf_threshold,
            plate_text=None,
            evidence_path=output_video_path,
            review_status="approved",
            count_details=dict(class_counts),
            total_count=total_counted,
        )

    # 7. Collect per-frame telemetry
    raw_telemetry = counter.get_telemetry() if counter else []
    telemetry_items = [
        TelemetryItem(
            frame_idx=t["frame_idx"],
            timestamp_s=t["timestamp_s"],
            cumulative_total=t["cumulative_total"],
            cumulative_classes=t["cumulative_classes"],
            visible_total=t["visible_total"],
            visible_classes=t["visible_classes"],
            newly_counted=t["newly_counted"],
        )
        for t in raw_telemetry
    ]

    violations_logged = len(violation_manager.violations_logged) if violation_manager else 0

    return AnalyzeVideoResponse(
        status="success",
        video_url=to_media_url(output_video_path),
        total_counted=total_counted,
        class_counts=class_counts,
        violations_logged=violations_logged,
        telemetry=telemetry_items,
        frames_processed=len(telemetry_items),
    )

import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from api.schemas import PlateScanResponse, PlateItem
from api.dependencies import get_db, get_plate_detector, get_plate_ocr
from api.utils import to_media_url, validate_file_extension, ALLOWED_IMAGE_EXTENSIONS
from database.database_manager import DatabaseManager
from ocr.plate_detector import PlateDetector
from ocr.plate_ocr import PlateOCR

router = APIRouter(prefix="/api/plates", tags=["Number Plate Scanner"])

DEFAULT_SAMPLE_PLATE = Path("data/input/plate_test.jpg")
OUTPUT_DIR = Path("data/output")


@router.post("/scan", response_model=PlateScanResponse)
async def scan_plate(
    file: Optional[UploadFile] = File(None),
    use_sample: bool = Form(False),
    save_to_db: bool = Form(False),
    plate_detector: Optional[PlateDetector] = Depends(get_plate_detector),
    plate_ocr: PlateOCR = Depends(get_plate_ocr),
    db: DatabaseManager = Depends(get_db)
) -> PlateScanResponse:
    """
    Detects license plates and extracts alphanumeric text using fine-tuned YOLO and EasyOCR.
    Accepts either an uploaded vehicle image or a request to analyze the benchmark sample photo.
    """
    if plate_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="License plate detector weights ('models/plate_detector.pt') are not available."
        )

    # 1. Load source image
    image_bgr: Optional[np.ndarray] = None

    if use_sample:
        if not DEFAULT_SAMPLE_PLATE.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Benchmark sample plate image not found at '{DEFAULT_SAMPLE_PLATE}'."
            )
        image_bgr = cv2.imread(str(DEFAULT_SAMPLE_PLATE))
    elif file is not None:
        validate_file_extension(file.filename, ALLOWED_IMAGE_EXTENSIONS)
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either upload an image 'file' or set 'use_sample=true'."
        )

    if image_bgr is None or image_bgr.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image. Please ensure the file is a valid image format."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Run plate detection & cropping
    detections = plate_detector.detect_plates(image_bgr)
    if not detections:
        return PlateScanResponse(
            status="success",
            annotated_image_url=None,
            total_plates=0,
            plates=[]
        )

    crops = plate_detector.crop_plates(image_bgr, detections)
    annotated = plate_detector.draw_detections(image_bgr, detections)

    annotated_filename = f"plate_annotated_{ts}.jpg"
    annotated_path = OUTPUT_DIR / annotated_filename
    cv2.imwrite(str(annotated_path), annotated)

    # 3. Perform OCR on each localized crop
    plate_items = []
    for idx, (det, crop) in enumerate(zip(detections, crops), start=1):
        clean_text = "UNREADABLE"
        raw_text: Optional[str] = None
        ocr_conf = 0.0
        is_low_conf = True
        conf_status = "unreadable"
        crop_url = None

        if crop is not None and crop.size > 0:
            crop_filename = f"plate_crop_{ts}_{idx}.jpg"
            crop_path = OUTPUT_DIR / crop_filename
            cv2.imwrite(str(crop_path), crop)
            crop_url = to_media_url(str(crop_path))

            if hasattr(plate_ocr, "read_plate_detailed"):
                res = plate_ocr.read_plate_detailed(crop)
                if isinstance(res, dict):
                    ocr_detail = res
                else:
                    text, conf = plate_ocr.read_plate(crop)
                    from ocr.plate_ocr import validate_plate_output
                    ocr_detail = validate_plate_output(str(text), float(conf) if conf else 0.0)
            else:
                text, conf = plate_ocr.read_plate(crop)
                from ocr.plate_ocr import validate_plate_output
                ocr_detail = validate_plate_output(str(text), float(conf) if conf else 0.0)

            norm_text = str(ocr_detail.get("normalized_text") or "")
            if norm_text:
                clean_text = norm_text
                raw_text = str(ocr_detail.get("raw_text") or norm_text)
                ocr_conf = float(ocr_detail.get("confidence", 0.0))
                is_low_conf = bool(ocr_detail.get("is_low_confidence", False))
                conf_status = str(ocr_detail.get("confidence_status", "medium"))

        x1, y1, x2, y2 = det.get("box", [0, 0, 0, 0])
        plate_items.append(
            PlateItem(
                idx=idx,
                plate_text=clean_text,
                raw_text=raw_text,
                confidence=ocr_conf,
                is_low_confidence=is_low_conf,
                confidence_status=conf_status,
                crop_url=crop_url,
                box=[int(x1), int(y1), int(x2), int(y2)]
            )
        )

        if save_to_db:
            # Mark low confidence reads for human review ('pending') instead of auto-approving
            initial_status = "pending" if (is_low_conf or clean_text == "UNREADABLE") else "approved"
            db.insert_event(
                event_type="PLATE_READ",
                vehicle_class="Vehicle",
                confidence=ocr_conf if ocr_conf > 0 else det.get("conf"),
                plate_text=clean_text,
                evidence_path=str(crop_path) if crop_url else None,
                review_status=initial_status
            )

    return PlateScanResponse(
        status="success",
        annotated_image_url=to_media_url(str(annotated_path)),
        total_plates=len(plate_items),
        plates=plate_items
    )

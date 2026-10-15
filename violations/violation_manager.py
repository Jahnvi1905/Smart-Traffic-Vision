"""
Traffic Violation Detection Suite for Smart Traffic Vision.
Handles:
1. Helmet Detection & Violation Logging (for motorcycles / two-wheelers)
2. Wrong-Way Movement Detection (multi-frame trajectory consistency)
3. Red-Light / Virtual Stop-Line Violation (with configurable signal state)
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

try:
    from ocr.plate_detector import PlateDetector
    from ocr.plate_ocr import PlateOCR
except ImportError:
    try:
        import sys
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from ocr.plate_detector import PlateDetector
        from ocr.plate_ocr import PlateOCR
    except Exception:
        PlateDetector = None
        PlateOCR = None



class HelmetViolationDetector:
    """
    Detects helmet compliance on motorcycles/two-wheelers using fine-tuned YOLO weights.
    Avoids duplicate violation records per tracked vehicle.
    """

    def __init__(self, model_path: str = "models/helmet_detector.pt", conf_thresh: float = 0.25):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.model = None
        self.reported_tracks = set()  # track_ids that already had violation logged
        self.class_map = {0: "Helmet", 1: "fd", 2: "motorcycle", 3: "nohelmet", 4: "platenumber"}
        self.load_model()

    def load_model(self) -> bool:
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                if hasattr(self.model, "names") and self.model.names:
                    self.class_map = {int(k): str(v) for k, v in self.model.names.items()}
                return True
            except Exception as e:
                print(f"[WARN] Failed to load helmet model from {self.model_path}: {e}")
                self.model = None
        return False

    def check_motorcycle(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], track_id: Optional[int] = None) -> Optional[Dict]:
        """
        Inspects rider region of a detected motorcycle for helmet / nohelmet.
        Returns violation dict if NO_HELMET is confirmed, None otherwise.
        """
        if self.model is None or frame is None:
            return None

        if track_id is not None and track_id in self.reported_tracks:
            return None

        x1, y1, x2, y2 = bbox
        fh, fw = frame.shape[:2]
        w = x2 - x1
        h = y2 - y1

        # Rider head region is typically in the top 45% of motorcycle box
        rx1 = max(0, int(x1 - 0.05 * w))
        ry1 = max(0, int(y1 - 0.15 * h))
        rx2 = min(fw, int(x2 + 0.05 * w))
        ry2 = min(fh, int(y1 + 0.50 * h))

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        rider_crop = frame[ry1:ry2, rx1:rx2]
        if rider_crop.size == 0:
            return None

        try:
            results = self.model(rider_crop, conf=self.conf_thresh, verbose=False)
            if not results or not results[0].boxes:
                return None

            boxes = results[0].boxes
            has_helmet = False
            has_nohelmet = False
            nohelmet_conf = 0.0

            for b in boxes:
                cid = int(b.cls[0].item())
                cname = self.class_map.get(cid, "").lower()
                conf = float(b.conf[0].item())
                if "nohelmet" in cname or cid == 3:
                    has_nohelmet = True
                    nohelmet_conf = max(nohelmet_conf, conf)
                elif "helmet" in cname or cid == 0:
                    has_helmet = True

            # If no-helmet is detected without a confirmed helmet
            if has_nohelmet and not has_helmet:
                if track_id is not None:
                    self.reported_tracks.add(track_id)
                return {
                    "event_type": "NO_HELMET",
                    "confidence": nohelmet_conf,
                    "bbox": (x1, y1, x2, y2),
                    "rider_box": (rx1, ry1, rx2, ry2),
                    "crop": rider_crop
                }

        except Exception as e:
            print(f"[WARN] Error running helmet detection: {e}")

        return None


class WrongWayDetector:
    """
    Detects wrong-way vehicle movement based on multi-frame centroid displacement.
    Requires consistent trajectory history and rejects stationary/noisy tracks.
    """

    def __init__(
        self,
        expected_direction: str = "down",  # "down" (+y), "up" (-y), "left" (-x), "right" (+x)
        min_observations: int = 8,
        min_displacement_px: int = 40,
        consistency_ratio: float = 0.75
    ):
        self.expected_direction = expected_direction.lower()
        self.min_observations = min_observations
        self.min_displacement_px = min_displacement_px
        self.consistency_ratio = consistency_ratio

        self.track_history = {}       # track_id -> list of (cx, cy)
        self.reported_tracks = set()  # track_ids already flagged

    def update(self, track_id: int, center: Tuple[int, int]) -> Optional[Dict]:
        """
        Updates trajectory history for track_id and checks for wrong-way movement.
        Returns violation dict if wrong-way movement is confirmed, None otherwise.
        """
        if track_id is None:
            return None

        if track_id not in self.track_history:
            self.track_history[track_id] = []
        self.track_history[track_id].append(center)

        # Keep last 30 observations
        if len(self.track_history[track_id]) > 30:
            self.track_history[track_id].pop(0)

        if track_id in self.reported_tracks:
            return None

        history = self.track_history[track_id]
        if len(history) < self.min_observations:
            return None

        start_pt = history[0]
        curr_pt = history[-1]
        dx = curr_pt[0] - start_pt[0]
        dy = curr_pt[1] - start_pt[1]
        total_dist = (dx ** 2 + dy ** 2) ** 0.5

        # Reject stationary vehicles
        if total_dist < self.min_displacement_px:
            return None

        # Check step-by-step consistency in recent frames
        steps = [
            (history[i][0] - history[i - 1][0], history[i][1] - history[i - 1][1])
            for i in range(1, len(history))
        ]

        is_wrong_way = False
        step_matches = 0

        if self.expected_direction == "down":
            # Expected +y movement; wrong-way is moving up (-y)
            if dy < -self.min_displacement_px:
                step_matches = sum(1 for _, sdy in steps if sdy < 0)
                if step_matches / len(steps) >= self.consistency_ratio:
                    is_wrong_way = True

        elif self.expected_direction == "up":
            # Expected -y movement; wrong-way is moving down (+y)
            if dy > self.min_displacement_px:
                step_matches = sum(1 for _, sdy in steps if sdy > 0)
                if step_matches / len(steps) >= self.consistency_ratio:
                    is_wrong_way = True

        elif self.expected_direction == "right":
            if dx < -self.min_displacement_px:
                step_matches = sum(1 for sdx, _ in steps if sdx < 0)
                if step_matches / len(steps) >= self.consistency_ratio:
                    is_wrong_way = True

        elif self.expected_direction == "left":
            if dx > self.min_displacement_px:
                step_matches = sum(1 for sdx, _ in steps if sdx > 0)
                if step_matches / len(steps) >= self.consistency_ratio:
                    is_wrong_way = True

        if is_wrong_way:
            self.reported_tracks.add(track_id)
            return {
                "event_type": "WRONG_WAY",
                "track_id": track_id,
                "displacement": total_dist,
                "direction": f"{dx:+d}x, {dy:+d}y",
                "center": curr_pt
            }

        return None


class RedLightDetector:
    """
    Virtual Stop-Line Red-Light violation detector.
    Note: Real optical signal state classification is marked INCOMPLETE because sample
    camera footage lacks traffic signal lamps in the camera FOV; a virtual signal state
    and virtual stop-line are provided for policy verification.
    """

    def __init__(
        self,
        stop_line: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 1100), (1080, 1100)),
        signal_state: str = "GREEN"  # "RED", "GREEN"
    ):
        self.stop_line_start = tuple(map(int, stop_line[0]))
        self.stop_line_end = tuple(map(int, stop_line[1]))
        self.signal_state = signal_state.upper()
        self.reported_tracks = set()

    def set_signal_state(self, state: str):
        """Sets the virtual signal state ('RED' or 'GREEN')."""
        self.signal_state = state.upper()

    def adjust_line(self, width: int, height: int, line_y_pct: float = 60.0):
        """Scales virtual stop line to actual video dimensions."""
        calc_y = int(height * (line_y_pct / 100.0))
        self.stop_line_start = (0, calc_y)
        self.stop_line_end = (width, calc_y)

    def check_crossing(self, track_id: int, prev_center: Tuple[int, int], curr_center: Tuple[int, int]) -> Optional[Dict]:
        """
        Checks if vehicle crossed stop-line while signal state is RED.
        """
        if self.signal_state != "RED":
            return None

        if track_id is not None and track_id in self.reported_tracks:
            return None

        # Check line crossing
        py1, py2 = prev_center[1], curr_center[1]
        line_y = (self.stop_line_start[1] + self.stop_line_end[1]) / 2.0

        crossed = (py1 <= line_y <= py2) or (py2 <= line_y <= py1)
        if crossed:
            if track_id is not None:
                self.reported_tracks.add(track_id)
            return {
                "event_type": "RED_LIGHT",
                "track_id": track_id,
                "stop_line_y": int(line_y),
                "center": curr_center
            }
        return None


class PlateAssociationManager:
    """
    Associates detected license plates with tracked vehicles using geometric containment/overlap.
    Maintains a per-track cache of plate text and confidence to minimize redundant OCR.
    """

    def __init__(
        self,
        plate_model_path: str = "models/plate_detector.pt",
        detection_interval: int = 5,
        min_plate_conf: float = 0.25,
        min_ocr_conf: float = 0.30
    ):
        self.plate_model_path = plate_model_path
        self.detection_interval = max(1, detection_interval)
        self.min_plate_conf = min_plate_conf
        self.min_ocr_conf = min_ocr_conf

        self.plate_detector = None
        if PlateDetector is not None and os.path.exists(plate_model_path):
            try:
                self.plate_detector = PlateDetector(model_path=plate_model_path, conf_threshold=min_plate_conf)
                self.plate_detector.load_model()
            except Exception as e:
                print(f"[WARN] Could not initialize PlateDetector: {e}")

        self.plate_ocr = None
        if PlateOCR is not None:
            try:
                self.plate_ocr = PlateOCR(gpu=False)
            except Exception as e:
                print(f"[WARN] Could not initialize PlateOCR: {e}")

        # track_id -> {"text": str, "conf": float, "box": (x1, y1, x2, y2), "frame_idx": int}
        self.track_plates: Dict[int, Dict] = {}

    def get_plate_for_track(self, track_id: Optional[int]) -> Optional[Dict]:
        if track_id is None:
            return None
        return self.track_plates.get(track_id)

    def _associate_plate_to_box(
        self,
        plate_box: Tuple[int, int, int, int],
        vehicle_candidates: List[Tuple[int, Tuple[int, int, int, int], str]]
    ) -> Optional[int]:
        """
        Finds the most plausible vehicle track_id for the given plate bounding box.
        Uses containment, overlap ratio, and bottom-proximity heuristics.
        """
        px1, py1, px2, py2 = plate_box
        p_w = px2 - px1
        p_h = py2 - py1
        p_area = max(1, p_w * p_h)
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0

        best_tid = None
        best_score = -1.0

        for tid, (vx1, vy1, vx2, vy2), vclass in vehicle_candidates:
            v_w = vx2 - vx1
            v_h = vy2 - vy1
            v_area = max(1, v_w * v_h)

            # Plate cannot be larger than 40% of the vehicle area
            if p_area > 0.40 * v_area:
                continue

            # Check overlap / intersection
            ix1 = max(px1, vx1)
            iy1 = max(py1, vy1)
            ix2 = min(px2, vx2)
            iy2 = min(py2, vy2)
            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter_area = inter_w * inter_h
            overlap = inter_area / float(p_area)

            # Check if plate center is inside vehicle
            inside = (vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2)

            # Plausibility threshold: either plate center is inside, or at least 60% overlap
            if inside or overlap >= 0.60:
                vert_pos = max(0.0, min(1.0, (pcy - vy1) / float(v_h))) if v_h > 0 else 0.5
                score = overlap + 0.2 * vert_pos

                if score > best_score:
                    best_score = score
                    best_tid = tid

        return best_tid

    def process_frame(
        self,
        frame: np.ndarray,
        tracked_boxes,
        frame_idx: int = 0
    ):
        """
        Runs plate detection on interval frames, associates detected plates with vehicles,
        and performs conservative OCR on valid crops.
        """
        if self.plate_detector is None or frame is None or tracked_boxes is None or len(tracked_boxes) == 0:
            return

        if frame_idx % self.detection_interval != 0:
            return

        candidates = []
        for b in tracked_boxes:
            tid = int(b.id[0].item()) if (hasattr(b, "id") and b.id is not None) else None
            if tid is None:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            candidates.append((tid, (x1, y1, x2, y2), "vehicle"))

        if not candidates:
            return

        try:
            plate_detections = self.plate_detector.detect_plates(frame)
        except Exception as e:
            print(f"[WARN] Error running plate detector in video pipeline: {e}")
            return

        if not plate_detections:
            return

        fh, fw = frame.shape[:2]

        for det in plate_detections:
            p_box = tuple(map(int, det["box"]))
            px1, py1, px2, py2 = p_box

            px1 = max(0, min(px1, fw - 1))
            py1 = max(0, min(py1, fh - 1))
            px2 = max(0, min(px2, fw))
            py2 = max(0, min(py2, fh))
            if px2 - px1 < 20 or py2 - py1 < 10:
                continue

            matched_tid = self._associate_plate_to_box((px1, py1, px2, py2), candidates)
            if matched_tid is None:
                continue

            existing = self.track_plates.get(matched_tid)
            if existing and existing.get("conf", 0.0) >= 0.70 and existing.get("text", "") not in ("", "UNKNOWN"):
                continue

            if self.plate_ocr is not None:
                crop = frame[py1:py2, px1:px2]
                if crop.size > 0:
                    try:
                        text, ocr_conf = self.plate_ocr.read_plate(crop)
                        cleaned = "".join(c for c in text.upper() if c.isalnum() or c == " ").strip()
                        alpha_num_count = sum(1 for c in cleaned if c.isalnum())
                        if alpha_num_count >= 4 and ocr_conf >= self.min_ocr_conf:
                            self.track_plates[matched_tid] = {
                                "text": cleaned,
                                "conf": round(float(ocr_conf), 2),
                                "box": (px1, py1, px2, py2),
                                "frame_idx": frame_idx
                            }
                        elif not existing:
                            self.track_plates[matched_tid] = {
                                "text": "UNKNOWN",
                                "conf": round(float(ocr_conf), 2),
                                "box": (px1, py1, px2, py2),
                                "frame_idx": frame_idx
                            }
                    except Exception as e:
                        print(f"[WARN] Error in video plate OCR: {e}")


class TrafficViolationManager:
    """
    Central violation management coordinating helmet detection, wrong-way movement,
    stop-line enforcement, and plate-to-vehicle association with evidence saving and SQLite logging.
    """

    def __init__(
        self,
        database_manager=None,
        evidence_dir: str = "data/output/violations",
        helmet_model_path: str = "models/helmet_detector.pt",
        plate_model_path: str = "models/plate_detector.pt",
        enable_helmet: bool = True,
        enable_wrong_way: bool = True,
        enable_red_light: bool = False,
        enable_plate_assoc: bool = True,
        expected_direction: str = "down",
        plate_interval: int = 5,
    ):
        self.db = database_manager
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self.enable_helmet = enable_helmet
        self.enable_wrong_way = enable_wrong_way
        self.enable_red_light = enable_red_light
        self.enable_plate_assoc = enable_plate_assoc

        self.helmet_detector = HelmetViolationDetector(model_path=helmet_model_path) if enable_helmet else None
        self.wrong_way_detector = WrongWayDetector(expected_direction=expected_direction) if enable_wrong_way else None
        self.red_light_detector = RedLightDetector() if enable_red_light else None

        self.plate_assoc = None
        if enable_plate_assoc and os.path.exists(plate_model_path):
            self.plate_assoc = PlateAssociationManager(
                plate_model_path=plate_model_path,
                detection_interval=plate_interval
            )

        self._prev_centers: Dict[int, Tuple[int, int]] = {}
        self.violations_logged = []

    def process_frame(
        self,
        frame: np.ndarray,
        tracked_boxes,
        target_classes: Dict[int, str],
        frame_idx: int = 0
    ) -> List[Dict]:
        """
        Evaluates active violation rules for current frame detections and tracks.
        Associates license plates and logs confirmed violations to database.
        """
        frame_violations = []

        if tracked_boxes is None or len(tracked_boxes) == 0:
            return frame_violations

        # 0. Plate detection & association on interval frames
        if self.plate_assoc:
            self.plate_assoc.process_frame(frame, tracked_boxes, frame_idx=frame_idx)

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        for box in tracked_boxes:
            cls_id = int(box.cls[0].item())
            class_name = target_classes.get(cls_id, "").lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            tid = int(box.id[0].item()) if (hasattr(box, "id") and box.id is not None) else None
            conf = float(box.conf[0].item())

            # Retrieve associated plate if available
            plate_info = self.plate_assoc.get_plate_for_track(tid) if self.plate_assoc else None
            plate_text = "UNKNOWN"
            plate_conf = 0.0
            if plate_info and plate_info.get("text") and plate_info["text"] != "UNKNOWN" and plate_info.get("conf", 0.0) >= 0.30:
                plate_text = plate_info["text"]
                plate_conf = plate_info.get("conf", 0.0)

            # 1. Helmet check for motorcycle / two-wheeler
            if self.enable_helmet and self.helmet_detector and ("motorcycle" in class_name or "two-wheeler" in class_name):
                h_violation = self.helmet_detector.check_motorcycle(frame, (x1, y1, x2, y2), track_id=tid)
                if h_violation:
                    ev_path = str(self.evidence_dir / f"helmet_viol_{ts_str}_t{tid or frame_idx}.jpg")
                    crop_img = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    if crop_img.size > 0:
                        cv2.imwrite(ev_path, crop_img)
                    else:
                        cv2.imwrite(ev_path, frame)
                    h_violation.update({
                        "track_id": tid,
                        "vehicle_class": class_name,
                        "evidence_path": ev_path,
                        "plate_text": plate_text,
                        "plate_conf": plate_conf
                    })
                    frame_violations.append(h_violation)
                    if self.db:
                        self.db.insert_event(
                            event_type="NO_HELMET",
                            track_id=tid,
                            vehicle_class=class_name,
                            confidence=round(h_violation["confidence"], 2),
                            plate_text=plate_text,
                            evidence_path=ev_path,
                            review_status="pending",
                            prevent_duplicates=True
                        )

            # 2. Wrong-Way movement check
            if self.enable_wrong_way and self.wrong_way_detector and tid is not None:
                ww_violation = self.wrong_way_detector.update(tid, (cx, cy))
                if ww_violation:
                    ev_path = str(self.evidence_dir / f"wrongway_viol_{ts_str}_t{tid}.jpg")
                    cv2.imwrite(ev_path, frame)
                    ww_violation.update({
                        "vehicle_class": class_name,
                        "evidence_path": ev_path,
                        "confidence": conf,
                        "plate_text": plate_text,
                        "plate_conf": plate_conf
                    })
                    frame_violations.append(ww_violation)
                    if self.db:
                        self.db.insert_event(
                            event_type="WRONG_WAY",
                            track_id=tid,
                            vehicle_class=class_name,
                            confidence=round(conf, 2),
                            plate_text=plate_text,
                            evidence_path=ev_path,
                            review_status="pending",
                            prevent_duplicates=True
                        )

            # 3. Virtual Red-Light / Stop-Line check
            if self.enable_red_light and self.red_light_detector and tid is not None:
                prev_center = self._prev_centers.get(tid)
                self._prev_centers[tid] = (cx, cy)
                if prev_center is not None:
                    rl_violation = self.red_light_detector.check_crossing(tid, prev_center, (cx, cy))
                    if rl_violation:
                        ev_path = str(self.evidence_dir / f"redlight_viol_{ts_str}_t{tid}.jpg")
                        cv2.imwrite(ev_path, frame)
                        rl_violation.update({
                            "vehicle_class": class_name,
                            "evidence_path": ev_path,
                            "confidence": conf,
                            "plate_text": plate_text,
                            "plate_conf": plate_conf
                        })
                        frame_violations.append(rl_violation)
                        if self.db:
                            self.db.insert_event(
                                event_type="RED_LIGHT",
                                track_id=tid,
                                vehicle_class=class_name,
                                confidence=round(conf, 2),
                                plate_text=plate_text,
                                evidence_path=ev_path,
                                review_status="pending",
                                prevent_duplicates=True
                            )
            elif tid is not None:
                self._prev_centers[tid] = (cx, cy)

        self.violations_logged.extend(frame_violations)
        return frame_violations


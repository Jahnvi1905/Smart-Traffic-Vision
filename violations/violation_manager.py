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
from typing import Any, Dict, List, Optional, Tuple
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
    Avoids duplicate violation records per tracked vehicle via configurable cooldown.
    """

    def __init__(
        self,
        model_path: str = "models/helmet_detector.pt",
        conf_thresh: float = 0.20,
        cooldown_seconds: float = 10.0
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))
        self.model = None
        self.reported_tracks = set()  # track_ids that already had violation logged
        self.last_reported: Dict[Any, float] = {}  # track_id or spatial key -> timestamp
        self.class_map = {0: "Helmet", 1: "fd", 2: "motorcycle", 3: "nohelmet", 4: "platenumber"}
        # Dedicated spatial tracking state across consecutive frames
        self._active_tracks: Dict[int, Dict[str, Any]] = {}
        self._next_track_id: int = 9001
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

    def check_motorcycle(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        track_id: Optional[int] = None,
        current_time: float = 0.0
    ) -> Optional[Dict]:
        """
        Inspects rider region of a detected motorcycle for helmet / nohelmet.
        Returns violation dict if NO_HELMET is confirmed, None otherwise.
        Respects track_id and spatial cooldown to prevent duplicate alerts.
        """
        if self.model is None or frame is None:
            return None

        x1, y1, x2, y2 = bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        key = track_id if track_id is not None else f"loc_{int(cx / 80)}_{int(cy / 80)}"

        # Cooldown check
        if key in self.last_reported and (current_time - self.last_reported[key]) < self.cooldown_seconds:
            return None

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
            helmet_conf = 0.0

            for b in boxes:
                cid = int(b.cls[0].item())
                cname = self.class_map.get(cid, "").lower()
                conf = float(b.conf[0].item())
                if "nohelmet" in cname or cid == 3:
                    has_nohelmet = True
                    nohelmet_conf = max(nohelmet_conf, conf)
                elif "helmet" in cname or cid == 0:
                    has_helmet = True
                    helmet_conf = max(helmet_conf, conf)

            # If no-helmet is detected without a confirmed helmet
            if has_nohelmet and not has_helmet:
                self.last_reported[key] = current_time
                if track_id is not None:
                    self.reported_tracks.add(track_id)
                reason = f"Rider without helmet detected with model confidence {nohelmet_conf:.2f}"
                print(f"[HELMET] VIOLATION: Track #{track_id or 'untracked'} -> {reason}")
                return {
                    "event_type": "NO_HELMET",
                    "confidence": nohelmet_conf,
                    "bbox": (x1, y1, x2, y2),
                    "rider_box": (rx1, ry1, rx2, ry2),
                    "crop": rider_crop,
                    "reason": reason
                }
            elif has_helmet:
                print(f"[HELMET] Compliant: Track #{track_id or 'untracked'} wearing helmet (confidence {helmet_conf:.2f})")

        except Exception as e:
            print(f"[WARN] Error running helmet detection: {e}")

        return None

    def detect_frame_violations(
        self,
        frame: np.ndarray,
        current_time: float = 0.0,
        tracked_boxes: Any = None,
        target_classes: Optional[Dict[int, str]] = None
    ) -> List[Dict]:
        """
        Runs dedicated spatial helmet detection across the full frame.
        Self-bootstraps motorcycle detection and rider region localization.
        Associates motorcycles, helmets, and nohelmets spatially.
        Supports multi-rider frames without false cross-rider suppression.
        Correlates with tracked_boxes (ByteTrack) when available; otherwise
        uses an internal spatial motorcycle tracker for stable deduplication.
        """
        if self.model is None or frame is None or frame.size == 0:
            return []

        fh, fw = frame.shape[:2]
        try:
            results = self.model(frame, conf=self.conf_thresh, verbose=False)
            if not results or not results[0].boxes:
                return []

            boxes = results[0].boxes
            motorcycles = []
            helmets = []
            nohelmets = []

            for b in boxes:
                cid = int(b.cls[0].item())
                cname = self.class_map.get(cid, "").lower()
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                # Constrain to frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(fw, x2), min(fh, y2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                item = {"box": (x1, y1, x2, y2), "conf": conf, "center": (cx, cy)}
                if cid == 2 or "motorcycle" in cname:
                    motorcycles.append(item)
                elif cid == 0 or "helmet" == cname:
                    helmets.append(item)
                elif cid == 3 or "nohelmet" in cname:
                    nohelmets.append(item)

            # Extract any external two-wheeler tracks from vehicle detector / ByteTrack
            external_two_wheelers = []
            if tracked_boxes is not None and len(tracked_boxes) > 0 and target_classes:
                for tbox in tracked_boxes:
                    t_cls = int(tbox.cls[0].item())
                    t_name = target_classes.get(t_cls, "").lower()
                    if "motorcycle" in t_name or "two-wheeler" in t_name:
                        tx1, ty1, tx2, ty2 = map(int, tbox.xyxy[0].tolist())
                        tid = int(tbox.id[0].item()) if (hasattr(tbox, "id") and tbox.id is not None) else None
                        external_two_wheelers.append({
                            "box": (tx1, ty1, tx2, ty2),
                            "center": ((tx1 + tx2) // 2, (ty1 + ty2) // 2),
                            "track_id": tid
                        })

            # Clean up stale active tracks (> 3.0 seconds since last seen)
            stale_keys = [
                k for k, v in self._active_tracks.items()
                if (current_time - v.get("last_seen", current_time)) > 3.0
            ]
            for k in stale_keys:
                del self._active_tracks[k]

            # Assign track IDs to motorcycles
            for moto in motorcycles:
                mx1, my1, mx2, my2 = moto["box"]
                mcx, mcy = moto["center"]
                assigned_tid = None

                # 1. Check match with external ByteTrack two-wheelers (IoU or center distance)
                for ext in external_two_wheelers:
                    if ext["track_id"] is not None:
                        ecx, ecy = ext["center"]
                        if abs(mcx - ecx) < 80 and abs(mcy - ecy) < 80:
                            assigned_tid = ext["track_id"]
                            break

                # 2. If no external track match, match with internal spatial tracker
                if assigned_tid is None:
                    best_match_id = None
                    min_dist = float("inf")
                    for tid, track_info in self._active_tracks.items():
                        tcx, tcy = track_info["center"]
                        dist = ((mcx - tcx) ** 2 + (mcy - tcy) ** 2) ** 0.5
                        if dist < 90 and dist < min_dist:
                            min_dist = dist
                            best_match_id = tid

                    if best_match_id is not None:
                        assigned_tid = best_match_id
                    else:
                        assigned_tid = self._next_track_id
                        self._next_track_id += 1

                # Update active track record
                self._active_tracks[assigned_tid] = {
                    "box": moto["box"],
                    "center": (mcx, mcy),
                    "last_seen": current_time
                }
                moto["track_id"] = assigned_tid

            confirmed_violations = []

            # Case A: If motorcycles were detected, associate heads spatially
            for moto in motorcycles:
                tid = moto["track_id"]
                mx1, my1, mx2, my2 = moto["box"]
                mw = mx2 - mx1
                mh = my2 - my1

                # Rider head search region: top 60% of motorcycle box, expanded slightly
                rx1 = max(0, mx1 - int(0.15 * mw))
                ry1 = max(0, my1 - int(0.25 * mh))
                rx2 = min(fw, mx2 + int(0.15 * mw))
                ry2 = min(fh, my1 + int(0.65 * mh))

                # Find heads whose centers lie within this rider region
                m_nohelmets = [
                    nh for nh in nohelmets
                    if rx1 <= nh["center"][0] <= rx2 and ry1 <= nh["center"][1] <= ry2
                ]
                m_helmets = [
                    h for h in helmets
                    if rx1 <= h["center"][0] <= rx2 and ry1 <= h["center"][1] <= ry2
                ]

                for nh in m_nohelmets:
                    nh_cx, nh_cy = nh["center"]
                    # Check if there is an overlapping Helmet box on this EXACT same head (competing classes)
                    coincident_helmets = [
                        h for h in m_helmets
                        if abs(h["center"][0] - nh_cx) < 40 and abs(h["center"][1] - nh_cy) < 40
                    ]

                    # If there are competing boxes on the same head, require nohelmet_conf > helmet_conf
                    if coincident_helmets:
                        max_h_conf = max(h["conf"] for h in coincident_helmets)
                        if nh["conf"] <= max_h_conf:
                            print(f"[HELMET] Compliant: Track #{tid} rider head has helmet conf {max_h_conf:.2f} >= nohelmet conf {nh['conf']:.2f}")
                            continue

                    # Cooldown check for this track_id
                    key = tid if tid is not None else f"loc_{int(mcx / 80)}_{int(mcy / 80)}"
                    if key in self.last_reported and (current_time - self.last_reported[key]) < self.cooldown_seconds:
                        continue

                    self.last_reported[key] = current_time
                    if tid is not None:
                        self.reported_tracks.add(tid)

                    # Extract context crop
                    cx1 = max(0, min(mx1, rx1))
                    cy1 = max(0, min(my1, ry1))
                    cx2 = min(fw, max(mx2, rx2))
                    cy2 = min(fh, max(my2, ry2))
                    crop = frame[cy1:cy2, cx1:cx2].copy()

                    reason = f"Rider without helmet detected with confidence {nh['conf']:.2f}"
                    print(f"[HELMET] VIOLATION: Track #{tid} -> {reason}")

                    confirmed_violations.append({
                        "event_type": "NO_HELMET",
                        "confidence": nh["conf"],
                        "bbox": moto["box"],
                        "rider_box": nh["box"],
                        "crop": crop,
                        "track_id": tid,
                        "reason": reason
                    })

            # Case B: If no motorcycles were detected, but isolated nohelmet heads were detected
            if not motorcycles and nohelmets:
                for nh in nohelmets:
                    nh_cx, nh_cy = nh["center"]
                    coincident_helmets = [
                        h for h in helmets
                        if abs(h["center"][0] - nh_cx) < 40 and abs(h["center"][1] - nh_cy) < 40
                    ]
                    if coincident_helmets:
                        max_h_conf = max(h["conf"] for h in coincident_helmets)
                        if nh["conf"] <= max_h_conf:
                            continue

                    key = f"loc_{int(nh_cx / 80)}_{int(nh_cy / 80)}"
                    if key in self.last_reported and (current_time - self.last_reported[key]) < self.cooldown_seconds:
                        continue

                    self.last_reported[key] = current_time
                    nx1, ny1, nx2, ny2 = nh["box"]
                    # Synthesize context box downwards
                    synth_bbox = (
                        max(0, nx1 - 30),
                        max(0, ny1 - 20),
                        min(fw, nx2 + 30),
                        min(fh, ny2 + int((ny2 - ny1) * 3.5))
                    )
                    crop = frame[synth_bbox[1]:synth_bbox[3], synth_bbox[0]:synth_bbox[2]].copy()
                    reason = f"Rider head without helmet detected with confidence {nh['conf']:.2f}"
                    print(f"[HELMET] VIOLATION: Untracked head at ({nh_cx}, {nh_cy}) -> {reason}")

                    confirmed_violations.append({
                        "event_type": "NO_HELMET",
                        "confidence": nh["conf"],
                        "bbox": synth_bbox,
                        "rider_box": nh["box"],
                        "crop": crop,
                        "track_id": None,
                        "reason": reason
                    })

            return confirmed_violations

        except Exception as e:
            print(f"[WARN] Error in dedicated helmet frame detection: {e}")
            return []


class WrongWayDetector:
    """
    Detects wrong-way vehicle movement based on multi-frame centroid displacement.
    Requires consistent trajectory history and rejects stationary/noisy tracks.
    Supports configurable expected traffic direction and cooldown.
    """

    def __init__(
        self,
        expected_direction: str = "down",  # "down" (+y), "up" (-y), "left" (-x), "right" (+x)
        min_observations: int = 8,
        min_displacement_px: int = 40,
        consistency_ratio: float = 0.75,
        cooldown_seconds: float = 10.0
    ):
        self.expected_direction = expected_direction.lower()
        self.min_observations = max(3, int(min_observations))
        self.min_displacement_px = max(10, int(min_displacement_px))
        self.consistency_ratio = float(consistency_ratio)
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))

        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.reported_tracks = set()  # track_ids already flagged
        self.last_reported: Dict[int, float] = {}

    def update(
        self,
        track_id: int,
        center: Tuple[int, int],
        current_time: float = 0.0
    ) -> Optional[Dict]:
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

        # Cooldown check
        if track_id in self.last_reported and (current_time - self.last_reported[track_id]) < self.cooldown_seconds:
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
            # Expected +x movement; wrong-way is moving left (-x)
            if dx < -self.min_displacement_px:
                step_matches = sum(1 for sdx, _ in steps if sdx < 0)
                if step_matches / len(steps) >= self.consistency_ratio:
                    is_wrong_way = True

        elif self.expected_direction == "left":
            # Expected -x movement; wrong-way is moving right (+x)
            if dx > self.min_displacement_px:
                step_matches = sum(1 for sdx, _ in steps if sdx > 0)
                if step_matches / len(steps) >= self.consistency_ratio:
                    is_wrong_way = True

        if is_wrong_way:
            self.reported_tracks.add(track_id)
            self.last_reported[track_id] = current_time
            reason = (
                f"Vehicle #{track_id} moved opposite to configured direction '{self.expected_direction}' "
                f"({dx:+d}x, {dy:+d}y, total displacement {total_dist:.1f}px)"
            )
            print(f"[WRONG_WAY] VIOLATION: {reason}")
            return {
                "event_type": "WRONG_WAY",
                "track_id": track_id,
                "displacement": total_dist,
                "direction": f"{dx:+d}x, {dy:+d}y",
                "center": curr_pt,
                "reason": reason
            }

        return None


class RedLightDetector:
    """
    Virtual Stop-Line Red-Light violation detector.
    Note: Real optical signal state classification is not performed because sample
    camera footage lacks traffic signal lamps in the camera FOV; a virtual signal state
    and virtual stop-line are provided for policy verification.
    """

    def __init__(
        self,
        stop_line: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 1100), (1080, 1100)),
        signal_state: str = "GREEN",  # "RED", "GREEN"
        cooldown_seconds: float = 10.0
    ):
        self.stop_line_start = tuple(map(int, stop_line[0]))
        self.stop_line_end = tuple(map(int, stop_line[1]))
        self.signal_state = signal_state.upper()
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))
        self.reported_tracks = set()
        self.last_reported: Dict[int, float] = {}

    def set_signal_state(self, state: str):
        """Sets the virtual signal state ('RED' or 'GREEN')."""
        self.signal_state = state.upper()

    def adjust_line(self, width: int, height: int, line_y_pct: float = 60.0):
        """Scales virtual stop line to actual video dimensions."""
        calc_y = int(height * (line_y_pct / 100.0))
        self.stop_line_start = (0, calc_y)
        self.stop_line_end = (width, calc_y)

    def check_crossing(
        self,
        track_id: int,
        prev_center: Tuple[int, int],
        curr_center: Tuple[int, int],
        current_time: float = 0.0
    ) -> Optional[Dict]:
        """
        Checks if vehicle crossed stop-line while signal state is RED.
        Inactive if signal state is not RED.
        """
        if self.signal_state != "RED":
            return None

        if track_id is not None and track_id in self.last_reported and (current_time - self.last_reported[track_id]) < self.cooldown_seconds:
            return None

        # Check line crossing
        py1, py2 = prev_center[1], curr_center[1]
        line_y = (self.stop_line_start[1] + self.stop_line_end[1]) / 2.0

        crossed = (py1 <= line_y <= py2) or (py2 <= line_y <= py1)
        if crossed:
            if track_id is not None:
                self.reported_tracks.add(track_id)
                self.last_reported[track_id] = current_time
            reason = (
                f"Vehicle #{track_id or 'untracked'} crossed virtual stop-line at y={int(line_y)} "
                f"while virtual signal was RED (policy-based enforcement; no optical signal classifier)"
            )
            print(f"[RED_LIGHT] VIOLATION: {reason}")
            return {
                "event_type": "RED_LIGHT",
                "track_id": track_id,
                "stop_line_y": int(line_y),
                "center": curr_center,
                "reason": reason
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
        signal_state: str = "GREEN",
        cooldown_seconds: float = 10.0,
        plate_interval: int = 5,
    ):
        self.db = database_manager
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self.enable_helmet = enable_helmet
        self.enable_wrong_way = enable_wrong_way
        self.enable_red_light = enable_red_light
        self.enable_plate_assoc = enable_plate_assoc
        self.expected_direction = expected_direction
        self.signal_state = signal_state.upper()
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))

        self.helmet_detector = HelmetViolationDetector(
            model_path=helmet_model_path,
            cooldown_seconds=cooldown_seconds
        ) if enable_helmet else None

        self.wrong_way_detector = WrongWayDetector(
            expected_direction=expected_direction,
            cooldown_seconds=cooldown_seconds
        ) if enable_wrong_way else None

        self.red_light_detector = RedLightDetector(
            signal_state=signal_state,
            cooldown_seconds=cooldown_seconds
        ) if enable_red_light else None

        self.plate_assoc = None
        if enable_plate_assoc and os.path.exists(plate_model_path):
            self.plate_assoc = PlateAssociationManager(
                plate_model_path=plate_model_path,
                detection_interval=plate_interval
            )

        self._prev_centers: Dict[int, Tuple[int, int]] = {}
        self.violations_logged = []

    def _create_annotated_evidence(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        event_type: str,
        track_id: Optional[int],
        vehicle_class: str,
        confidence: float,
        plate_text: Optional[str] = None,
        reason: Optional[str] = None
    ) -> np.ndarray:
        """
        Creates a high-contrast annotated evidence frame with bounding boxes,
        violation banner, metadata tags, and timestamp watermark.
        """
        annotated = frame.copy()
        fh, fw = annotated.shape[:2]
        x1, y1, x2, y2 = bbox

        viol_color = (0, 0, 255)  # Bright Red in BGR

        # 1. Bounding box around vehicle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), viol_color, 3)

        # 2. Top Header Banner
        banner_h = 42
        cv2.rectangle(annotated, (0, 0), (fw, banner_h), (16, 20, 28), cv2.FILLED)
        cv2.line(annotated, (0, banner_h), (fw, banner_h), viol_color, 2)

        type_label = event_type.replace('_', ' ')
        header_text = f"VIOLATION ALERT: {type_label} | ID: #{track_id if track_id is not None else 'N/A'} {vehicle_class.upper()} ({confidence*100:.0f}%)"
        cv2.putText(annotated, header_text, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        # 3. Label tag above vehicle
        tag = f"{vehicle_class.upper()} #{track_id if track_id is not None else 'N/A'}"
        if plate_text and plate_text not in ("", "UNKNOWN"):
            tag += f" [{plate_text}]"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tag_y1 = max(banner_h + 5, y1 - th - 8)
        tag_y2 = max(banner_h + th + 12, y1)
        tag_x1 = max(0, x1)
        tag_x2 = min(fw, x1 + tw + 10)
        cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), viol_color, cv2.FILLED)
        cv2.putText(annotated, tag, (tag_x1 + 4, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Reason and timestamp banner at bottom
        footer_h = 32
        cv2.rectangle(annotated, (0, fh - footer_h), (fw, fh), (16, 20, 28), cv2.FILLED)
        cv2.line(annotated, (0, fh - footer_h), (fw, fh - footer_h), (60, 75, 95), 1)

        ts_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer_text = f"{ts_text} | Evidence Record"
        if reason:
            footer_text += f" | {reason}"
        if len(footer_text) > 105:
            footer_text = footer_text[:102] + "..."
        cv2.putText(annotated, footer_text, (18, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 225, 235), 1, cv2.LINE_AA)

        return annotated

    def process_frame(
        self,
        frame: np.ndarray,
        tracked_boxes,
        target_classes: Dict[int, str],
        frame_idx: int = 0,
        timestamp_s: float = 0.0
    ) -> List[Dict]:
        """
        Evaluates active violation rules for current frame detections and tracks.
        Associates license plates and logs confirmed violations to database.
        """
        frame_violations = []
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_time = timestamp_s if timestamp_s > 0.0 else float(frame_idx)

        # 0. Plate detection & association on interval frames
        if self.plate_assoc and tracked_boxes is not None and len(tracked_boxes) > 0:
            self.plate_assoc.process_frame(frame, tracked_boxes, frame_idx=frame_idx)

        # 1. Dedicated Helmet Detection (Self-bootstrapped, spatial multi-rider support)
        if self.enable_helmet and self.helmet_detector:
            h_violations = self.helmet_detector.detect_frame_violations(
                frame,
                current_time=current_time,
                tracked_boxes=tracked_boxes,
                target_classes=target_classes
            )
            for h_violation in h_violations:
                tid = h_violation.get("track_id")
                # Retrieve associated plate if available
                plate_text = "UNKNOWN"
                plate_conf = 0.0
                if tid is not None and self.plate_assoc:
                    plate_info = self.plate_assoc.get_plate_for_track(tid)
                    if plate_info and plate_info.get("text") and plate_info["text"] != "UNKNOWN" and plate_info.get("conf", 0.0) >= 0.30:
                        plate_text = plate_info["text"]
                        plate_conf = plate_info.get("conf", 0.0)

                ev_path = str(self.evidence_dir / f"helmet_viol_{ts_str}_t{tid or frame_idx}.jpg")
                annotated_ev = self._create_annotated_evidence(
                    frame,
                    h_violation["bbox"],
                    "NO_HELMET",
                    tid,
                    "motorcycle",
                    h_violation["confidence"],
                    plate_text=plate_text,
                    reason=h_violation.get("reason")
                )
                # Highlight the rider's unhelmeted head region if available
                if "rider_box" in h_violation:
                    rx1, ry1, rx2, ry2 = h_violation["rider_box"]
                    cv2.rectangle(annotated_ev, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)
                    cv2.putText(annotated_ev, "NO HELMET", (rx1, max(20, ry1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

                cv2.imwrite(ev_path, annotated_ev)

                h_violation.update({
                    "vehicle_class": "motorcycle",
                    "evidence_path": ev_path,
                    "plate_text": plate_text,
                    "plate_conf": plate_conf
                })
                frame_violations.append(h_violation)
                if self.db:
                    self.db.insert_event(
                        event_type="NO_HELMET",
                        track_id=tid,
                        vehicle_class="motorcycle",
                        confidence=round(h_violation["confidence"], 2),
                        plate_text=plate_text,
                        evidence_path=ev_path,
                        review_status="pending",
                        prevent_duplicates=True
                    )

        if tracked_boxes is None or len(tracked_boxes) == 0:
            return frame_violations

        # 2. Tracking-dependent checks for tracked_boxes (Wrong-Way, Virtual Red-Light)
        for box in tracked_boxes:
            cls_id = int(box.cls[0].item())
            class_name = target_classes.get(cls_id, "").lower() if target_classes else ""
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

            # 2. Wrong-Way movement check
            if self.enable_wrong_way and self.wrong_way_detector and tid is not None:
                ww_violation = self.wrong_way_detector.update(
                    tid,
                    (cx, cy),
                    current_time=current_time
                )
                if ww_violation:
                    ev_path = str(self.evidence_dir / f"wrongway_viol_{ts_str}_t{tid}.jpg")
                    annotated_ev = self._create_annotated_evidence(
                        frame,
                        (x1, y1, x2, y2),
                        "WRONG_WAY",
                        tid,
                        class_name,
                        conf,
                        plate_text=plate_text,
                        reason=ww_violation.get("reason")
                    )
                    cv2.imwrite(ev_path, annotated_ev)

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
                    rl_violation = self.red_light_detector.check_crossing(
                        tid,
                        prev_center,
                        (cx, cy),
                        current_time=current_time
                    )
                    if rl_violation:
                        ev_path = str(self.evidence_dir / f"redlight_viol_{ts_str}_t{tid}.jpg")
                        annotated_ev = self._create_annotated_evidence(
                            frame,
                            (x1, y1, x2, y2),
                            "RED_LIGHT",
                            tid,
                            class_name,
                            conf,
                            plate_text=plate_text,
                            reason=rl_violation.get("reason")
                        )
                        cv2.imwrite(ev_path, annotated_ev)

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


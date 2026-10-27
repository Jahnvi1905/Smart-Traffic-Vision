import cv2
import numpy as np

# Target classes for vehicles matching custom YOLO model
DEFAULT_TARGET_CLASSES = {
    0: "bus",
    1: "car",
    2: "three-wheeler",
    3: "truck",
    4: "two-wheeler",
    5: "van"
}

# Colors for HUD and counts (BGR)
CLASS_COLORS = {
    "car": (0, 255, 0),           # Vibrant Green
    "motorcycle": (255, 200, 0),  # Amber / Cyan-ish
    "two-wheeler": (255, 200, 0), # Amber
    "three-wheeler": (0, 215, 255),# Gold / Yellow-Orange
    "bus": (0, 165, 255),         # Vibrant Orange
    "truck": (200, 50, 255),      # Magenta / Violet
    "van": (255, 105, 180)        # Hot Pink
}


def ccw(a, b, c):
    """Checks counter-clockwise orientation of three points."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(p, a, b):
    """Checks if point p lies on segment ab."""
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def segments_intersect(p1, p2, p3, p4):
    """
    Returns True if line segment p1-p2 intersects line segment p3-p4.
    p1, p2: trajectory of vehicle (prev_center -> curr_center)
    p3, p4: counting line (line_start -> line_end)
    """
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Handle colinear/endpoint touches
    if d1 == 0 and on_segment(p1, p3, p4):
        return True
    if d2 == 0 and on_segment(p2, p3, p4):
        return True
    if d3 == 0 and on_segment(p3, p1, p2):
        return True
    if d4 == 0 and on_segment(p4, p1, p2):
        return True

    return False


class VehicleCounter:
    """
    Virtual-line vehicle counter supporting ByteTrack tracking IDs or
    lightweight centroid-distance tracking fallback.
    Maintains separate counts per vehicle category without double-counting.
    """

    def __init__(
        self,
        line_start: tuple = (0, 1000),
        line_end: tuple = (1080, 1000),
        target_classes: dict = None
    ):
        """
        Args:
            line_start: (x, y) start point of the virtual counting line.
            line_end: (x, y) end point of the virtual counting line.
            target_classes: Dictionary mapping class ID to class name.
        """
        self.line_start = tuple(map(int, line_start))
        self.line_end = tuple(map(int, line_end))
        self.target_classes = target_classes or DEFAULT_TARGET_CLASSES

        # Count tracking data
        self.class_counts = {name: 0 for name in self.target_classes.values()}
        self.counted_ids = set()
        self.prev_positions = {}  # track_id -> (cx, cy)
        self.current_visible_count = 0
        self.current_visible_classes = {name: 0 for name in self.target_classes.values()}
        self.telemetry = []  # per-frame count telemetry for player synchronization

        # Fallback centroid tracker when box.id is None (ByteTrack disabled)
        self._fallback_next_id = 10001
        self._fallback_tracks = {}  # id -> {'center': (cx, cy), 'class_id': int, 'missed': int}

    @property
    def total_count(self) -> int:
        return sum(self.class_counts.values())

    def reset(self):
        """Resets all counts, telemetry, and active tracks."""
        self.class_counts = {name: 0 for name in self.target_classes.values()}
        self.counted_ids.clear()
        self.prev_positions.clear()
        self._fallback_tracks.clear()
        self._fallback_next_id = 10001
        self.current_visible_count = 0
        self.current_visible_classes = {name: 0 for name in self.target_classes.values()}
        self.telemetry.clear()

    def adjust_line(self, frame_width: int, frame_height: int, line_pos_pct: float = None):
        """
        Adjusts virtual counting line to match actual frame dimensions.
        """
        lx1, ly1 = self.line_start
        lx2, ly2 = self.line_end
        if line_pos_pct is not None:
            calc_y = int(frame_height * (line_pos_pct / 100.0))
            self.line_start = (0, calc_y)
            self.line_end = (frame_width, calc_y)
        else:
            # Clamp to frame boundary
            self.line_start = (max(0, min(lx1, frame_width)), max(0, min(ly1, frame_height)))
            self.line_end = (max(0, min(lx2, frame_width)), max(0, min(ly2, frame_height)))

    def _assign_fallback_ids(self, detections: list) -> list:
        """
        Assigns persistent pseudo-track IDs based on nearest centroid distance.
        detections: list of dicts with 'curr_center', 'cls_id', 'box'
        """
        assigned = []
        unmatched = list(range(len(detections)))

        # Update existing fallback tracks
        for tid, tinfo in list(self._fallback_tracks.items()):
            tx, ty = tinfo['center']
            tcls = tinfo['class_id']
            best_idx = None
            best_dist = 160.0  # Search radius in pixels

            for idx in unmatched:
                det = detections[idx]
                if det['cls_id'] == tcls:
                    cx, cy = det['curr_center']
                    dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx

            if best_idx is not None:
                cx, cy = detections[best_idx]['curr_center']
                self._fallback_tracks[tid] = {'center': (cx, cy), 'class_id': tcls, 'missed': 0}
                assigned.append((best_idx, tid))
                unmatched.remove(best_idx)
            else:
                tinfo['missed'] += 1
                if tinfo['missed'] > 15:
                    del self._fallback_tracks[tid]

        # Create new tracks for unmatched detections
        for idx in unmatched:
            det = detections[idx]
            tid = self._fallback_next_id
            self._fallback_next_id += 1
            self._fallback_tracks[tid] = {'center': det['curr_center'], 'class_id': det['cls_id'], 'missed': 0}
            assigned.append((idx, tid))

        assigned_map = dict(assigned)
        return [assigned_map.get(i) for i in range(len(detections))]

    def update(self, tracked_boxes, frame_idx: int = 0, timestamp_s: float = 0.0) -> list:
        """
        Updates the counter with current frame's detected/tracked boxes.
        Works seamlessly with ByteTrack IDs OR internal centroid tracker fallback.
        Maintains both cumulative line crossings and currently visible vehicles.
        Records per-frame telemetry for playback synchronization.

        Args:
            tracked_boxes: List of Ultralytics box objects or dicts with cls, conf, id, xyxy.
            frame_idx: Optional frame index for telemetry synchronization.
            timestamp_s: Optional video timestamp in seconds for telemetry.

        Returns:
            list: List of newly counted track IDs in this frame.
        """
        newly_counted = []
        if tracked_boxes is None or len(tracked_boxes) == 0:
            self.current_visible_count = 0
            self.current_visible_classes = {name: 0 for name in self.target_classes.values()}
            self.telemetry.append({
                "frame_idx": int(frame_idx),
                "timestamp_s": round(float(timestamp_s), 3),
                "cumulative_total": int(self.total_count),
                "cumulative_classes": dict(self.class_counts),
                "visible_total": 0,
                "visible_classes": dict(self.current_visible_classes),
                "newly_counted": []
            })
            return newly_counted

        # Parse boxes
        parsed = []
        has_bytetrack_id = False
        for box in tracked_boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in self.target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            curr_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            tid = int(box.id[0].item()) if (hasattr(box, "id") and box.id is not None) else None
            if tid is not None:
                has_bytetrack_id = True

            parsed.append({
                "track_id": tid,
                "cls_id": cls_id,
                "class_name": self.target_classes[cls_id],
                "curr_center": curr_center,
                "bbox": (x1, y1, x2, y2)
            })

        # Update currently visible vehicles on screen
        self.current_visible_count = len(parsed)
        self.current_visible_classes = {name: 0 for name in self.target_classes.values()}
        for item in parsed:
            cname = item["class_name"]
            self.current_visible_classes[cname] = self.current_visible_classes.get(cname, 0) + 1

        if not parsed:
            self.telemetry.append({
                "frame_idx": int(frame_idx),
                "timestamp_s": round(float(timestamp_s), 3),
                "cumulative_total": int(self.total_count),
                "cumulative_classes": dict(self.class_counts),
                "visible_total": 0,
                "visible_classes": dict(self.current_visible_classes),
                "newly_counted": []
            })
            return newly_counted

        # If ByteTrack IDs were not provided, use centroid association fallback
        if not has_bytetrack_id:
            assigned_ids = self._assign_fallback_ids(parsed)
            for item, aid in zip(parsed, assigned_ids):
                item["track_id"] = aid

        # Check line crossing
        lx1, ly1 = self.line_start
        lx2, ly2 = self.line_end
        min_lx, max_lx = min(lx1, lx2), max(lx1, lx2)
        min_ly, max_ly = min(ly1, ly2), max(ly1, ly2)

        for item in parsed:
            track_id = item["track_id"]
            if track_id is None:
                continue

            class_name = item["class_name"]
            curr_center = item["curr_center"]
            cx, cy = curr_center

            if track_id in self.prev_positions:
                prev_center = self.prev_positions[track_id]
                px, py = prev_center

                if track_id not in self.counted_ids:
                    # 1. Standard segment intersection
                    crossed = segments_intersect(prev_center, curr_center, self.line_start, self.line_end)

                    # 2. Horizontal crossing check (if counting line is roughly horizontal)
                    if not crossed and abs(ly1 - ly2) <= 10:
                        line_y = (ly1 + ly2) / 2.0
                        # Trajectory crossed line_y and vehicle x was within line span
                        if (py <= line_y <= cy or cy <= line_y <= py):
                            avg_x = (px + cx) / 2.0
                            if (min_lx - 20) <= avg_x <= (max_lx + 20):
                                crossed = True

                    if crossed:
                        self.counted_ids.add(track_id)
                        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
                        newly_counted.append(track_id)
                        print(f"[COUNTER] Vehicle #{track_id} ({class_name}) crossed counting line! "
                              f"Total: {self.total_count} {self.class_counts}")

            self.prev_positions[track_id] = curr_center

        # Record per-frame telemetry for downstream synchronization
        self.telemetry.append({
            "frame_idx": int(frame_idx),
            "timestamp_s": round(float(timestamp_s), 3),
            "cumulative_total": int(self.total_count),
            "cumulative_classes": dict(self.class_counts),
            "visible_total": int(self.current_visible_count),
            "visible_classes": dict(self.current_visible_classes),
            "newly_counted": list(newly_counted)
        })

        return newly_counted

    def get_telemetry(self) -> list:
        """Returns the recorded frame-by-frame telemetry list."""
        return self.telemetry

    def draw(self, frame):
        """
        Renders the virtual counting line and a clear, modern HUD dashboard.
        Clearly separates cumulative line crossings from currently visible vehicles.
        """
        h, w = frame.shape[:2]

        # 1. Draw Virtual Counting Line with glow/accent effect
        line_color = (0, 220, 255)  # Bright amber-gold / cyan
        cv2.line(frame, self.line_start, self.line_end, (20, 20, 20), 5, lineType=cv2.LINE_AA)
        cv2.line(frame, self.line_start, self.line_end, line_color, 3, lineType=cv2.LINE_AA)

        # Counting Line Label
        mid_x = int((self.line_start[0] + self.line_end[0]) / 2)
        mid_y = int((self.line_start[1] + self.line_end[1]) / 2) - 10
        cv2.putText(
            frame,
            "--- VIRTUAL COUNTING LINE ---",
            (mid_x - 140, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 240, 255),
            2,
            lineType=cv2.LINE_AA
        )

        # 2. Draw modern HUD Dashboard (top-left corner)
        hud_w = 380
        hud_h = max(240, 125 + len(self.class_counts) * 26)
        hud_x, hud_y = 20, 20

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (16, 20, 26), cv2.FILLED)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Dashboard border & header accent
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (70, 85, 105), 2)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + 38), (35, 48, 68), cv2.FILLED)

        # Header Title
        cv2.putText(
            frame,
            "SMART TRAFFIC VISION - HUD",
            (hud_x + 14, hud_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 230, 255),
            2,
            lineType=cv2.LINE_AA
        )

        # Summary Metrics: Cumulative Crossings vs In-Frame Visible
        summary_text = f"Crossed: {self.total_count}  |  Visible: {self.current_visible_count}"
        cv2.putText(
            frame,
            summary_text,
            (hud_x + 14, hud_y + 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )

        # Subtle divider
        cv2.line(frame, (hud_x + 12, hud_y + 80), (hud_x + hud_w - 12, hud_y + 80), (60, 75, 95), 1)

        # Subhead: Cumulative Crossings by Category
        cv2.putText(
            frame,
            "CUMULATIVE CROSSINGS (IN-FRAME ACTIVE):",
            (hud_x + 14, hud_y + 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 175, 195),
            1,
            lineType=cv2.LINE_AA
        )

        # Category Counts
        y_offset = hud_y + 122
        spacing = 26
        for class_name, count in self.class_counts.items():
            color = CLASS_COLORS.get(class_name, (200, 200, 200))
            vis = self.current_visible_classes.get(class_name, 0)
            # Small color badge
            cv2.circle(frame, (hud_x + 20, y_offset - 5), 5, color, cv2.FILLED)
            # Text line with cumulative and active count
            stat_line = f"{class_name.capitalize():<14}: {count}"
            cv2.putText(
                frame,
                stat_line,
                (hud_x + 32, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                1,
                lineType=cv2.LINE_AA
            )
            # If vehicles are currently visible, show live indicator
            if vis > 0:
                cv2.putText(
                    frame,
                    f"[{vis} live]",
                    (hud_x + 235, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (0, 255, 180),
                    1,
                    lineType=cv2.LINE_AA
                )
            y_offset += spacing

        return frame

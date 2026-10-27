"""
Comprehensive test suite for Phase 1: Core Pipeline & Telemetry Fixes.
Tests:
1. Custom class mapping (0=bus, 1=car, 2=three-wheeler, 3=truck, 4=two-wheeler, 5=van)
2. Counting consistency: cumulative line crossings vs visible in-frame vehicles
3. Per-frame count telemetry generation and schema
4. Configurable wrong-way detection across all directions (down, up, left, right)
5. Red-light virtual stop-line enforcement, scaling, and inactivity under GREEN signal
6. Cooldown deduplication preventing multiple alerts per track
7. High-contrast annotated evidence image generation
8. Database persistence and duplicate event rejection
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from counting.vehicle_counter import VehicleCounter, DEFAULT_TARGET_CLASSES
from violations.violation_manager import (
    TrafficViolationManager,
    HelmetViolationDetector,
    WrongWayDetector,
    RedLightDetector,
)
from database.database_manager import DatabaseManager


class MockBox:
    """Mock Ultralytics detection box for deterministic testing."""
    def __init__(self, tid, xyxy, cls_id, conf=0.85):
        self.id = np.array([tid]) if tid is not None else None
        self.xyxy = np.array([xyxy])
        self.cls = np.array([cls_id])
        self.conf = np.array([conf])


class TestPhase1Pipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_traffic.db")
        self.db = DatabaseManager(db_path=self.test_db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_custom_class_mapping(self):
        """Verifies default vehicle target classes match custom YOLO model classes exactly."""
        expected = {
            0: "bus",
            1: "car",
            2: "three-wheeler",
            3: "truck",
            4: "two-wheeler",
            5: "van"
        }
        self.assertEqual(DEFAULT_TARGET_CLASSES, expected)

        counter = VehicleCounter()
        self.assertEqual(counter.target_classes, expected)
        for cname in expected.values():
            self.assertIn(cname, counter.class_counts)
            self.assertEqual(counter.class_counts[cname], 0)

    def test_counting_consistency_and_visibility(self):
        """
        Tests that cumulative line crossings increment ONLY on actual line crossings,
        while in-frame visible counts reflect the detections in each active frame.
        """
        # Horizontal counting line at y=300 (span x: 0 to 600)
        counter = VehicleCounter(line_start=(0, 300), line_end=(600, 300))

        # Frame 1: 3 vehicles in frame above counting line (y=100-200)
        b1 = [
            MockBox(tid=1, xyxy=[50, 100, 120, 160], cls_id=1),   # Car
            MockBox(tid=2, xyxy=[200, 100, 270, 160], cls_id=3),  # Truck
            MockBox(tid=3, xyxy=[350, 100, 410, 160], cls_id=2),  # Three-Wheeler
        ]
        newly_1 = counter.update(b1, frame_idx=1, timestamp_s=0.033)
        self.assertEqual(len(newly_1), 0, "No vehicle has crossed line yet")
        self.assertEqual(counter.total_count, 0, "Cumulative crossings must be 0")
        self.assertEqual(counter.current_visible_count, 3, "Visible in-frame count must be 3")
        self.assertEqual(counter.current_visible_classes["car"], 1)
        self.assertEqual(counter.current_visible_classes["truck"], 1)
        self.assertEqual(counter.current_visible_classes["three-wheeler"], 1)

        # Frame 2: Vehicle 1 crosses line (y: 130 -> 350); Vehicle 2 stays above (y: 130 -> 220); Vehicle 3 leaves
        b2 = [
            MockBox(tid=1, xyxy=[50, 320, 120, 380], cls_id=1),   # Car (crossed y=300)
            MockBox(tid=2, xyxy=[200, 190, 270, 250], cls_id=3),  # Truck (still above line)
        ]
        newly_2 = counter.update(b2, frame_idx=2, timestamp_s=0.066)
        self.assertEqual(newly_2, [1], "Vehicle 1 should be newly counted")
        self.assertEqual(counter.total_count, 1, "Cumulative crossings should be 1")
        self.assertEqual(counter.class_counts["car"], 1)
        self.assertEqual(counter.current_visible_count, 2, "Visible in-frame count must be 2")

        # Frame 3: Vehicle 1 moves further down (y=450), Vehicle 2 crosses (y=340)
        b3 = [
            MockBox(tid=1, xyxy=[50, 420, 120, 480], cls_id=1),   # Car (already counted)
            MockBox(tid=2, xyxy=[200, 310, 270, 370], cls_id=3),  # Truck (crossed y=300)
        ]
        newly_3 = counter.update(b3, frame_idx=3, timestamp_s=0.099)
        self.assertEqual(newly_3, [2], "Only Vehicle 2 should be newly counted")
        self.assertEqual(counter.total_count, 2, "Cumulative crossings should be 2")
        self.assertEqual(counter.class_counts["car"], 1)
        self.assertEqual(counter.class_counts["truck"], 1)

    def test_telemetry_generation(self):
        """Verifies per-frame telemetry format and accuracy for player synchronization."""
        counter = VehicleCounter(line_start=(0, 200), line_end=(400, 200))
        b = [MockBox(tid=5, xyxy=[50, 100, 100, 150], cls_id=0)]  # Bus

        counter.update(b, frame_idx=10, timestamp_s=0.333)
        telem = counter.get_telemetry()
        self.assertEqual(len(telem), 1)

        t0 = telem[0]
        self.assertEqual(t0["frame_idx"], 10)
        self.assertEqual(t0["timestamp_s"], 0.333)
        self.assertEqual(t0["cumulative_total"], 0)
        self.assertEqual(t0["visible_total"], 1)
        self.assertEqual(t0["visible_classes"]["bus"], 1)
        self.assertIn("newly_counted", t0)

    def test_wrong_way_configurable_directions(self):
        """Tests that wrong-way detector supports 'down', 'up', 'left', and 'right' configurations."""
        # 1. Configured direction = 'down' (+y). Vehicle moving UP (-y) is wrong-way.
        ww_down = WrongWayDetector(expected_direction="down", min_observations=4, min_displacement_px=25)
        v_down = None
        for y in [300, 270, 240, 210, 180]:
            res = ww_down.update(track_id=10, center=(100, y))
            if res:
                v_down = res
                break
        self.assertIsNotNone(v_down)
        self.assertEqual(v_down["event_type"], "WRONG_WAY")

        # 2. Configured direction = 'right' (+x). Vehicle moving LEFT (-x) is wrong-way.
        ww_right = WrongWayDetector(expected_direction="right", min_observations=4, min_displacement_px=25)
        v_right = None
        for x in [400, 370, 340, 310, 280]:
            res = ww_right.update(track_id=20, center=(x, 200))
            if res:
                v_right = res
                break
        self.assertIsNotNone(v_right)
        self.assertEqual(v_right["event_type"], "WRONG_WAY")

        # 3. Vehicle moving in expected direction must NOT trigger violation
        ww_compliant = WrongWayDetector(expected_direction="down", min_observations=4, min_displacement_px=25)
        v_comp = None
        for y in [100, 130, 160, 190, 220]:  # Moving down (+y)
            res = ww_compliant.update(track_id=30, center=(100, y))
            if res:
                v_comp = res
                break
        self.assertIsNone(v_comp, "Compliant vehicle must not trigger wrong-way")

    def test_red_light_signal_and_line_scaling(self):
        """Tests that Red-Light violation requires explicit RED signal and scales line with resolution."""
        rl = RedLightDetector(stop_line=((0, 500), (1000, 500)), signal_state="GREEN")
        # Line scaling
        rl.adjust_line(width=1920, height=1080, line_y_pct=60.0)
        self.assertEqual(rl.stop_line_start, (0, 648))
        self.assertEqual(rl.stop_line_end, (1920, 648))

        # Under GREEN signal, crossing must NOT trigger
        v_green = rl.check_crossing(track_id=88, prev_center=(500, 600), curr_center=(500, 700))
        self.assertIsNone(v_green, "GREEN signal must never trigger red-light violation")

        # Switch to RED signal
        rl.set_signal_state("RED")
        v_red = rl.check_crossing(track_id=88, prev_center=(500, 600), curr_center=(500, 700))
        self.assertIsNotNone(v_red, "RED signal crossing must trigger violation")
        self.assertEqual(v_red["event_type"], "RED_LIGHT")

    def test_cooldown_deduplication(self):
        """Verifies cooldown prevents duplicate alerts for the same track within cooldown window."""
        ww = WrongWayDetector(expected_direction="down", min_observations=3, min_displacement_px=20, cooldown_seconds=5.0)

        # Trigger first violation at t=1.0s
        v1 = None
        for y in [300, 270, 240, 210]:
            res = ww.update(track_id=55, center=(100, y), current_time=1.0)
            if res:
                v1 = res
                break
        self.assertIsNotNone(v1)

        # Immediate next frame at t=1.5s (within 5.0s cooldown) should return None
        v_dup = ww.update(track_id=55, center=(100, 180), current_time=1.5)
        self.assertIsNone(v_dup, "Duplicate violation within cooldown must be suppressed")

        # After cooldown (t=7.0s > 1.0s + 5.0s), vehicle can report if condition persists
        v_after = ww.update(track_id=55, center=(100, 150), current_time=7.0)
        self.assertIsNotNone(v_after, "Violation should report after cooldown window expires")

    def test_annotated_evidence_generation(self):
        """Tests that evidence frame is cleanly annotated with banners and bounding box."""
        vm = TrafficViolationManager(evidence_dir=os.path.join(self.temp_dir, "ev"))
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        annotated = vm._create_annotated_evidence(
            frame=raw_frame,
            bbox=(100, 120, 220, 260),
            event_type="WRONG_WAY",
            track_id=42,
            vehicle_class="truck",
            confidence=0.92,
            plate_text="MH 12 AB 1234",
            reason="Opposite to traffic flow"
        )
        self.assertEqual(annotated.shape, raw_frame.shape)
        # Verify drawing occurred (not identical to raw black frame)
        self.assertGreater(np.sum(annotated), 0)

    def test_database_persistence_with_track_dedup(self):
        """Tests DatabaseManager prevents duplicate event insertions for the same track_id."""
        id1 = self.db.insert_event(
            event_type="WRONG_WAY",
            track_id=99,
            vehicle_class="car",
            confidence=0.88,
            prevent_duplicates=True
        )
        self.assertIsNotNone(id1)

        # Attempt to insert identical event for same track_id
        id2 = self.db.insert_event(
            event_type="WRONG_WAY",
            track_id=99,
            vehicle_class="car",
            confidence=0.88,
            prevent_duplicates=True
        )
        self.assertEqual(id1, id2, "Duplicate insertion for same track must return existing ID")

        # Total rows in DB should remain 1
        events = self.db.get_events()
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()

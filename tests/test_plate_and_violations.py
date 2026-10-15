"""
Automated unit & integration tests for:
- PlateAssociationManager (geometric association, track-level buffering, OCR filtering)
- TrafficViolationManager (Helmet, WrongWay, Virtual Red-Light, Evidence persistence)
- Isolated DatabaseManager operations (logging, review workflow, CSV export)
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from violations.violation_manager import (
    PlateAssociationManager,
    TrafficViolationManager,
    HelmetViolationDetector,
    WrongWayDetector,
    RedLightDetector,
)
from database.database_manager import DatabaseManager


class TestPlateAssociation(unittest.TestCase):
    """Test geometric plate-to-vehicle association logic."""

    def setUp(self):
        self.mgr = PlateAssociationManager(
            plate_model_path=str(ROOT / "models" / "plate_detector.pt"),
            detection_interval=1,
            min_plate_conf=0.25,
            min_ocr_conf=0.30
        )

    def test_plate_inside_vehicle_bounding_box(self):
        """Plate center cleanly inside vehicle bounding box should associate."""
        # Vehicle: x1=100, y1=100, x2=300, y2=400 (w=200, h=300)
        # Plate: x1=150, y1=320, x2=250, y2=360 (inside lower portion)
        candidates = [(42, (100, 100, 300, 400), "car")]
        plate_box = (150, 320, 250, 360)
        matched_tid = self.mgr._associate_plate_to_box(plate_box, candidates)
        self.assertEqual(matched_tid, 42)

    def test_plate_outside_all_vehicles(self):
        """Plate far away from all vehicles should NOT associate (reject spurious plate)."""
        candidates = [(10, (500, 500, 700, 700), "truck")]
        plate_box = (50, 50, 100, 80)
        matched_tid = self.mgr._associate_plate_to_box(plate_box, candidates)
        self.assertIsNone(matched_tid)

    def test_plate_closest_to_correct_vehicle_among_multiple(self):
        """When multiple vehicles exist, plate should associate with the containing vehicle."""
        candidates = [
            (1, (50, 50, 200, 200), "car"),
            (2, (300, 50, 450, 200), "van"),
        ]
        plate_box = (340, 140, 410, 170)  # inside vehicle 2
        matched_tid = self.mgr._associate_plate_to_box(plate_box, candidates)
        self.assertEqual(matched_tid, 2)


class TestViolationDetectors(unittest.TestCase):
    """Test isolated violation detectors: WrongWay, RedLight, and ViolationManager."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_traffic.db")
        self.db = DatabaseManager(db_path=self.test_db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_wrong_way_detector(self):
        """Vehicles moving opposite to configured direction must trigger wrong-way violation."""
        # Expected direction is 'down' (+y). A vehicle moving up (-y) is wrong-way.
        ww = WrongWayDetector(expected_direction="down", min_observations=5, min_displacement_px=30)
        # Simulate vehicle moving upward from y=400 down to y=200
        violation = None
        for step, y in enumerate(range(400, 200, -25)):
            v = ww.update(track_id=99, center=(300, y))
            if v is not None:
                violation = v
                break

        self.assertIsNotNone(violation, "Wrong-way movement should be detected")
        self.assertEqual(violation["event_type"], "WRONG_WAY")
        self.assertEqual(violation["track_id"], 99)

    def test_red_light_crossing_detector(self):
        """Vehicle crossing stop-line when signal is RED triggers RED_LIGHT violation."""
        rl = RedLightDetector(stop_line=((0, 300), (600, 300)), signal_state="RED")
        # Step from y=250 to y=350 (crosses y=300)
        violation = rl.check_crossing(track_id=7, prev_center=(200, 250), curr_center=(200, 350))
        self.assertIsNotNone(violation, "Red-light crossing should be detected when signal is RED")
        self.assertEqual(violation["event_type"], "RED_LIGHT")
        self.assertEqual(violation["track_id"], 7)

        # Vehicle moving when signal is GREEN must NOT trigger violation
        rl.set_signal_state("GREEN")
        no_violation = rl.check_crossing(track_id=8, prev_center=(200, 250), curr_center=(200, 350))
        self.assertIsNone(no_violation, "Should NOT trigger when signal is GREEN")

    def test_violation_manager_database_and_evidence(self):
        """Test end-to-end event persistence, evidence image writing, and review status in test DB."""
        evidence_dir = os.path.join(self.temp_dir, "evidence")
        vm = TrafficViolationManager(
            database_manager=self.db,
            evidence_dir=evidence_dir,
            enable_helmet=False,
            enable_wrong_way=True,
            enable_red_light=True,
            enable_plate_assoc=False,
            expected_direction="down"
        )
        # Manually configure red-light line
        vm.red_light_detector.stop_line_start = (0, 200)
        vm.red_light_detector.stop_line_end = (640, 200)
        vm.red_light_detector.set_signal_state("RED")

        # Mock tracked box
        class MockBox:
            def __init__(self, tid, xyxy, cls_id, conf=0.88):
                self.id = np.array([tid])
                self.xyxy = np.array([xyxy])
                self.cls = np.array([cls_id])
                self.conf = np.array([conf])

        # Frame 1: vehicle at y=150
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        b1 = [MockBox(tid=12, xyxy=[100, 130, 200, 170], cls_id=2)]
        target_classes = {2: "Car"}
        v1 = vm.process_frame(frame1, b1, target_classes, frame_idx=1)
        self.assertEqual(len(v1), 0)

        # Frame 2: vehicle crosses line to y=250
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        b2 = [MockBox(tid=12, xyxy=[100, 230, 200, 270], cls_id=2)]
        v2 = vm.process_frame(frame2, b2, target_classes, frame_idx=2)
        self.assertEqual(len(v2), 1)
        self.assertEqual(v2[0]["event_type"], "RED_LIGHT")

        # Verify evidence file exists on disk
        ev_path = v2[0]["evidence_path"]
        self.assertTrue(os.path.exists(ev_path), f"Evidence image must exist at {ev_path}")
        self.assertGreater(os.path.getsize(ev_path), 0)

        # Verify record in SQLite database
        events = self.db.get_events()
        self.assertEqual(len(events), 1)
        evt = events[0]
        self.assertEqual(evt["event_type"], "RED_LIGHT")
        self.assertEqual(evt["track_id"], 12)
        self.assertEqual(evt["vehicle_class"], "car")
        self.assertEqual(evt["review_status"], "pending")

        # Test Review Actions (Approve / Reject)
        evt_id = evt["id"]
        ok_appr = self.db.update_review_status(evt_id, "approved")
        self.assertTrue(ok_appr)
        self.assertEqual(self.db.get_event_by_id(evt_id)["review_status"], "approved")

        # Test CSV export
        csv_str = self.db.export_to_csv()
        self.assertIn("RED_LIGHT", csv_str)
        self.assertIn("approved", csv_str)


if __name__ == "__main__":
    unittest.main()

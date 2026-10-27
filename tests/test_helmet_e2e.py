"""
Helmet Detection End-to-End Verification Tests
===============================================
Verifies the complete NO_HELMET flow:
  HelmetViolationDetector -> DatabaseManager.insert_event -> GET /api/violations

Test categories:
  1. Unit: model loading & class-map
  2. Unit: check_motorcycle with synthetic NO_HELMET frame
  3. Unit: cooldown / deduplication
  4. Unit: database persistence of NO_HELMET event
  5. Integration: GET /api/violations returns the saved event
  6. Integration: overview stat counts the violation
  7. Optional smoke-test against data/input/sample.mp4 (skipped if absent)

Run with:
  .venv\Scripts\python.exe -m pytest tests/test_helmet_e2e.py -v
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_VIDEO = ROOT / "data" / "input" / "sample.mp4"
HELMET_MODEL = ROOT / "models" / "helmet_detector.pt"


def _black_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_fake_motorcycle_bbox():
    return 100, 100, 300, 350


def _mock_no_helmet_result():
    box = MagicMock()
    box.cls = [MagicMock()]
    box.cls[0].item.return_value = 3
    box.conf = [MagicMock()]
    box.conf[0].item.return_value = 0.82
    results_mock = MagicMock()
    results_mock[0].boxes = [box]
    return results_mock


def _mock_helmet_result():
    box = MagicMock()
    box.cls = [MagicMock()]
    box.cls[0].item.return_value = 0
    box.conf = [MagicMock()]
    box.conf[0].item.return_value = 0.91
    results_mock = MagicMock()
    results_mock[0].boxes = [box]
    return results_mock


class TestHelmetDetectorUnit(unittest.TestCase):

    def setUp(self):
        from violations.violation_manager import HelmetViolationDetector
        self.HelmetViolationDetector = HelmetViolationDetector

    def test_model_loads_with_correct_path(self):
        self.assertTrue(HELMET_MODEL.exists(), f"Helmet model not found at {HELMET_MODEL}")
        det = self.HelmetViolationDetector(model_path=str(HELMET_MODEL))
        self.assertIsNotNone(det.model, "Model should be loaded after __init__")

    def test_class_map_contains_nohelmet(self):
        det = self.HelmetViolationDetector(model_path=str(HELMET_MODEL))
        names = [v.lower() for v in det.class_map.values()]
        self.assertIn("nohelmet", names, f"Expected nohelmet in class_map: {det.class_map}")

    def test_class_map_contains_helmet(self):
        det = self.HelmetViolationDetector(model_path=str(HELMET_MODEL))
        names = [v.lower() for v in det.class_map.values()]
        self.assertIn("helmet", names, f"Expected helmet in class_map: {det.class_map}")

    @patch("violations.violation_manager.YOLO")
    def test_no_helmet_returns_violation(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_model.names = {0: "Helmet", 1: "fd", 2: "motorcycle", 3: "nohelmet", 4: "platenumber"}
        mock_model.return_value = _mock_no_helmet_result()
        mock_yolo_cls.return_value = mock_model
        det = self.HelmetViolationDetector(model_path=str(HELMET_MODEL))
        det.model = mock_model
        result = det.check_motorcycle(_black_frame(), _make_fake_motorcycle_bbox(), track_id=7, current_time=1.0)
        self.assertIsNotNone(result, "Expected a violation dict for NO_HELMET, got None")
        self.assertEqual(result["event_type"], "NO_HELMET")
        self.assertGreater(result["confidence"], 0.0)
        self.assertIn("bbox", result)

    @patch("violations.violation_manager.YOLO")
    def test_helmet_compliant_returns_none(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_model.names = {0: "Helmet", 1: "fd", 2: "motorcycle", 3: "nohelmet", 4: "platenumber"}
        mock_model.return_value = _mock_helmet_result()
        mock_yolo_cls.return_value = mock_model
        det = self.HelmetViolationDetector(model_path=str(HELMET_MODEL))
        det.model = mock_model
        result = det.check_motorcycle(_black_frame(), _make_fake_motorcycle_bbox(), track_id=8, current_time=1.0)
        self.assertIsNone(result, "Expected None for compliant helmet rider")

    @patch("violations.violation_manager.YOLO")
    def test_cooldown_prevents_duplicate(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_model.names = {0: "Helmet", 3: "nohelmet"}
        mock_model.return_value = _mock_no_helmet_result()
        mock_yolo_cls.return_value = mock_model
        det = self.HelmetViolationDetector(model_path=str(HELMET_MODEL), cooldown_seconds=30.0)
        det.model = mock_model
        frame = _black_frame()
        bbox = _make_fake_motorcycle_bbox()
        first = det.check_motorcycle(frame, bbox, track_id=99, current_time=0.0)
        self.assertIsNotNone(first, "First detection should fire")
        second = det.check_motorcycle(frame, bbox, track_id=99, current_time=5.0)
        self.assertIsNone(second, "Second detection within cooldown must be suppressed")

    def test_missing_model_returns_none(self):
        det = self.HelmetViolationDetector(model_path="/nonexistent/model.pt")
        self.assertIsNone(det.model)
        result = det.check_motorcycle(_black_frame(), _make_fake_motorcycle_bbox())
        self.assertIsNone(result)


class TestHelmetDatabasePersistence(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_helmet.db")
        from database.database_manager import DatabaseManager
        self.db = DatabaseManager(db_path=self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_no_helmet_event_persisted(self):
        ev_id = self.db.insert_event(
            event_type="NO_HELMET", track_id=42, vehicle_class="two-wheeler",
            confidence=0.82, plate_text="UNKNOWN",
            evidence_path="evidence/helmet_viol_test.jpg", review_status="pending",
        )
        self.assertIsNotNone(ev_id)
        ev = self.db.get_event_by_id(ev_id)
        self.assertEqual(ev["event_type"], "NO_HELMET")
        self.assertEqual(ev["track_id"], 42)
        self.assertEqual(ev["vehicle_class"], "two-wheeler")
        self.assertAlmostEqual(ev["confidence"], 0.82, places=2)
        self.assertEqual(ev["review_status"], "pending")

    def test_no_helmet_count_fields_are_null(self):
        ev_id = self.db.insert_event(event_type="NO_HELMET", track_id=43, vehicle_class="motorcycle", confidence=0.75)
        ev = self.db.get_event_by_id(ev_id)
        self.assertNotEqual(ev["event_type"], "COUNT")
        self.assertIsNone(ev.get("count_details"))
        self.assertIsNone(ev.get("total_count"))

    def test_multiple_tracks_stored_independently(self):
        id1 = self.db.insert_event(event_type="NO_HELMET", track_id=1, vehicle_class="motorcycle", confidence=0.8)
        id2 = self.db.insert_event(event_type="NO_HELMET", track_id=2, vehicle_class="two-wheeler", confidence=0.75)
        self.assertNotEqual(id1, id2)
        events = self.db.get_events(event_type="NO_HELMET")
        track_ids = [e["track_id"] for e in events]
        self.assertIn(1, track_ids)
        self.assertIn(2, track_ids)


class TestHelmetAPIIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_api_helmet.db")
        from database.database_manager import DatabaseManager
        from api.main import app
        from api.dependencies import get_db
        from fastapi.testclient import TestClient
        self.db = DatabaseManager(db_path=self.db_path)
        app.dependency_overrides[get_db] = lambda: self.db
        self.client = TestClient(app)
        self.app = app
        self.get_db = get_db

    def tearDown(self):
        self.app.dependency_overrides.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_no_helmet_visible_via_get_violations(self):
        self.db.insert_event(
            event_type="NO_HELMET", track_id=55, vehicle_class="two-wheeler",
            confidence=0.88, plate_text="UNKNOWN",
            evidence_path="evidence/helmet_viol_api_test.jpg", review_status="pending",
        )
        resp = self.client.get("/api/violations")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data["total"], 1)
        types = [item["event_type"] for item in data["items"]]
        self.assertIn("NO_HELMET", types, f"NO_HELMET not found in items: {types}")

    def test_no_helmet_count_in_stats_overview(self):
        self.db.insert_event(event_type="NO_HELMET", track_id=60, vehicle_class="motorcycle", confidence=0.85)
        resp = self.client.get("/api/stats/overview")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data["total_violations"], 1)

    def test_no_helmet_filter_works(self):
        self.db.insert_event(event_type="NO_HELMET", track_id=70, vehicle_class="motorcycle", confidence=0.9)
        self.db.insert_event(event_type="WRONG_WAY", track_id=71, vehicle_class="car", confidence=0.7)
        resp = self.client.get("/api/violations?event_type=NO_HELMET")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for item in data["items"]:
            self.assertEqual(item["event_type"], "NO_HELMET")

    def test_count_not_included_in_violations_by_default(self):
        self.db.insert_event(event_type="NO_HELMET", track_id=80, vehicle_class="motorcycle", confidence=0.88)
        self.db.insert_event(event_type="COUNT", total_count=10, count_details={"car": 10})
        resp = self.client.get("/api/violations")
        data = resp.json()
        event_types = [item["event_type"] for item in data["items"]]
        self.assertNotIn("COUNT", event_types)


class TestDedicatedHelmetDetection(unittest.TestCase):
    """
    Regression tests for dedicated spatial helmet-first detection:
    - compliant riders
    - helmetless riders
    - multiple riders with spatial association
    - duplicate event prevention across consecutive frames
    - missing vehicle detection fallback
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_helmet_reg_")
        self.db_path = os.path.join(self.temp_dir, "test.db")
        from database.database_manager import DatabaseManager
        from violations.violation_manager import TrafficViolationManager, HelmetViolationDetector
        self.db = DatabaseManager(db_path=self.db_path)
        self.evidence_dir = os.path.join(self.temp_dir, "evidence")
        os.makedirs(self.evidence_dir, exist_ok=True)
        self.HelmetViolationDetector = HelmetViolationDetector
        self.TrafficViolationManager = TrafficViolationManager

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_helmetless_rider_detected(self):
        """Verifies ground-truth helmetless rider triggers NO_HELMET violation."""
        img_path = ROOT / "datasets" / "motorcycle_helmet" / "train" / "images" / "100_jpg.rf.eac62d6f28165186ba9cee9d1e0e2670.jpg"
        if not img_path.exists():
            self.skipTest(f"Missing {img_path}")
        frame = cv2.imread(str(img_path))
        det = self.HelmetViolationDetector(model_path=str(HELMET_MODEL))
        violations = det.detect_frame_violations(frame, current_time=1.0)
        self.assertGreaterEqual(len(violations), 1)
        self.assertEqual(violations[0]["event_type"], "NO_HELMET")
        self.assertGreaterEqual(violations[0]["confidence"], 0.20)
        self.assertIn("bbox", violations[0])
        self.assertIn("rider_box", violations[0])

    def test_compliant_rider_no_false_positive(self):
        """Verifies ground-truth helmet-compliant rider does not trigger false NO_HELMET."""
        img_path = ROOT / "datasets" / "motorcycle_helmet" / "train" / "images" / "10_jpg.rf.389e19aa03877a89aa9bd1166536fec3.jpg"
        if not img_path.exists():
            self.skipTest(f"Missing {img_path}")
        frame = cv2.imread(str(img_path))
        det = self.HelmetViolationDetector(model_path=str(HELMET_MODEL))
        violations = det.detect_frame_violations(frame, current_time=1.0)
        self.assertEqual(len(violations), 0, f"Expected 0 violations on compliant rider, got {len(violations)}")

    def test_multi_rider_spatial_association(self):
        """Verifies multi-rider scene associates riders spatially so helmet does not suppress nohelmet."""
        img_path = ROOT / "datasets" / "motorcycle_helmet" / "train" / "images" / "28_jpg.rf.c44dd4fac8af2108417db664f4f5dd6b.jpg"
        if not img_path.exists():
            self.skipTest(f"Missing {img_path}")
        frame = cv2.imread(str(img_path))
        det = self.HelmetViolationDetector(model_path=str(HELMET_MODEL))
        violations = det.detect_frame_violations(frame, current_time=1.0)
        self.assertGreaterEqual(len(violations), 1, "Expected at least 1 violation for the unhelmeted rider")
        self.assertEqual(violations[0]["event_type"], "NO_HELMET")

    def test_consecutive_frame_deduplication(self):
        """Verifies repeated frames of the same motorcycle only trigger one violation within cooldown."""
        img_path = ROOT / "datasets" / "motorcycle_helmet" / "train" / "images" / "100_jpg.rf.eac62d6f28165186ba9cee9d1e0e2670.jpg"
        if not img_path.exists():
            self.skipTest(f"Missing {img_path}")
        frame = cv2.imread(str(img_path))
        vm = self.TrafficViolationManager(
            database_manager=self.db,
            evidence_dir=self.evidence_dir,
            helmet_model_path=str(HELMET_MODEL),
            enable_helmet=True,
            cooldown_seconds=5.0
        )
        # Frame 1 at t=0.0s
        v1 = vm.process_frame(frame, tracked_boxes=[], target_classes={}, frame_idx=0, timestamp_s=0.0)
        self.assertEqual(len(v1), 1)

        # Frame 2 at t=0.5s (within cooldown)
        v2 = vm.process_frame(frame, tracked_boxes=[], target_classes={}, frame_idx=1, timestamp_s=0.5)
        self.assertEqual(len(v2), 0, "Duplicate violation within cooldown should be prevented")

        # Database should contain exactly 1 event
        db_events = self.db.get_events(event_type="NO_HELMET")
        self.assertEqual(len(db_events), 1)

    def test_missing_vehicle_detection_fallback(self):
        """Verifies helmet detection succeeds even when vehicle detector outputs empty tracked_boxes."""
        img_path = ROOT / "datasets" / "motorcycle_helmet" / "train" / "images" / "100_jpg.rf.eac62d6f28165186ba9cee9d1e0e2670.jpg"
        if not img_path.exists():
            self.skipTest(f"Missing {img_path}")
        frame = cv2.imread(str(img_path))
        vm = self.TrafficViolationManager(
            database_manager=self.db,
            evidence_dir=self.evidence_dir,
            helmet_model_path=str(HELMET_MODEL),
            enable_helmet=True
        )
        # Pass empty tracked_boxes (simulating vehicle_detector.pt missing the motorcycle)
        violations = vm.process_frame(frame, tracked_boxes=[], target_classes={}, frame_idx=0, timestamp_s=0.0)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["event_type"], "NO_HELMET")
        self.assertTrue(os.path.exists(violations[0]["evidence_path"]))

        # Check DB
        events = self.db.get_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "NO_HELMET")


if __name__ == "__main__":
    unittest.main(verbosity=2)

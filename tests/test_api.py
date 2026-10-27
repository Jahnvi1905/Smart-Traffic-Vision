import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from api.dependencies import get_db, get_vehicle_detector, get_plate_detector, get_plate_ocr
from api.utils import to_media_url
from database.database_manager import DatabaseManager


class TestFastAPIBackend(unittest.TestCase):

    def setUp(self):
        # 1. Setup temporary workspace and isolated test database
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_traffic.db")
        self.test_db = DatabaseManager(db_path=self.test_db_path)

        # 2. Override get_db dependency
        app.dependency_overrides[get_db] = lambda: self.test_db

        # 3. Initialize test client
        self.client = TestClient(app)

    def tearDown(self):
        # Clear dependency overrides and remove temporary files
        app.dependency_overrides.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_health_endpoint(self):
        """Verifies GET /api/health returns 200 with operational status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["app"], "Smart Traffic Vision API")
        self.assertIn("timestamp", data)

    def test_diagnostics_endpoint(self):
        """Verifies GET /api/diagnostics inspects models, database, and directories."""
        response = self.client.get("/api/diagnostics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("models", data)
        self.assertIn("database", data)
        self.assertTrue(data["database"]["connected"])
        self.assertIn("directories", data)
        self.assertIn("openh264", data)

    def test_stats_overview_empty_db(self):
        """Verifies GET /api/stats/overview returns zeroed structure when DB is empty."""
        response = self.client.get("/api/stats/overview")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_vehicles"], 0)
        self.assertEqual(data["total_violations"], 0)
        self.assertEqual(data["violations_by_type"]["NO_HELMET"], 0)
        self.assertEqual(data["category_breakdown"]["car"], 0)
        self.assertEqual(len(data["recent_events"]), 0)

    def test_stats_overview_with_data_and_count_exclusion(self):
        """
        Verifies that COUNT events are aggregated into total_vehicles and categories,
        while strictly EXCLUDED from total_violations count.
        """
        # Insert COUNT event (road traffic flow)
        self.test_db.insert_event(
            event_type="COUNT",
            vehicle_class="Total: 7",
            confidence=0.35,
            plate_text="{'bus': 1, 'car': 3, 'three-wheeler': 1, 'truck': 2, 'two-wheeler': 0, 'van': 0}",
            evidence_path="data/output/processed_vid1.mp4",
            review_status="approved"
        )

        # Insert 3 genuine safety violations
        self.test_db.insert_event(
            event_type="NO_HELMET",
            track_id=10,
            vehicle_class="motorcycle",
            confidence=0.85,
            evidence_path="data/output/violations/helmet_1.jpg",
            review_status="pending"
        )
        self.test_db.insert_event(
            event_type="WRONG_WAY",
            track_id=11,
            vehicle_class="car",
            confidence=0.90,
            evidence_path="data/output/violations/wrongway_1.jpg",
            review_status="approved"
        )
        self.test_db.insert_event(
            event_type="RED_LIGHT",
            track_id=12,
            vehicle_class="truck",
            confidence=0.95,
            evidence_path="data/output/violations/redlight_1.jpg",
            review_status="pending"
        )

        response = self.client.get("/api/stats/overview")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Vehicle counts should be 7
        self.assertEqual(data["total_vehicles"], 7)
        self.assertEqual(data["category_breakdown"]["car"], 3)
        self.assertEqual(data["category_breakdown"]["bus"], 1)
        self.assertEqual(data["category_breakdown"]["truck"], 2)

        # Safety violations total must be 3 (COUNT event MUST NOT be counted)
        self.assertEqual(data["total_violations"], 3)
        self.assertEqual(data["violations_by_type"]["NO_HELMET"], 1)
        self.assertEqual(data["violations_by_type"]["WRONG_WAY"], 1)
        self.assertEqual(data["violations_by_type"]["RED_LIGHT"], 1)

        # Verify recent events convert evidence paths to safe media URLs
        for evt in data["recent_events"]:
            if evt["evidence_url"]:
                self.assertTrue(evt["evidence_url"].startswith("/media/"))
                self.assertNotIn(":\\", evt["evidence_url"])

    def test_violations_get_and_filtering(self):
        """Verifies GET /api/violations pagination, search, and type filtering."""
        # Insert events
        self.test_db.insert_event(event_type="COUNT", vehicle_class="Total: 5")
        self.test_db.insert_event(event_type="NO_HELMET", vehicle_class="motorcycle", plate_text="DL01AB1234")
        self.test_db.insert_event(event_type="WRONG_WAY", vehicle_class="car", plate_text="MH02CD5678", review_status="approved")
        self.test_db.insert_event(event_type="RED_LIGHT", vehicle_class="bus", plate_text="KA03EF9012")

        # 1. Default request should exclude COUNT
        res_all = self.client.get("/api/violations")
        self.assertEqual(res_all.status_code, 200)
        data_all = res_all.json()
        self.assertEqual(data_all["total"], 3)
        event_types = [item["event_type"] for item in data_all["items"]]
        self.assertNotIn("COUNT", event_types)

        # 2. Filter by event_type
        res_filter = self.client.get("/api/violations?event_type=NO_HELMET")
        self.assertEqual(res_filter.status_code, 200)
        data_filter = res_filter.json()
        self.assertEqual(data_filter["total"], 1)
        self.assertEqual(data_filter["items"][0]["event_type"], "NO_HELMET")

        # 3. Search query
        res_search = self.client.get("/api/violations?search=MH02")
        self.assertEqual(res_search.status_code, 200)
        data_search = res_search.json()
        self.assertEqual(data_search["total"], 1)
        self.assertEqual(data_search["items"][0]["plate_text"], "MH02CD5678")

        # 4. Pagination
        res_page = self.client.get("/api/violations?page=1&limit=2")
        self.assertEqual(res_page.status_code, 200)
        data_page = res_page.json()
        self.assertEqual(len(data_page["items"]), 2)
        self.assertEqual(data_page["pages"], 2)

    def test_violations_review_status_patch(self):
        """Verifies PATCH /api/violations/{id}/review updates status cleanly."""
        eid = self.test_db.insert_event(
            event_type="NO_HELMET",
            vehicle_class="motorcycle",
            review_status="pending"
        )

        # Valid update
        res = self.client.patch(f"/api/violations/{eid}/review", json={"status": "approved"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["review_status"], "approved")

        # Verify DB state
        row = self.test_db.get_event_by_id(eid)
        self.assertEqual(row["review_status"], "approved")

        # Invalid status should return 422
        res_inv = self.client.patch(f"/api/violations/{eid}/review", json={"status": "bogus_status"})
        self.assertEqual(res_inv.status_code, 422)

        # Non-existent ID should return 404
        res_404 = self.client.patch("/api/violations/999999/review", json={"status": "rejected"})
        self.assertEqual(res_404.status_code, 404)

    def test_reports_export_csv(self):
        """Verifies GET /api/reports/export returns downloadable CSV attachment."""
        self.test_db.insert_event(event_type="NO_HELMET", vehicle_class="motorcycle", plate_text="DL01AB")
        self.test_db.insert_event(event_type="WRONG_WAY", vehicle_class="car", plate_text="MH02CD")

        res = self.client.get("/api/reports/export")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "text/csv; charset=utf-8")
        self.assertIn("attachment; filename=", res.headers["content-disposition"])
        content = res.text
        self.assertIn("id,timestamp,track_id,vehicle_class", content)
        self.assertIn("NO_HELMET", content)
        self.assertIn("WRONG_WAY", content)

    def test_to_media_url_conversion(self):
        """Verifies safe media URL conversion and local path sanitization."""
        self.assertEqual(to_media_url(None), None)
        self.assertEqual(to_media_url(""), None)

        # Windows paths
        win_vpath = r"C:\Users\ASUS\OneDrive\Desktop\CV_Project_Traffic\data\output\violations\alert_1.jpg"
        self.assertEqual(to_media_url(win_vpath), "/media/violations/alert_1.jpg")

        win_opath = r"C:\Users\ASUS\OneDrive\Desktop\CV_Project_Traffic\data\output\processed_vid.mp4"
        self.assertEqual(to_media_url(win_opath), "/media/output/processed_vid.mp4")

        # Relative paths
        self.assertEqual(to_media_url("data/output/violations/alert_2.jpg"), "/media/violations/alert_2.jpg")
        self.assertEqual(to_media_url("data/output/detected_3.jpg"), "/media/output/detected_3.jpg")

    def test_analyze_image_endpoint(self):
        """Verifies POST /api/traffic/analyze-image processes image and returns bounding boxes."""
        # Create a small synthetic test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)

        # Mock vehicle detector to avoid slow real PyTorch inference in unit test
        mock_detector = MagicMock()
        mock_detector.target_class_ids = [0, 1, 2, 3, 4, 5]
        mock_detector.target_classes = {1: "car"}
        mock_detector.draw_detections.side_effect = lambda f, b: f

        # Mock result box
        mock_box = MagicMock()
        mock_box.cls = np.array([1])
        mock_box.conf = np.array([0.92])
        mock_box.xyxy = np.array([[10, 10, 50, 50]])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_detector.model.return_value = [mock_result]

        app.dependency_overrides[get_vehicle_detector] = lambda: mock_detector

        files = {"file": ("test.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
        res = self.client.post("/api/traffic/analyze-image", files=files, data={"conf_threshold": 0.35})

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_detected"], 1)
        self.assertEqual(data["class_counts"]["car"], 1)
        self.assertEqual(data["detections"][0]["label"], "Car")
        self.assertEqual(data["detections"][0]["confidence"], 0.92)
        self.assertTrue(data["annotated_image_url"].startswith("/media/output/"))

    def test_plates_scan_endpoint(self):
        """Verifies POST /api/plates/scan with mocked plate detector and OCR."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)

        mock_plate_det = MagicMock()
        mock_plate_det.detect_plates.return_value = [{"box": [10, 10, 40, 30], "conf": 0.88}]
        mock_plate_det.crop_plates.return_value = [np.zeros((20, 30, 3), dtype=np.uint8)]
        mock_plate_det.draw_detections.side_effect = lambda f, d: f

        mock_plate_ocr = MagicMock()
        mock_plate_ocr.read_plate.return_value = ("DL 01 AB 1234", 0.94)

        app.dependency_overrides[get_plate_detector] = lambda: mock_plate_det
        app.dependency_overrides[get_plate_ocr] = lambda: mock_plate_ocr

        files = {"file": ("plate.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
        res = self.client.post("/api/plates/scan", files=files)

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_plates"], 1)
        self.assertEqual(data["plates"][0]["plate_text"], "DL 01 AB 1234")
        self.assertEqual(data["plates"][0]["confidence"], 0.94)
        self.assertTrue(data["plates"][0]["crop_url"].startswith("/media/output/"))

    def test_analyze_video_mocked(self):
        """Verifies POST /api/traffic/analyze-video pipeline validation and telemetry return."""
        # Create a small valid test mp4 video
        test_video_path = os.path.join(self.temp_dir, "test_input.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(test_video_path, fourcc, 10.0, (64, 64))
        for _ in range(5):
            vw.write(np.zeros((64, 64, 3), dtype=np.uint8))
        vw.release()

        # Mock vehicle detector process_video to return valid telemetry
        mock_detector = MagicMock()
        mock_detector.target_classes = {0: "bus", 1: "car"}

        def fake_process_video(input_path, output_path, counter=None, **kwargs):
            # Create dummy output file
            with open(output_path, "wb") as f:
                f.write(b"dummy video bytes exceeding 100 bytes" * 10)
            if counter:
                counter.counted_ids = {1, 2}
                counter.class_counts = {"car": 2}
                counter.telemetry = [
                    {
                        "frame_idx": 1,
                        "timestamp_s": 0.1,
                        "cumulative_total": 1,
                        "cumulative_classes": {"car": 1},
                        "visible_total": 1,
                        "visible_classes": {"car": 1},
                        "newly_counted": [1]
                    },
                    {
                        "frame_idx": 2,
                        "timestamp_s": 0.2,
                        "cumulative_total": 2,
                        "cumulative_classes": {"car": 2},
                        "visible_total": 1,
                        "visible_classes": {"car": 1},
                        "newly_counted": [2]
                    }
                ]
            return True

        mock_detector.process_video.side_effect = fake_process_video
        app.dependency_overrides[get_vehicle_detector] = lambda: mock_detector

        with open(test_video_path, "rb") as vf:
            files = {"file": ("traffic.mp4", vf, "video/mp4")}
            res = self.client.post("/api/traffic/analyze-video", files=files, data={"enable_counter": True})

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_counted"], 2)
        self.assertEqual(data["class_counts"]["car"], 2)
        self.assertEqual(len(data["telemetry"]), 2)
        self.assertEqual(data["telemetry"][1]["cumulative_total"], 2)
        self.assertTrue(data["video_url"].startswith("/media/output/"))

        # Verify COUNT event recorded in SQLite with structured fields
        count_events = self.test_db.get_events(event_type="COUNT")
        self.assertEqual(len(count_events), 1)
        self.assertEqual(count_events[0]["total_count"], 2)
        self.assertIsNone(count_events[0]["plate_text"], "COUNT events must NEVER store text in plate_text")
        self.assertEqual(count_events[0]["vehicle_class"], "Traffic Stream")


if __name__ == "__main__":
    unittest.main()

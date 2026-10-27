import io
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app
from api.dependencies import get_db, get_plate_detector, get_plate_ocr
from database.database_manager import DatabaseManager
from ocr.plate_ocr import normalize_plate_text, validate_plate_output, PlateOCR


class TestHardeningPass(unittest.TestCase):
    """
    Regression test suite for hardening pass:
    - Structured COUNT event persistence without plate_text pollution.
    - Non-destructive migration and backward compatibility for legacy SQLite data.
    - Exclusion of COUNT events from safety violation totals and review updates.
    - Conservative license plate OCR normalization and raw output preservation.
    - Low-confidence OCR classification and handling.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "hardening_test.db")
        self.test_db = DatabaseManager(db_path=self.test_db_path)
        app.dependency_overrides[get_db] = lambda: self.test_db
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ── 1. COUNT Event Persistence Tests ──────────────────────────────────────────

    def test_count_persistence_structured(self):
        """
        Verifies that COUNT events are stored with structured count_details and total_count,
        while plate_text is strictly NULL.
        """
        counts = {"bus": 0, "car": 8, "three-wheeler": 1, "truck": 2, "two-wheeler": 3, "van": 1}
        event_id = self.test_db.insert_event(
            event_type="COUNT",
            total_count=15,
            count_details=counts,
            evidence_path="data/output/test_video.mp4",
            review_status="approved"
        )

        event = self.test_db.get_event_by_id(event_id)
        self.assertIsNotNone(event)
        self.assertEqual(event["event_type"], "COUNT")
        self.assertEqual(event["total_count"], 15)
        self.assertIsNone(event["plate_text"], "COUNT events must NEVER store text in plate_text")
        self.assertEqual(event["vehicle_class"], "Traffic Stream")

        # Verify count_details is valid JSON
        parsed_details = json.loads(event["count_details"])
        self.assertEqual(parsed_details["car"], 8)
        self.assertEqual(parsed_details["truck"], 2)

    def test_count_persistence_legacy_dict_rerouting(self):
        """
        Verifies that if legacy code calls insert_event with a dictionary in plate_text
        and vehicle_class='Total: 13', plate_text is sanitized to NULL, count_details
        receives the data, and total_count is correctly extracted.
        """
        event_id = self.test_db.insert_event(
            event_type="COUNT",
            vehicle_class="Total: 13",
            confidence=0.35,
            plate_text="{'bus': 0, 'car': 6, 'three-wheeler': 0, 'truck': 1, 'two-wheeler': 5, 'van': 1}",
            evidence_path="data/output/legacy_run.mp4",
            review_status="approved"
        )

        event = self.test_db.get_event_by_id(event_id)
        self.assertIsNotNone(event)
        self.assertEqual(event["event_type"], "COUNT")
        self.assertIsNone(event["plate_text"], "Legacy plate_text dictionary must be sanitized to NULL")
        self.assertEqual(event["total_count"], 13)
        self.assertEqual(event["vehicle_class"], "Traffic Stream")
        self.assertIn("car", event["count_details"])

    # ── 2. Database Migration & Backward Compatibility ──────────────────────────

    def test_legacy_database_migration(self):
        """
        Verifies non-destructive migration of existing database tables created prior to
        the introduction of count_details and total_count columns.
        """
        legacy_db_path = os.path.join(self.temp_dir, "legacy_schema.db")

        # 1. Create legacy schema manually without count_details / total_count
        conn = sqlite3.connect(legacy_db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                track_id INTEGER,
                vehicle_class TEXT,
                event_type TEXT NOT NULL,
                confidence REAL,
                plate_text TEXT,
                evidence_path TEXT,
                review_status TEXT DEFAULT 'pending'
            );
        """)
        # Insert raw legacy COUNT row with dictionary in plate_text
        cur.execute("""
            INSERT INTO events (timestamp, vehicle_class, event_type, confidence, plate_text, review_status)
            VALUES (?, ?, ?, ?, ?, ?);
        """, (
            "2026-10-25 12:00:00",
            "Total: 13",
            "COUNT",
            0.35,
            "{'bus': 0, 'car': 6, 'three-wheeler': 0, 'truck': 1, 'two-wheeler': 5, 'van': 1}",
            "approved"
        ))
        # Insert a genuine violation row to ensure violations are preserved
        cur.execute("""
            INSERT INTO events (timestamp, track_id, vehicle_class, event_type, confidence, plate_text, review_status)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            "2026-10-25 12:05:00",
            42,
            "motorcycle",
            "NO_HELMET",
            0.91,
            "MH12AB1234",
            "pending"
        ))
        conn.commit()
        conn.close()

        # 2. Open legacy database with DatabaseManager (triggers init_db migration)
        migrated_db = DatabaseManager(db_path=legacy_db_path)

        # 3. Verify columns were added without dropping or losing records
        migrated_events = migrated_db.get_events()
        self.assertEqual(len(migrated_events), 2)

        # 4. Verify the COUNT record was safely migrated
        count_rec = [e for e in migrated_events if e["event_type"] == "COUNT"][0]
        self.assertIsNone(count_rec["plate_text"])
        self.assertIsNotNone(count_rec["count_details"])
        self.assertEqual(count_rec["total_count"], 13)
        self.assertEqual(count_rec["vehicle_class"], "Traffic Stream")

        # 5. Verify the violation record was untouched
        violation_rec = [e for e in migrated_events if e["event_type"] == "NO_HELMET"][0]
        self.assertEqual(violation_rec["plate_text"], "MH12AB1234")
        self.assertEqual(violation_rec["track_id"], 42)

    # ── 3. COUNT Record Exclusion from Review & Safety Totals ───────────────────

    def test_count_records_cannot_be_reviewed(self):
        """
        Verifies that PATCH /api/violations/{id}/review returns HTTP 400 Bad Request
        when attempting to review a COUNT event.
        """
        count_id = self.test_db.insert_event(
            event_type="COUNT",
            total_count=10,
            count_details={"car": 10},
            review_status="approved"
        )

        response = self.client.patch(
            f"/api/violations/{count_id}/review",
            json={"status": "approved"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("cannot be reviewed as safety violations", response.json()["detail"])

    def test_safety_violations_exclude_counts_in_stats(self):
        """
        Verifies that GET /api/stats/overview strictly excludes COUNT events from total_violations.
        """
        # Insert 2 count sessions
        self.test_db.insert_event(event_type="COUNT", total_count=5, count_details={"car": 5})
        self.test_db.insert_event(event_type="COUNT", total_count=8, count_details={"truck": 8})

        # Insert 1 genuine safety violation
        self.test_db.insert_event(event_type="NO_HELMET", vehicle_class="motorcycle", confidence=0.88)

        response = self.client.get("/api/stats/overview")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["total_vehicles"], 13)
        self.assertEqual(data["total_violations"], 1, "total_violations must strictly exclude COUNT records")
        self.assertEqual(data["category_breakdown"]["car"], 5)
        self.assertEqual(data["category_breakdown"]["truck"], 8)

    # ── 4. Plate OCR Normalization & Validation Tests ────────────────────────────

    def test_ocr_normalization_boundary_artifacts(self):
        """
        Verifies conservative normalization strips boundary bracket and edge noise artifacts
        like 'MH 20 EE 7602]' -> 'MH 20 EE 7602' without guessing or fabricating characters.
        """
        # Trailing bracket artifact observed in audit
        self.assertEqual(normalize_plate_text("MH 20 EE 7602]"), "MH 20 EE 7602")

        # Bracket boundary artifacts
        self.assertEqual(normalize_plate_text("[DL 01 AB 1234]"), "DL 01 AB 1234")

        # Leading/trailing pipe or divider lines
        self.assertEqual(normalize_plate_text(" | MH 20 EE 7602 | "), "MH 20 EE 7602")

        # Multiple whitespace and casing
        self.assertEqual(normalize_plate_text("  mh   20   ee   7602  "), "MH 20 EE 7602")

        # Hyphenated plate formatting
        self.assertEqual(normalize_plate_text("KA-03-EF-9012"), "KA 03 EF 9012")

        # Empty / None
        self.assertEqual(normalize_plate_text(""), "")
        self.assertEqual(normalize_plate_text(None), "")

    def test_ocr_low_confidence_classification(self):
        """
        Verifies that confidence < 0.50 (such as 0.476) is correctly classified as
        is_low_confidence=True and confidence_status='low', preserving raw text.
        """
        # Audit sample: 0.476 confidence with trailing bracket
        val_sample = validate_plate_output("MH 20 EE 7602]", 0.476)
        self.assertEqual(val_sample["raw_text"], "MH 20 EE 7602]")
        self.assertEqual(val_sample["normalized_text"], "MH 20 EE 7602")
        self.assertEqual(val_sample["confidence"], 0.476)
        self.assertTrue(val_sample["is_low_confidence"])
        self.assertEqual(val_sample["confidence_status"], "low")
        self.assertTrue(val_sample["is_valid_format"])

        # High confidence sample
        val_high = validate_plate_output("MH 20 EE 7602", 0.88)
        self.assertFalse(val_high["is_low_confidence"])
        self.assertEqual(val_high["confidence_status"], "high")

        # Medium confidence sample (0.65)
        val_med = validate_plate_output("MH 20 EE 7602", 0.65)
        self.assertFalse(val_med["is_low_confidence"])
        self.assertEqual(val_med["confidence_status"], "medium")

        # Unreadable / empty
        val_empty = validate_plate_output("", 0.0)
        self.assertTrue(val_empty["is_low_confidence"])
        self.assertEqual(val_empty["confidence_status"], "unreadable")
        self.assertFalse(val_empty["is_valid_format"])

    # ── 5. Plate Scanner API Audit Integration ──────────────────────────────────

    def test_plate_scanner_audit_metadata_and_pending_review(self):
        """
        Verifies that /api/plates/scan returns raw_text and low-confidence flags,
        and saves low-confidence reads to the database with review_status='pending'.
        """
        mock_detector = MagicMock()
        mock_detector.detect_plates.return_value = [{"box": [10, 10, 200, 80], "conf": 0.85}]
        mock_detector.crop_plates.return_value = [
            # Dummy 10x10 numpy image
            __import__("numpy").zeros((10, 10, 3), dtype=__import__("numpy").uint8)
        ]
        mock_detector.draw_detections.return_value = (
            __import__("numpy").zeros((100, 100, 3), dtype=__import__("numpy").uint8)
        )

        mock_ocr = MagicMock()
        mock_ocr.read_plate_detailed.return_value = {
            "raw_text": "MH 20 EE 7602]",
            "normalized_text": "MH 20 EE 7602",
            "confidence": 0.476,
            "is_low_confidence": True,
            "confidence_status": "low",
            "is_valid_format": True
        }

        app.dependency_overrides[get_plate_detector] = lambda: mock_detector
        app.dependency_overrides[get_plate_ocr] = lambda: mock_ocr

        # Send scan request with save_to_db=True
        img_bytes = io.BytesIO()
        import cv2
        _, encoded = cv2.imencode(".jpg", __import__("numpy").zeros((50, 50, 3), dtype=__import__("numpy").uint8))
        img_bytes.write(encoded.tobytes())
        img_bytes.seek(0)

        response = self.client.post(
            "/api/plates/scan",
            files={"file": ("vehicle.jpg", img_bytes, "image/jpeg")},
            data={"save_to_db": "true"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_plates"], 1)

        plate_item = data["plates"][0]
        self.assertEqual(plate_item["plate_text"], "MH 20 EE 7602")
        self.assertEqual(plate_item["raw_text"], "MH 20 EE 7602]")
        self.assertEqual(plate_item["confidence"], 0.476)
        self.assertTrue(plate_item["is_low_confidence"])
        self.assertEqual(plate_item["confidence_status"], "low")

        # Verify record in SQLite database was marked 'pending' for manual review
        saved_events = self.test_db.get_events(event_type="PLATE_READ")
        self.assertEqual(len(saved_events), 1)
        self.assertEqual(saved_events[0]["plate_text"], "MH 20 EE 7602")
        self.assertEqual(saved_events[0]["review_status"], "pending",
                         "Low confidence plate reads must be logged as pending for human verification")


if __name__ == "__main__":
    unittest.main()

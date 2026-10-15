import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.database_manager import DatabaseManager


def run_database_test():
    print("=" * 60)
    print("STEP 12 — SQLite Database Manager Verification Test")
    print("=" * 60)

    db_path = os.path.join("data", "traffic_violations.db")
    print(f"[INFO] Target Database File: {db_path}")

    # 1. Initialize manager & auto-create database
    db = DatabaseManager(db_path=db_path)
    if not os.path.exists(db_path):
        print(f"[ERROR] Database file not found at: {db_path}")
        sys.exit(1)
    print(f"[PASS] Database file created/verified successfully.")

    # 2. Insert test event record
    print("\n[TEST 1] Inserting test event record...")
    test_track_id = 999
    test_event_type = "TEST_NO_HELMET"
    test_class = "motorcycle"
    test_conf = 0.945
    test_plate = "MH 20 EE 7602"
    test_evidence = os.path.join("data", "output", "plate_crop_1.jpeg")

    record_id = db.insert_event(
        event_type=test_event_type,
        track_id=test_track_id,
        vehicle_class=test_class,
        confidence=test_conf,
        plate_text=test_plate,
        evidence_path=test_evidence,
        review_status="pending"
    )
    print(f"[PASS] Inserted test event record with ID: {record_id}")

    # 3. Retrieve and verify the record
    print("\n[TEST 2] Retrieving event record by ID...")
    fetched = db.get_event_by_id(record_id)
    if not fetched:
        print(f"[ERROR] Failed to fetch record ID {record_id}")
        sys.exit(1)

    print(f"  ID           : {fetched['id']}")
    print(f"  Timestamp    : {fetched['timestamp']}")
    print(f"  Track ID     : {fetched['track_id']}")
    print(f"  Class        : {fetched['vehicle_class']}")
    print(f"  Event Type   : {fetched['event_type']}")
    print(f"  Confidence   : {fetched['confidence']}")
    print(f"  Plate Text   : {fetched['plate_text']}")
    print(f"  Evidence Path: {fetched['evidence_path']}")
    print(f"  Review Status: {fetched['review_status']}")

    assert fetched["track_id"] == test_track_id, "Track ID mismatch!"
    assert fetched["plate_text"] == test_plate, "Plate text mismatch!"
    assert fetched["review_status"] == "pending", "Initial status mismatch!"
    print("[PASS] Field verification passed.")

    # 4. Update review status
    print("\n[TEST 3] Updating review status to 'approved'...")
    updated = db.update_review_status(record_id, "approved")
    if not updated:
        print("[ERROR] Failed to update review status.")
        sys.exit(1)

    fetched_updated = db.get_event_by_id(record_id)
    assert fetched_updated["review_status"] == "approved", "Updated status mismatch!"
    print(f"[PASS] Review status updated to '{fetched_updated['review_status']}'.")

    # 5. Clean up test record
    print("\n[TEST 4] Cleaning up test record...")
    deleted = db.delete_event(record_id)
    if not deleted:
        print("[ERROR] Failed to delete test record.")
        sys.exit(1)

    verified_deleted = db.get_event_by_id(record_id)
    assert verified_deleted is None, "Test record was not deleted!"
    print("[PASS] Test record successfully deleted and verified clean.")

    print("\n" + "=" * 60)
    print("[SUCCESS] All Database Manager verification tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_database_test()

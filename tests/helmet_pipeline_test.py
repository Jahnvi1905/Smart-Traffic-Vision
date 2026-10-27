"""
helmet_pipeline_test.py
Real end-to-end test: vehicle detector + helmet detector on actual dataset images.
Writes results to stdout. Run with:
  .venv/Scripts/python.exe tests/helmet_pipeline_test.py
"""
import os
import sys
import shutil
import tempfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.database_manager import DatabaseManager
from violations.violation_manager import HelmetViolationDetector
from ultralytics import YOLO

SEP = "=" * 65
VIDEO_PATH = "data/input/helmet_test.mp4"
HELMET_MODEL = "models/helmet_detector.pt"
VEHICLE_MODEL = "models/vehicle_detector.pt"
LABELS_DIR = "datasets/motorcycle_helmet/train/labels"
IMAGES_DIR = "datasets/motorcycle_helmet/train/images"


def collect_images():
    nohelmet_imgs, helmet_imgs = [], []
    for fname in sorted(os.listdir(LABELS_DIR)):
        if not fname.endswith(".txt"):
            continue
        classes = set()
        with open(os.path.join(LABELS_DIR, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
        img_path = os.path.join(IMAGES_DIR, fname.replace(".txt", ".jpg"))
        if not os.path.exists(img_path):
            continue
        if 3 in classes and 2 in classes:
            nohelmet_imgs.append(img_path)
        elif 0 in classes and 3 not in classes and 2 in classes:
            helmet_imgs.append(img_path)
    return nohelmet_imgs, helmet_imgs


def build_test_video(nohelmet_imgs, helmet_imgs, out_path, fps=5, target_wh=(640, 640)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = target_wh
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    def write_frames(paths, count, repeat=4):
        for p in paths[:count]:
            img = cv2.imread(p)
            if img is None:
                continue
            resized = cv2.resize(img, (w, h))
            for _ in range(repeat):
                writer.write(resized)

    # Structure: helmet compliant → nohelmet → helmet compliant again (false-positive check)
    write_frames(helmet_imgs, 6, repeat=4)   # 24 frames of helmeted riders
    write_frames(nohelmet_imgs, 10, repeat=4) # 40 frames of helmetless riders
    write_frames(helmet_imgs[6:12], 6, repeat=4)  # 24 more helmeted riders
    writer.release()
    return out_path


def run_pipeline_on_video(video_path, helmet_det, vehicle_model, db, evidence_dir):
    """Full pipeline: vehicle detector → helmet detector on every frame."""
    vehicle_classes = {int(k): str(v).lower() for k, v in vehicle_model.names.items()}
    two_wheeler_ids = [
        k for k, v in vehicle_classes.items()
        if any(kw in v for kw in ("two", "motorcycle", "moto", "bike"))
    ]
    print(f"  Vehicle model classes: {vehicle_classes}")
    print(f"  Two-wheeler class IDs: {two_wheeler_ids}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {total_frames} frames @ {fps:.1f}fps")

    frame_idx = 0
    two_wheeler_total = 0
    violations = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps
        results = vehicle_model(frame, conf=0.15, verbose=False)
        boxes = results[0].boxes if results else None

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = vehicle_classes.get(cls_id, "?")
                veh_conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if cls_id in two_wheeler_ids:
                    two_wheeler_total += 1
                    track_id = frame_idx * 100 + cls_id  # synthetic unique track

                    h_result = helmet_det.check_motorcycle(
                        frame, (x1, y1, x2, y2),
                        track_id=track_id,
                        current_time=current_time,
                    )
                    if h_result:
                        ev_path = str(evidence_dir / f"nohelmet_f{frame_idx}_t{track_id}.jpg")
                        cv2.imwrite(ev_path, frame)
                        ev_id = db.insert_event(
                            event_type="NO_HELMET",
                            track_id=track_id,
                            vehicle_class=cls_name,
                            confidence=round(h_result["confidence"], 3),
                            evidence_path=ev_path,
                            review_status="pending",
                        )
                        violations.append({
                            "frame": frame_idx,
                            "time_s": round(current_time, 2),
                            "veh_class": cls_name,
                            "veh_conf": round(veh_conf, 3),
                            "helmet_conf": round(h_result["confidence"], 3),
                            "ev_path": ev_path,
                            "db_id": ev_id,
                        })
        frame_idx += 1

    cap.release()
    return violations, two_wheeler_total, frame_idx


def run_direct_model_test(helmet_model):
    """Test helmet model directly on source dataset images (bypasses vehicle detector)."""
    nohelmet_imgs, helmet_imgs = collect_images()

    print(f"\nTesting {min(10, len(nohelmet_imgs))} NO-HELMET ground-truth images:")
    detected_correct = 0
    for img_path in nohelmet_imgs[:10]:
        img = cv2.imread(img_path)
        results = helmet_model(img, conf=0.15, verbose=False)
        detections = []
        if results and results[0].boxes:
            for b in results[0].boxes:
                cid = int(b.cls[0].item())
                cname = helmet_model.names.get(cid, "?")
                cconf = float(b.conf[0].item())
                detections.append((cname, cconf))
        found_nohelmet = any("nohelmet" in d[0].lower() for d in detections)
        found_helmet = any(d[0].lower() == "helmet" for d in detections)
        det_str = ", ".join(f"{n}({c:.2f})" for n, c in detections) or "(none)"
        label = "CORRECT" if found_nohelmet and not found_helmet else ("CO-DETECT" if found_nohelmet else "MISSED")
        if found_nohelmet:
            detected_correct += 1
        print(f"  [{label}] {os.path.basename(img_path)}: {det_str}")

    print(f"\nTesting {min(10, len(helmet_imgs))} HELMET-COMPLIANT images (false-positive check):")
    false_positives = 0
    for img_path in helmet_imgs[:10]:
        img = cv2.imread(img_path)
        results = helmet_model(img, conf=0.15, verbose=False)
        detections = []
        if results and results[0].boxes:
            for b in results[0].boxes:
                cid = int(b.cls[0].item())
                cname = helmet_model.names.get(cid, "?")
                cconf = float(b.conf[0].item())
                detections.append((cname, cconf))
        found_nohelmet = any("nohelmet" in d[0].lower() for d in detections)
        found_helmet = any(d[0].lower() == "helmet" for d in detections)
        det_str = ", ".join(f"{n}({c:.2f})" for n, c in detections) or "(none)"
        # False positive: nohelmet detected WITHOUT helmet co-detection on a helmeted rider
        if found_nohelmet and not found_helmet:
            false_positives += 1
            label = "FP!"
        else:
            label = "OK"
        print(f"  [{label}] {os.path.basename(img_path)}: {det_str}")

    return detected_correct, min(10, len(nohelmet_imgs)), false_positives, min(10, len(helmet_imgs))


if __name__ == "__main__":
    print(SEP)
    print("SMART TRAFFIC VISION — HELMET DETECTION E2E REAL PIPELINE TEST")
    print(SEP)

    # Setup temp DB + evidence dir
    temp_dir = tempfile.mkdtemp(prefix="helmet_e2e_")
    db_path = os.path.join(temp_dir, "test.db")
    evidence_dir = Path(temp_dir) / "evidence"
    evidence_dir.mkdir()

    try:
        db = DatabaseManager(db_path=db_path)
        helmet_det = HelmetViolationDetector(
            model_path=HELMET_MODEL,
            conf_thresh=0.20,
            cooldown_seconds=1.5,  # short cooldown for dense test frames
        )

        print(f"\n[1] Helmet model loaded: {helmet_det.model is not None}")
        print(f"    Class map: {helmet_det.class_map}")

        print("\n[2] Loading vehicle detector...")
        vehicle_model = YOLO(VEHICLE_MODEL)
        print(f"    Vehicle classes: {dict(vehicle_model.names)}")

        # Build test video if not present
        nohelmet_imgs, helmet_imgs = collect_images()
        print(f"\n[3] Dataset: {len(nohelmet_imgs)} nohelmet images, {len(helmet_imgs)} helmet images")

        if not os.path.exists(VIDEO_PATH):
            print(f"    Building test video at {VIDEO_PATH}...")
            build_test_video(nohelmet_imgs, helmet_imgs, VIDEO_PATH)
        else:
            print(f"    Using existing test video: {VIDEO_PATH}")

        sz = os.path.getsize(VIDEO_PATH) / 1024
        cap = cv2.VideoCapture(VIDEO_PATH)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"    Video: {sz:.1f} KB, {nf} frames")

        # --- PHASE A: Full pipeline on test video ---
        print(f"\n{SEP}")
        print("[PHASE A] Full pipeline: vehicle detector -> helmet detector on test video")
        print(SEP)
        violations, tw_count, total_frames = run_pipeline_on_video(
            VIDEO_PATH, helmet_det, vehicle_model, db, evidence_dir
        )

        print(f"\n  Frames processed: {total_frames}")
        print(f"  Two-wheeler detections: {tw_count}")
        print(f"  NO_HELMET violations logged: {len(violations)}")

        if violations:
            print("\n  Violations detail:")
            for v in violations:
                print(f"    frame={v['frame']} t={v['time_s']}s | "
                      f"class={v['veh_class']} veh_conf={v['veh_conf']} | "
                      f"helmet_conf={v['helmet_conf']} | db_id={v['db_id']}")
                print(f"      evidence: {v['ev_path']}")
        else:
            print("\n  [NOTE] No violations via pipeline. This is expected if the vehicle")
            print("  detector does not classify these images as two-wheelers/motorcycles.")
            print("  Proceeding to direct helmet model test (Phase B).")

        # DB verification
        saved = db.get_events(event_type="NO_HELMET")
        print(f"\n  DB check: {len(saved)} NO_HELMET records in SQLite")

        # --- PHASE B: Direct helmet model test on source images ---
        print(f"\n{SEP}")
        print("[PHASE B] Direct helmet model inference on source dataset images")
        print(SEP)
        helmet_model = YOLO(HELMET_MODEL)
        correct, total_nh, fps_count, total_h = run_direct_model_test(helmet_model)

        # --- FINAL SUMMARY ---
        print(f"\n{SEP}")
        print("FINAL SUMMARY")
        print(SEP)
        print(f"Phase A — Full pipeline (vehicle det + helmet det):")
        print(f"  Two-wheelers detected by vehicle model: {tw_count}")
        print(f"  NO_HELMET violations generated:         {len(violations)}")
        print(f"  Evidence files saved:                   {len(violations)}")
        print(f"  SQLite records:                         {len(saved)}")
        print()
        print(f"Phase B — Direct helmet model on source images (conf>=0.15):")
        print(f"  NO-HELMET ground truth images: {correct}/{total_nh} detected correctly")
        print(f"  HELMET ground truth images:    {fps_count}/{total_h} false positives")
        print()

        if tw_count == 0:
            print("[IMPORTANT] The vehicle detector (vehicle_detector.pt) did NOT classify")
            print("  any objects as two-wheelers/motorcycles in the test video frames.")
            print("  This is the primary bottleneck — helmet detection only runs after a")
            print("  motorcycle is identified by the vehicle detector.")
            print()
            print("  Vehicle detector class names:", dict(vehicle_model.names))

        print("Evidence directory:", str(evidence_dir))
        print("DB path:", db_path)

    finally:
        # Only remove temp dir after results printed
        print(f"\n[Cleanup] Removing temp dir: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

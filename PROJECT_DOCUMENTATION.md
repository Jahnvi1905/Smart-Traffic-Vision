# Smart Traffic Vision — Complete Project Documentation

> **Version**: 1.0 | **Last Updated**: October 15, 2026  
> **Companion Document**: [`HANDOVER.md`](file:///c:/Users/ASUS/OneDrive/Desktop/CV_Project_Traffic/HANDOVER.md)  
> **Project Proposal**: [`SMART TRAFFIC VISION.pdf`](file:///c:/Users/ASUS/OneDrive/Desktop/CV_Project_Traffic/SMART%20TRAFFIC%20VISION.pdf)

---

## A. Executive Summary

Smart Traffic Vision is a Python-based AI traffic monitoring system that performs:

- **Multi-class vehicle detection** (Bus, Car, Three-Wheeler, Truck, Two-Wheeler, Van) using YOLO11 Nano fine-tuned on Indian traffic data.
- **Object tracking** with Ultralytics ByteTrack integration and centroid-distance fallback.
- **Virtual-line vehicle counting** using line-segment intersection on tracked centroids.
- **Helmet compliance detection** using a fine-tuned YOLO11 model on motorcycle rider head regions.
- **Wrong-way movement detection** using multi-frame trajectory displacement vectors.
- **Virtual stop-line / red-light policy enforcement** with configurable signal state.
- **License plate detection** (YOLO) and **OCR** (EasyOCR) with in-video plate-to-vehicle association.
- **SQLite event logging**, review workflow (Approve/Reject/Delete), and CSV export.
- **Streamlit SaaS-style web UI** with 4 pages, Light/Dark theme, and evidence preview.

---

## B. System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (app.py)                    │
│  ┌──────────┐ ┌──────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Home   │ │Analyze Traffic│ │Plate Scanner│ │Records/Reports│
│  └──────────┘ └──────┬───────┘ └──────┬──────┘ └──────┬──────┘ │
│                      │                │               │         │
│  ┌───────────────────▼────────────────▼───────────────▼───────┐ │
│  │              Backend Pipeline Orchestration                 │ │
│  └─┬──────────┬──────────┬──────────┬──────────┬─────────────┘ │
│    │          │          │          │          │                │
│  ┌─▼──────┐┌─▼───────┐┌─▼───────┐┌─▼───────┐┌─▼────────────┐ │
│  │Vehicle ││Vehicle  ││Violation││Plate    ││Database      │ │
│  │Detector││Counter  ││Manager  ││Det+OCR  ││Manager       │ │
│  └────────┘└─────────┘└─────────┘└─────────┘└──────────────┘ │
└─────────────────────────────────────────────────────────────────┘

Data Flow (Video):
  Upload → VehicleDetector.process_video()
         → ByteTrack or fallback tracking
         → VehicleCounter.update() per frame
         → ViolationManager.process_frame() per frame
             ├── HelmetViolationDetector.check_motorcycle()
             ├── WrongWayDetector.update()
             ├── RedLightDetector.check_crossing()
             └── PlateAssociationManager.process_frame()
         → Annotated H.264 MP4 output
         → DatabaseManager.insert_event() for counts/violations
```

---

## C. Directory Structure

```
CV_Project_Traffic/
├── app.py                          # Streamlit application (1805 lines)
├── requirements.txt                # Python dependencies
├── openh264-2.5.0-win64.dll        # OpenH264 codec for H.264 MP4 output
├── yolo11n.pt                      # Base COCO YOLO11 Nano weights (80 classes)
├── SMART TRAFFIC VISION.pdf        # Project proposal document
├── HANDOVER.md                     # Team handover document
├── PROJECT_DOCUMENTATION.md        # This document
│
├── models/
│   ├── vehicle_detector.pt         # 6-class Indian traffic detector (5.5 MB)
│   ├── plate_detector.pt           # License plate detector (6.2 MB)
│   ├── helmet_detector.pt          # Helmet compliance detector, 15 epochs (5.5 MB)
│   └── helmet_detector_backup.pt   # Baseline 1-epoch helmet checkpoint backup
│
├── detection/
│   └── vehicle_detector.py         # VehicleDetector class (290 lines)
│
├── counting/
│   └── vehicle_counter.py          # VehicleCounter with line-crossing logic (342 lines)
│
├── violations/
│   ├── __init__.py                 # Package exports
│   └── violation_manager.py        # HelmetViolationDetector, WrongWayDetector,
│                                   # RedLightDetector, PlateAssociationManager,
│                                   # TrafficViolationManager
│
├── ocr/
│   ├── plate_detector.py           # PlateDetector - YOLO-based plate localization
│   └── plate_ocr.py                # PlateOCR - EasyOCR text extraction
│
├── database/
│   └── database_manager.py         # SQLite DatabaseManager (261 lines)
│
├── tracking/
│   └── bytetrack.yaml              # ByteTrack tracker configuration
│
├── datasets/
│   ├── india_traffic/              # India Traffic Analysis v2 (901 images, 6 classes)
│   ├── india_traffic_detection/    # Cleaned split with verified data.yaml
│   └── motorcycle_helmet/          # Motorcycle & Helmet v4 (205 images, 5 classes)
│
├── data/
│   ├── input/                      # Sample media files
│   │   ├── sample.jpg              # Test image
│   │   ├── sample.mp4              # Test video (1080×1920, 30fps)
│   │   └── plate_test.jpg          # License plate test image
│   ├── output/                     # Processed outputs
│   │   └── violations/             # Violation evidence snapshots
│   └── traffic_violations.db       # Production SQLite database
│
├── tests/
│   └── test_plate_and_violations.py # Automated test suite (6 tests)
│
├── utils/
│   └── train_helmet.py             # Helmet model training script
│
└── runs/
    └── helmet/
        ├── smoke_test/             # 1-epoch baseline run
        └── train_15ep/             # 15-epoch final run
            ├── weights/best.pt     # Best checkpoint (deployed)
            ├── results.csv         # Per-epoch metrics
            └── *.png               # Training curves and confusion matrices
```

---

## D. Model Inventory

### D.1 Vehicle Detector (`models/vehicle_detector.pt`)

| Property | Value |
|---|---|
| Base Architecture | YOLO11 Nano (2.6M params, 6.4 GFLOPs) |
| Training Data | `datasets/india_traffic` (631 train / 180 val / 90 test) |
| Training Config | 5 epochs, batch 8, imgsz 384, CPU |
| Classes (6) | `Bus`, `Car`, `Three-Wheeler`, `Truck`, `Two-Wheeler`, `Van` |
| Validation mAP50 | 0.485 |
| Validation mAP50-95 | 0.318 |
| File Size | 5.5 MB |

### D.2 Helmet Detector (`models/helmet_detector.pt`)

| Property | Value |
|---|---|
| Base Architecture | YOLO11 Nano (2.6M params, 6.4 GFLOPs) |
| Training Data | `datasets/motorcycle_helmet` (163 train / 41 val / 1 test) |
| Training Config | 15 epochs, batch 8, imgsz 320, patience 10, CPU |
| Classes (5) | `Helmet`, `fd`, `motorcycle`, `nohelmet`, `platenumber` |
| Baseline (1 epoch) | mAP50=0.449, mAP50-95=0.198, P=0.264, R=0.680 |
| **Final (15 epochs)** | **mAP50=0.913, mAP50-95=0.671, P=0.847, R=0.812** |
| Improvement | +103% mAP50, +239% mAP50-95, +221% Precision |
| Best Epoch | Epoch 14 (mAP50=0.913, mAP50-95=0.671) |
| File Size | 5.5 MB |
| Backup | `models/helmet_detector_backup.pt` (1-epoch baseline) |

**Per-class metrics (best checkpoint):**

| Class | mAP50-95 |
|---|---|
| Helmet | 0.662 |
| motorcycle | 0.680 |

### D.3 Plate Detector (`models/plate_detector.pt`)

| Property | Value |
|---|---|
| Base Architecture | YOLO (Custom) |
| Classes (1) | `license_plate` |
| Verified Confidence | 0.949 on `plate_test.jpg` |
| File Size | 6.2 MB |

### D.4 Base YOLO (`yolo11n.pt`)

| Property | Value |
|---|---|
| Role | Pretrained seed weights for fine-tuning |
| Classes | 80 COCO classes |
| File Size | 5.6 MB |

---

## E. Dataset Inventory

### E.1 India Traffic Analysis v2

| Property | Value |
|---|---|
| Location | `datasets/india_traffic/` |
| Source | Roboflow (`kaceys-workspace/india_traffic_analysis_v2/2`) |
| License | CC BY 4.0 |
| Classes (6) | Bus, Car, Three-Wheeler, Truck, Two-Wheeler, Van |
| Train | 631 images / 631 labels |
| Validation | 180 images / 180 labels |
| Test | 90 images / 90 labels |
| Total | 901 images |
| Format | YOLOv5/YOLO11 (normalized xywh) |

### E.2 Motorcycle and Helmet v4

| Property | Value |
|---|---|
| Location | `datasets/motorcycle_helmet/` |
| Source | Roboflow (`traffic-violation/motorcycle-and-helmet/4`) |
| License | CC BY 4.0 |
| Classes (5) | Helmet, fd, motorcycle, nohelmet, platenumber |
| Train | 163 images / 163 labels |
| Validation | 41 images / 41 labels |
| Test | 1 image / 1 label |
| Total | 205 images |
| Class Balance (train) | Helmet: 186, motorcycle: 150, nohelmet: 27, platenumber: 15, fd: 1 |
| Note | `nohelmet` class is underrepresented; `fd` has only 1 sample |

---

## F. Algorithm Details

### F.1 Vehicle Detection Pipeline

1. **Image**: Load → YOLO inference with `target_class_ids` filter → annotate bounding boxes with class labels and confidence → return annotated image + class-wise counts.

2. **Video**: Read frames → skip every `frame_skip` frames → run YOLO `.track()` with ByteTrack (or `.predict()` without tracking) → update `VehicleCounter` → update `TrafficViolationManager` → annotate frame with boxes, labels, HUD, counting line → write to H.264 MP4 via OpenH264.

### F.2 Virtual-Line Counting (`VehicleCounter`)

- A horizontal counting line is placed at a configurable Y-percentage of the video height (default 55%).
- For each tracked vehicle, the previous and current centroid positions form a trajectory segment.
- Line-segment intersection test (`segments_intersect()`) determines if the vehicle crossed the counting line.
- Once counted, a `track_id` is added to `counted_ids` to prevent double-counting.
- When ByteTrack is disabled, a centroid-distance fallback tracker maintains associations across frames.

### F.3 Helmet Compliance Detection (`HelmetViolationDetector`)

- Triggered only for detected motorcycles/two-wheelers.
- Extracts the upper 50% of the motorcycle bounding box (rider head region).
- Runs `models/helmet_detector.pt` on the crop to detect `Helmet` vs `nohelmet`.
- If `nohelmet` is detected without a confirmed `Helmet`, a `NO_HELMET` violation is logged.
- Per-track deduplication prevents repeat violations for the same rider.

### F.4 Wrong-Way Detection (`WrongWayDetector`)

- Maintains per-track centroid history (up to 30 frames).
- Requires minimum 8 observations and 40px total displacement to reject stationary noise.
- Checks if the dominant movement direction is opposite to the configured `expected_direction`.
- Uses step-by-step consistency ratio (≥75%) to confirm sustained wrong-way movement.
- Supports four directions: `up`, `down`, `left`, `right`.

### F.5 Virtual Stop-Line / Red-Light Detection (`RedLightDetector`)

- A virtual stop line is placed at a configurable Y-position.
- The signal state is set manually by the operator (RED or GREEN).
- When signal state is RED and a tracked vehicle's centroid crosses the stop line between consecutive frames, a `RED_LIGHT` violation is logged.
- Per-track deduplication prevents repeat violations.
- **Note**: Optical traffic light classification is not implemented (requires signal lamps in camera FOV).

### F.6 Plate-to-Vehicle Association (`PlateAssociationManager`)

- Runs plate detection every N frames (configurable `detection_interval`, default 5) to reduce CPU load.
- For each detected plate, finds the most plausible vehicle using geometric containment and overlap scoring:
  - Plate area must be ≤40% of vehicle area.
  - Plate center must be inside vehicle box, or ≥60% overlap required.
  - Vertical position bias favors plates in the lower portion of the vehicle.
- Runs EasyOCR on the best crop per track.
- Requires ≥4 alphanumeric characters and ≥30% OCR confidence to accept plate text.
- High-confidence plates (≥70%) are cached and not re-read.
- Associated plate text is attached to violation events (replaces `"UNKNOWN"` in database).

### F.7 License Plate OCR Pipeline

- **Detection**: `PlateDetector` localizes plates using `models/plate_detector.pt`.
- **Cropping**: Bounding box coordinates are clamped to image boundaries.
- **OCR**: `PlateOCR` uses EasyOCR (English) to extract alphanumeric text.
- **Post-processing**: Text is cleaned (uppercase, alphanumeric + space only).
- Verified output: `"MH 20 EE 7602"` from `plate_test.jpg` at 0.476 OCR confidence.

---

## G. Database Specification

### G.1 Schema

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    track_id INTEGER,
    vehicle_class TEXT,
    event_type TEXT NOT NULL,      -- COUNT, PLATE_READ, NO_HELMET, WRONG_WAY, RED_LIGHT
    confidence REAL,
    plate_text TEXT,
    evidence_path TEXT,
    review_status TEXT DEFAULT 'pending'  -- pending, approved, rejected
);

CREATE INDEX idx_events_track_id ON events (track_id);
CREATE INDEX idx_events_event_type ON events (event_type);
CREATE INDEX idx_events_review_status ON events (review_status);
```

### G.2 Event Types

| Event Type | Description | When Logged |
|---|---|---|
| `COUNT` | Vehicle counting summary | After video processing completes |
| `PLATE_READ` | License plate OCR result | After plate scan in Number Plate Scanner tab |
| `NO_HELMET` | Motorcycle rider without helmet | During video processing (helmet detection enabled) |
| `WRONG_WAY` | Vehicle moving against expected direction | During video processing (wrong-way detection enabled) |
| `RED_LIGHT` | Vehicle crossing stop-line during RED signal | During video processing (red-light detection enabled) |

### G.3 API Methods

| Method | Parameters | Description |
|---|---|---|
| `insert_event()` | event_type, track_id, vehicle_class, confidence, plate_text, evidence_path, review_status, prevent_duplicates | Insert event with duplicate check |
| `get_events()` | event_type, review_status, track_id, limit, offset | Query events with filtering |
| `get_event_by_id()` | event_id | Retrieve single event |
| `update_review_status()` | event_id, new_status | Change review status |
| `delete_event()` | event_id | Remove event record |
| `export_to_csv()` | event_type | Export filtered events as CSV string |

### G.4 Current Production Data

- **File**: `data/traffic_violations.db`
- **Records**: 8 events (all `COUNT` type, all `approved`)
- **Evidence**: 8 MP4 files in `data/output/`, all verified existing on disk

---

## H. Streamlit Application (app.py)

### H.1 Pages

| Page | Function |
|---|---|
| **Home** | Welcome dashboard with system capabilities overview |
| **Analyze Traffic** | Upload image/video, configure detection parameters, run analysis, view results |
| **Number Plate Scanner** | Upload plate image, detect and crop plate, run OCR, save to database |
| **Records & Reports** | Browse all events, view evidence, review actions, CSV export |

### H.2 Configuration Options (Analyze Traffic)

- **Confidence Threshold**: 0.1–1.0 slider (default 0.35)
- **Model Selection**: Vehicle detector or base YOLO
- **Enable ByteTrack**: Toggle tracking on/off
- **Enable Vehicle Counting**: Toggle counting line
- **Enable Helmet Detection**: Toggle helmet violation check
- **Enable Wrong-Way Detection**: Toggle wrong-way violation check
- **Max Frames**: Limit video processing frames

### H.3 Theme

- Light and Dark themes toggle in sidebar
- All UI elements are readable in both themes

---

## I. Test Suite

### I.1 Automated Tests (`tests/test_plate_and_violations.py`)

| Test | Class | Status |
|---|---|---|
| `test_plate_inside_vehicle_bounding_box` | TestPlateAssociation | ✅ PASS |
| `test_plate_outside_all_vehicles` | TestPlateAssociation | ✅ PASS |
| `test_plate_closest_to_correct_vehicle_among_multiple` | TestPlateAssociation | ✅ PASS |
| `test_wrong_way_detector` | TestViolationDetectors | ✅ PASS |
| `test_red_light_crossing_detector` | TestViolationDetectors | ✅ PASS |
| `test_violation_manager_database_and_evidence` | TestViolationDetectors | ✅ PASS |

**Run command**: `.\.venv\Scripts\python.exe -m unittest tests/test_plate_and_violations.py`

### I.2 Manual Verification Results

| Workflow | Result |
|---|---|
| Image upload → detection → annotated image → class-wise counts | ✅ Verified |
| Video upload → detection → ByteTrack → counting → playable MP4 → counts | ✅ Verified (1.3 MB output) |
| Plate image → detection (0.949 conf) → OCR ("MH 20 EE 7602") | ✅ Verified |
| Database insert → evidence path validation → retrieval → CSV export | ✅ Verified |
| Helmet model training (15 epochs) → mAP50 improvement 0.449→0.913 | ✅ Verified |
| Wrong-way detector (trajectory-based) | ✅ Verified via unit test |
| Red-light detector (virtual stop-line) | ✅ Verified via unit test |
| Plate-to-vehicle association (geometric containment) | ✅ Verified via unit test |

---

## J. Training Scripts and Reproduction

### J.1 Helmet Model Training

**Script**: [`utils/train_helmet.py`](file:///c:/Users/ASUS/OneDrive/Desktop/CV_Project_Traffic/utils/train_helmet.py)

```bash
# Activate virtual environment first
.\.venv\Scripts\python.exe utils/train_helmet.py
```

**Configuration**:
- Pretrained weights: `yolo11n.pt`
- Dataset: `datasets/motorcycle_helmet/data.yaml`
- Epochs: 15 | Batch: 8 | Image size: 320 | Device: CPU
- Patience: 10 (early stopping)
- Output: `runs/helmet/train_15ep/weights/best.pt`

**Training Time**: ~5.6 minutes on CPU (Intel Core i5-10300H @ 2.50GHz)

### J.2 Vehicle Model

The vehicle detector was fine-tuned separately using similar YOLO training on `datasets/india_traffic/data.yaml`. The trained weights are in `models/vehicle_detector.pt`.

---

## K. Deployment and Setup

### K.1 Prerequisites

- **Python**: 3.10–3.13 (64-bit)
- **OS**: Windows 10/11 (tested), macOS, or Linux
- **Disk Space**: ~1.5 GB (venv + models + datasets)
- **RAM**: 4 GB minimum, 8 GB recommended

### K.2 Installation

```bash
# 1. Clone or copy the project
cd CV_Project_Traffic

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
# source .venv/bin/activate     # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import ultralytics; import streamlit; import easyocr; print('All dependencies OK')"
```

### K.3 Launch

```bash
# Start the application
.\.venv\Scripts\streamlit.exe run app.py

# Application opens at: http://localhost:8501
```

### K.4 Run Tests

```bash
.\.venv\Scripts\python.exe -m unittest tests/test_plate_and_violations.py
```

### K.5 Important Notes

- **Always use `.venv` Python**: The system Python (Anaconda) may have binary mismatches with pandas. Use `.\.venv\Scripts\python.exe` for all commands.
- **OpenH264 DLL**: The `openh264-2.5.0-win64.dll` must be in the project root for H.264 video output. Without it, the fallback codec (`mp4v`) produces less compatible MP4.
- **EasyOCR First Run**: The first OCR call downloads language model files (~40 MB). Subsequent runs use cached models.

---

## L. Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Image/video I/O, annotation, video writing |
| `ultralytics` | YOLO11 inference, training, and ByteTrack tracking |
| `numpy` | Array operations |
| `pandas` | Data manipulation for UI tables |
| `streamlit` | Web application framework |
| `easyocr` | Optical character recognition for license plates |
| `torch` | PyTorch backend (installed with ultralytics) |

---

## M. Requirement Verification Matrix

Audited against [`SMART TRAFFIC VISION.pdf`](file:///c:/Users/ASUS/OneDrive/Desktop/CV_Project_Traffic/SMART%20TRAFFIC%20VISION.pdf):

| # | Requirement | Status | Module | Evidence |
|---|---|---|---|---|
| 1 | Vehicle detection & classification | ✅ Implemented and verified | `detection/vehicle_detector.py` | 6 classes, verified on sample.jpg |
| 2 | Image analysis | ✅ Implemented and verified | `app.py` | Upload + annotated output + counts |
| 3 | Recorded video analysis | ✅ Implemented and verified | `vehicle_detector.py` | Playable H.264 MP4 output |
| 4 | Object tracking & unique IDs | ✅ Implemented and verified | `vehicle_detector.py` | ByteTrack + centroid fallback |
| 5 | Virtual-line vehicle counting | ✅ Implemented and verified | `vehicle_counter.py` | Line-segment intersection counting |
| 6 | Helmet detection | ✅ Implemented and verified | `violation_manager.py` | mAP50=0.913 after 15 epochs |
| 7 | Seat-belt detection | ⬜ Not applicable / future scope | N/A | Per proposal: "where camera conditions permit" |
| 8 | Red-light violation detection | ⚠️ Partially implemented | `violation_manager.py` | Virtual stop-line verified; optical signal N/A |
| 9 | Wrong-way movement detection | ✅ Implemented and verified | `violation_manager.py` | Multi-frame trajectory test passes |
| 10 | Number-plate detection | ✅ Implemented and verified | `ocr/plate_detector.py` | 0.949 confidence on plate_test.jpg |
| 11 | OCR & plate text extraction | ✅ Implemented and verified | `ocr/plate_ocr.py` | "MH 20 EE 7602" extracted |
| 12 | Plate-to-vehicle association | ✅ Implemented and verified | `violation_manager.py` | PlateAssociationManager with geometric matching |
| 13 | Evidence image/video capture | ✅ Implemented and verified | `violation_manager.py` | Snapshots saved to `data/output/violations/` |
| 14 | SQLite database logging & review | ✅ Implemented and verified | `database_manager.py` | 8 records, review workflow, CSV export |
| 15 | Traffic statistics & dashboard | ✅ Implemented and verified | `app.py` | Metric cards, bar chart, activity list |
| 16 | CSV/report export | ✅ Implemented and verified | `database_manager.py` | `export_to_csv()` method + UI download |
| 17 | Simulated challan / violation records | ✅ Implemented and verified | `app.py` | Capture → Review → Approve/Reject → CSV |
| 18 | Live CCTV / webcam support | ⬜ Not applicable / future scope | N/A | Proposal: "Future support for live CCTV" |
| 19 | Light/Dark theme & 4 UI pages | ✅ Implemented and verified | `app.py` | Home, Analyze, Plate Scanner, Records |
| 20 | Error handling & duplicate prevention | ✅ Implemented and verified | `database_manager.py` | Parameterized queries, duplicate check |

**Summary**: 16/20 requirements fully implemented and verified, 1 partially implemented (red-light without optical signal), 2 not applicable per proposal scope, 1 not applicable per camera limitations.

---

## N. Known Limitations and Future Work

### N.1 Current Limitations

1. **Vehicle Detector mAP**: mAP50=0.485 is moderate. More training data or larger model architecture would improve accuracy.
2. **Helmet Dataset Imbalance**: `nohelmet` class has only 27 training samples; `fd` has 1 sample. More data would improve reliability.
3. **No Optical Traffic Light Detection**: The virtual red-light system uses manual signal state. Actual traffic light classification requires signal lamps in the camera field of view.
4. **CPU-Only Inference**: All models run on CPU. GPU acceleration would significantly improve video processing speed.
5. **No Real-Time Streaming**: The system processes uploaded files only. Live CCTV/webcam feed requires additional integration.
6. **Plate OCR Accuracy**: EasyOCR confidence on Indian plates is moderate (~47%). Specialized Indian plate OCR models would improve accuracy.

### N.2 Future Enhancements

1. **GPU Acceleration**: Enable CUDA for faster inference.
2. **Live Camera Integration**: RTSP/WebRTC streaming support.
3. **Optical Traffic Light Detection**: Train a classifier for red/green/yellow signal states.
4. **Seat-Belt Detection**: Requires high-resolution interior-visible camera feeds.
5. **More Training Data**: Augment helmet and vehicle datasets for better generalization.
6. **Cloud Deployment**: Docker containerization and cloud hosting.

---

## O. Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `pandas` binary mismatch error | System Anaconda Python conflicts | Always use `.\.venv\Scripts\python.exe` |
| Blank/0:00 output video | Missing OpenH264 DLL or codec issue | Ensure `openh264-2.5.0-win64.dll` is in project root |
| Zero vehicle counts | Counting line not crossed within processed frames | Increase `max_frames` or use longer video |
| EasyOCR encoding error on Windows | Unicode charmap issue | Already handled: `verbose=False` in `PlateOCR.__init__` |
| "Module not found" errors | Virtual environment not activated | Run `.\.venv\Scripts\Activate.ps1` first |
| Port conflict on 8501 | Another Streamlit instance running | Use `streamlit run app.py --server.port 8502` |

---

## P. Changelog

| Date | Change | Files Modified |
|---|---|---|
| 2026-10-15 | Helmet model retrained: 1→15 epochs (mAP50: 0.449→0.913) | `models/helmet_detector.pt`, `utils/train_helmet.py` |
| 2026-10-15 | Added PlateAssociationManager for in-video plate-to-vehicle matching | `violations/violation_manager.py`, `violations/__init__.py` |
| 2026-10-15 | Integrated plate association into TrafficViolationManager | `violations/violation_manager.py`, `app.py` |
| 2026-10-15 | Added virtual red-light check integration in process_frame loop | `violations/violation_manager.py` |
| 2026-10-15 | Added `export_to_csv()` to DatabaseManager | `database/database_manager.py` |
| 2026-10-15 | Created automated test suite (6 tests, all passing) | `tests/test_plate_and_violations.py` |
| 2026-10-15 | Created PROJECT_DOCUMENTATION.md | `PROJECT_DOCUMENTATION.md` |

---

## Q. Contacts and References

- **Project Proposal**: `SMART TRAFFIC VISION.pdf` (included in project root)
- **YOLO Documentation**: https://docs.ultralytics.com/
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **Streamlit**: https://docs.streamlit.io/
- **ByteTrack**: https://github.com/ifzhang/ByteTrack
- **India Traffic Dataset**: https://universe.roboflow.com/kaceys-workspace/india_traffic_analysis_v2
- **Motorcycle Helmet Dataset**: https://universe.roboflow.com/traffic-violation/motorcycle-and-helmet

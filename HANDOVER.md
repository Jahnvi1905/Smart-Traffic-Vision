# Smart Traffic Vision — Technical Handover Document

> **For**: Team Members and Future Developers  
> **Detailed Documentation**: See [`PROJECT_DOCUMENTATION.md`](file:///c:/Users/ASUS/OneDrive/Desktop/CV_Project_Traffic/PROJECT_DOCUMENTATION.md) for comprehensive system documentation.

---

## 1. Quick Start

```bash
# 1. Activate the virtual environment (IMPORTANT: do not use system Python)
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
# source .venv/bin/activate     # Linux/macOS

# 2. Launch the application
streamlit run app.py
# Opens at http://localhost:8501

# 3. Run automated tests
python -m unittest tests/test_plate_and_violations.py
```

> **⚠️ Always use `.\.venv\Scripts\python.exe`** — the system Anaconda Python has binary mismatches with pandas.

---

## 2. Project Overview

Smart Traffic Vision is a Python + Streamlit AI traffic monitoring system:

- **Vehicle Detection**: 6-class YOLO11 Nano (Bus, Car, Three-Wheeler, Truck, Two-Wheeler, Van)
- **Object Tracking**: ByteTrack integration + centroid-distance fallback
- **Vehicle Counting**: Virtual-line crossing with line-segment intersection
- **Helmet Compliance**: YOLO11 model detecting Helmet vs nohelmet on rider crops
- **Wrong-Way Detection**: Multi-frame trajectory displacement analysis
- **Virtual Red-Light**: Policy-based stop-line crossing detection
- **License Plate OCR**: YOLO plate detection + EasyOCR text extraction
- **Plate-to-Vehicle Association**: Geometric containment matching in video pipeline
- **Database**: SQLite event logging with review workflow and CSV export
- **UI**: 4-page Streamlit app with Light/Dark theme

---

## 3. Requirement Verification Matrix

Audited against `SMART TRAFFIC VISION.pdf`:

| # | Requirement | Status |
|---|---|---|
| 1 | Vehicle detection & classification | ✅ Verified |
| 2 | Image analysis | ✅ Verified |
| 3 | Recorded video analysis | ✅ Verified |
| 4 | Object tracking & unique IDs | ✅ Verified |
| 5 | Virtual-line vehicle counting | ✅ Verified |
| 6 | Helmet detection | ✅ Verified (mAP50=0.913) |
| 7 | Seat-belt detection | ⬜ Future scope (per proposal) |
| 8 | Red-light violation detection | ⚠️ Partial (virtual stop-line, no optical signal) |
| 9 | Wrong-way movement detection | ✅ Verified |
| 10 | Number-plate detection | ✅ Verified (0.949 conf) |
| 11 | OCR & plate text extraction | ✅ Verified ("MH 20 EE 7602") |
| 12 | Plate-to-vehicle association | ✅ Verified (geometric matching) |
| 13 | Evidence capture | ✅ Verified |
| 14 | SQLite logging & review | ✅ Verified |
| 15 | Traffic statistics dashboard | ✅ Verified |
| 16 | CSV export | ✅ Verified |
| 17 | Violation records lifecycle | ✅ Verified |
| 18 | Live CCTV / webcam | ⬜ Future scope (per proposal) |
| 19 | Light/Dark theme & 4 pages | ✅ Verified |
| 20 | Error handling & deduplication | ✅ Verified |

**Result**: 16/20 fully implemented, 1 partial, 2 future scope, 1 N/A.

---

## 4. Model Inventory

| Model File | Task | Classes | Key Metric |
|---|---|---|---|
| `models/vehicle_detector.pt` | Vehicle Detection | 6 | mAP50=0.485 |
| `models/helmet_detector.pt` | Helmet Compliance | 5 | **mAP50=0.913** (15 epochs) |
| `models/plate_detector.pt` | Plate Localization | 1 | Conf=0.949 |
| `yolo11n.pt` | Pretrained base | 80 | COCO baseline |
| `models/helmet_detector_backup.pt` | Baseline backup | 5 | mAP50=0.449 (1 epoch) |

---

## 5. Key Modules

| Module | File | Responsibility |
|---|---|---|
| VehicleDetector | `detection/vehicle_detector.py` | YOLO inference, tracking, video processing |
| VehicleCounter | `counting/vehicle_counter.py` | Line-crossing counting with centroid tracking |
| HelmetViolationDetector | `violations/violation_manager.py` | Rider crop → helmet/nohelmet classification |
| WrongWayDetector | `violations/violation_manager.py` | Multi-frame trajectory wrong-way detection |
| RedLightDetector | `violations/violation_manager.py` | Virtual stop-line crossing detection |
| PlateAssociationManager | `violations/violation_manager.py` | In-video plate-to-vehicle geometric matching |
| TrafficViolationManager | `violations/violation_manager.py` | Orchestrates all violation checks per frame |
| PlateDetector | `ocr/plate_detector.py` | YOLO-based plate bounding box detection |
| PlateOCR | `ocr/plate_ocr.py` | EasyOCR text extraction from plate crops |
| DatabaseManager | `database/database_manager.py` | SQLite CRUD, review workflow, CSV export |

---

## 6. Database

- **File**: `data/traffic_violations.db`
- **Current Records**: 8 events (all COUNT, all approved)
- **Event Types**: COUNT, PLATE_READ, NO_HELMET, WRONG_WAY, RED_LIGHT
- **Review Statuses**: pending, approved, rejected

```sql
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
```

---

## 7. Test Suite

All 6 automated tests pass:

```
$ python -m unittest tests/test_plate_and_violations.py
......
----------------------------------------------------------------------
Ran 6 tests in 0.199s
OK
```

Tests cover: plate-to-vehicle association (3 geometric cases), wrong-way detection, red-light crossing, and end-to-end violation database/evidence persistence.

---

## 8. Known Limitations

1. Vehicle detector mAP50 is moderate (0.485); more training data would help.
2. Helmet dataset has class imbalance (nohelmet: 27 samples, fd: 1 sample).
3. No optical traffic light classification (requires signal lamps in camera FOV).
4. CPU-only inference (no GPU acceleration configured).
5. Plate OCR confidence is moderate (~47%) on Indian plates.
6. No live camera streaming (uploaded files only).

---

## 9. What Was Changed in Final Pass (Oct 15, 2026)

1. **Helmet model retrained**: 1 epoch → 15 epochs. mAP50 improved from 0.449 to 0.913 (+103%).
2. **PlateAssociationManager added**: Geometric plate-to-vehicle matching in video pipeline.
3. **TrafficViolationManager enhanced**: Red-light integration into frame loop, plate text attached to violation events.
4. **DatabaseManager**: Added `export_to_csv()` method.
5. **Automated tests**: Created `tests/test_plate_and_violations.py` (6 tests, all passing).
6. **Documentation**: Created `PROJECT_DOCUMENTATION.md` (17 sections) and updated `HANDOVER.md`.

---

## 10. For More Details

See [`PROJECT_DOCUMENTATION.md`](file:///c:/Users/ASUS/OneDrive/Desktop/CV_Project_Traffic/PROJECT_DOCUMENTATION.md) for:
- Complete architecture diagrams
- Algorithm details for all detection and violation modules
- Per-epoch training metrics
- Dataset class distribution analysis
- Troubleshooting guide
- Full changelog

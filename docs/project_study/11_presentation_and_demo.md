# Document 11: Project Presentation Deck, Live Demo Script & Troubleshooting Guide

## 1. Presentation Slide Outline (12 Slides)

This structured 10–12 slide deck is designed for an 8–10 minute capstone presentation to an academic faculty panel or industry evaluator:

```
+-------------------------------------------------------------------------------+
|  Slide 1: Title & Team Credentials      |  Slide 7: Spatial Helmet Solution   |
|  Slide 2: Problem Statement             |  Slide 8: License Plate OCR & ANPR  |
|  Slide 3: Proposed Solution Overview    |  Slide 9: FastAPI & Database Schema |
|  Slide 4: System Architecture & Dataflow|  Slide 10: React Dashboard & UI     |
|  Slide 5: Object Detection (YOLO11n)    |  Slide 11: Quantitative Evaluation  |
|  Slide 6: Multi-Object Tracking (Byte)  |  Slide 12: Conclusion & Future Scope|
+-------------------------------------------------------------------------------+
```

---

### Slide-by-Slide Detailed Content

#### Slide 1: Title & Team
- **Title**: Smart Traffic Vision: Intelligent Multi-Class Traffic Monitoring & Automated Safety Enforcement System
- **Subtitle**: Computer Vision & Deep Learning Pipeline with FastAPI and React 19
- **Team**: Main Developer (Lead Architect), Ruchi (Dataset Research), Afzal (Quality Assurance)
- **Time**: 30 seconds

#### Slide 2: The Urban Traffic Problem
- **Key Points**:
  - Manual CCTV surveillance causes operator fatigue; critical safety infractions go unnoticed.
  - High traffic heterogeneity in developing nations (India) with mixed vehicle classes and chaotic lane discipline.
  - Legacy systems operate in silos (counting separated from violation auditing and license plate recognition).
- **Time**: 45 seconds

#### Slide 3: Smart Traffic Vision Solution
- **Key Points**:
  - Unified automated platform: Vehicle Detection + ByteTrack Tracking + Virtual-Line Counting + Safety Violation Enforcement + ANPR.
  - Edge-capable lightweight deep learning inference running on commodity hardware.
  - Real-time tamper-evident evidence archiving for legal auditability.
- **Time**: 45 seconds

#### Slide 4: System Architecture
- **Visual**: Three-Tier Architecture Mermaid Diagram (from Document 01).
- **Key Points**:
  - Decoupled architecture separating low-level vision loops from client presentation.
  - High-concurrency asynchronous FastAPI backend serving static media and REST endpoints.
  - Single-Page Application (SPA) dashboard built with React 19, TypeScript, and Vite.
- **Time**: 60 seconds

#### Slide 5: Deep Learning Vehicle Detection (YOLO11n)
- **Visual**: Bounding box diagram with confidence scores and class breakdown.
- **Key Points**:
  - Ultralytics `YOLO11n` nano architecture (2.6M parameters, ~5.4 MB weights).
  - Fine-tuned on Indian Traffic Dataset across 6 classes (`Bus`, `Car`, `Three-Wheeler`, `Truck`, `Two-Wheeler`, `Van`).
  - Anchor-free convolutional heads operating at 30+ FPS.
- **Time**: 45 seconds

#### Slide 6: Multi-Object Tracking & Line Counting
- **Visual**: ByteTrack two-stage matching flowchart & CCW vector intersection diagram.
- **Key Points**:
  - ByteTrack recovers occluded vehicles by matching low-score detection pools via Kalman filter predictions.
  - Counter-Clockwise (`ccw`) vector geometry tests trajectory line crossings without double-counting.
  - Generates synchronized per-frame telemetry.
- **Time**: 60 seconds

#### Slide 7: Dedicated Spatial Helmet-First Detection (Bottleneck Fix)
- **Visual**: Diagram showing motorcycle search zone and orange head box.
- **Key Points**:
  - Overhead vehicle detectors miss frontal motorcycles, causing standard crop pipelines to fail.
  - Solved via a **dedicated spatial helmet-first path**: `helmet_detector.pt` detects motorcycles and heads across the full frame.
  - Evaluates multi-rider scenes independently (compliant riders never suppress genuine violations).
  - Achieved **93.3% recall** and **0.0% false alarm rate** on 30 ground-truth images.
- **Time**: 90 seconds (Highlight of presentation!)

#### Slide 8: License Plate Localization & EasyOCR
- **Visual**: Plate crop with CRAFT heatmaps and normalized text output.
- **Key Points**:
  - Two-stage ANPR: YOLO plate detector + EasyOCR (CRAFT + ResNet-BiLSTM-CTC).
  - Conservative normalization: strips mounting bracket noise (`[]{}()`).
  - Zero hallucination policy: flags readings $< 0.50$ confidence as `low_confidence` for human review.
- **Time**: 45 seconds

#### Slide 9: FastAPI Backend & SQLite Hardening
- **Visual**: Database schema diagram with `events` table and migration arrows.
- **Key Points**:
  - Asynchronous endpoints, Pydantic type validation, and automatic OpenAPI docs.
  - SQLite schema hardening: `COUNT` analytical events strictly segregated from safety violations.
  - Built-in deduplication queries preventing duplicate records during cooldown.
- **Time**: 45 seconds

#### Slide 10: Modern React Command Center Dashboard
- **Visual**: Screenshots of Overview KPIs, Traffic Analysis player, and Evidence Modal.
- **Key Points**:
  - Dark-themed control-room design system with Recharts SVG analytics.
  - Live video player synchronized with per-frame count telemetry.
  - Optimistic UI updates when approving or rejecting violation records.
- **Time**: 45 seconds

#### Slide 11: Testing, Verification & Results
- **Visual**: Test summary table (52/52 Tests Passed in 8.5s).
- **Key Points**:
  - 52 automated pytest unit and integration tests passing.
  - Verified on 30 ground-truth dataset images (93.3% recall, 0% FPR).
  - Verified end-to-end on `helmet_test.mp4` generating 3 distinct non-duplicate evidence events.
- **Time**: 45 seconds

#### Slide 12: Conclusion & Future Scope
- **Key Points**:
  - Summary: Fully operational, modular, edge-ready intelligent traffic auditing platform.
  - Future Roadmap: Physical traffic light hardware integration, multi-camera vehicle re-identification, edge TensorRT acceleration, and automated challan SMS gateways.
- **Time**: 30 seconds

---

## 2. Step-by-Step Live Demo Script (5–7 Minutes)

Follow this exact spoken script and screen navigation sequence during the demonstration:

```
[Start: Diagnostics] --> [Overview Dashboard] --> [Traffic Video Analysis] --> [Safety Violations Modal] --> [Plate Scanner] --> [CSV Export]
```

### Pre-Demo Setup (Do this before examiners enter)
1. In Terminal 1: Start FastAPI backend:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```
2. In Terminal 2: Start React frontend:
   ```powershell
   cd frontend && npm run dev
   ```
3. Open browser to `http://localhost:5173`.

---

### Step 1: System Health & Diagnostics (1 Minute)
- **Screen Action**: Navigate to **Diagnostics** in the sidebar.
- **Spoken Cue (English)**:
  > *"Respected examiners, we begin our demonstration at the Diagnostics console. Here we verify that our FastAPI backend is online, all four deep learning model weight files are loaded, SQLite database connectivity is active, and our OpenH264 video codec is verified. This ensures our entire edge system is healthy before processing traffic feeds."*
- **Expected UI**: Green status badges (`HEALTHY`, `CONNECTED`, `LOADED`).

---

### Step 2: Overview Command Center (1 Minute)
- **Screen Action**: Navigate to **Overview** tab.
- **Spoken Cue (English)**:
  > *"The Overview dashboard provides high-level situational awareness. The top KPI cards show total cumulative vehicles counted, active safety violations, pending review cases, and camera uptime. Below, Recharts renders real-time visual analytics: the left chart displays traffic density categorized into the 6 Indian vehicle classes, while the right chart displays the distribution of safety violations. Notice that session count summaries are strictly isolated from safety violation figures."*
- **Expected UI**: 4 StatCards, BarChart, DonutChart, and Recent Events list.

---

### Step 3: Video Analysis with Synchronized Telemetry (2 Minutes)
- **Screen Action**: Navigate to **Traffic Analysis**. Click **Upload Video** and select `data/input/helmet_test.mp4`. Ensure **Helmet Detection** is checked, and click **Start Analysis**.
- **Spoken Cue (English)**:
  > *"Now we demonstrate our core video processing pipeline. We are uploading `helmet_test.mp4`, a video containing a mix of compliant and unhelmeted motorcycle riders. When I click 'Start Analysis', the video is ingested by our backend. OpenCV demuxes frames, YOLO and ByteTrack track vehicles, our dedicated helmet detector analyzes rider heads across the frame, and our counting line records line crossings.*
  > *(Video loads)*
  > *As the processed video plays, notice the telemetry HUD: the vehicle count and category breakdowns dynamically update in perfect synchronization with the video playback timestamp."*
- **Expected UI**: Video player playing processed MP4 with bounding boxes, counting line, and dynamic telemetry cards updating below.

---

### Step 4: Safety Violations & Evidence Inspection (1.5 Minutes)
- **Screen Action**: Navigate to **Safety Violations** in the sidebar. Filter by `NO_HELMET`. Click on the latest row to open the Modal.
- **Spoken Cue (English)**:
  > *"In the Safety Violations console, all detected infractions appear in an auditable ledger. Here we see our newly detected `NO_HELMET` events. Clicking on this event opens our high-resolution evidence modal. Notice the tamper-evident visual annotations: a thick red box encloses the motorcycle, a distinct orange box identifies the rider's bare head labeled 'NO HELMET', and bottom banners log the exact timestamp and model confidence.*
  > *To complete the administrative workflow, the supervisor can click 'Approve' or 'Reject'. Notice how the status badge updates instantly through an optimistic UI update without refreshing the page."*
- **Expected UI**: Fullscreen modal showing high-contrast annotated evidence JPEG; clicking `Approve` changes badge to green.

---

### Step 5: License Plate Scanner & ANPR (1 Minute)
- **Screen Action**: Navigate to **Plate Scanner**. Upload a vehicle plate image. Click **Scan Plate**.
- **Spoken Cue (English)**:
  > *"Next, we inspect our Automatic Number Plate Recognition module. Our plate detector localizes the plate coordinates, crops the sub-image, and feeds it to EasyOCR. Here we see the extracted alphanumeric text string, alongside raw unedited text, confidence percentage, and our audit status badge. If confidence falls below 50%, our system conservatively flags it as 'low confidence' rather than guessing characters."*
- **Expected UI**: Display of cropped plate, extracted text (e.g. `MH 20 EE 7602`), and confidence badge.

---

### Step 6: Records Ledger & CSV Export (30 Seconds)
- **Screen Action**: Navigate to **Records & Reports**. Click **Export to CSV**.
- **Spoken Cue (English)**:
  > *"Finally, the Records & Reports view acts as our permanent legal ledger. Operators can search by plate or filter by violation type. Clicking 'Export to CSV' streams an aggregated CSV export directly from SQLite for municipal record-keeping."*
- **Expected UI**: Browser automatically downloads `traffic_violations_export.csv`.

---

## 3. Common Demo Failure Modes & Rapid Troubleshooting

| Failure / Error Symptom | Root Cause | Exact Instant Fix |
| :--- | :--- | :--- |
| **`Connection Refused` on API calls** | FastAPI backend is not running on port 8000 | Open terminal, run `.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000` |
| **Port 8000 already in use (`WinError 10048`)** | An orphaned Python process is holding port 8000 | In PowerShell: `Get-Process python \| Stop-Process -Force`, then restart uvicorn |
| **Video fails to play in browser (black box)** | VideoWriter failed or missing OpenH264 DLL | Ensure `openh264-2.5.0-win64.dll` is present in project root directory |
| **0 NO_HELMET violations logged after editing code** | Running Uvicorn server has old code in memory | Restart Uvicorn server in terminal (press `Ctrl+C` and restart) to load updated Python modules |
| **Vite proxy error (`ECONNREFUSED 127.0.0.1:8000`)** | Frontend launched before backend started | Ensure backend is running first; refresh browser at `http://localhost:5173` |

---

## 4. Individual Team Speaking Points (Viva Day)

### Main Developer (Lead Architect)
> *"My primary responsibility was engineering the complete three-tier architecture. I integrated the YOLO11n object detectors, implemented ByteTrack multi-object tracking, and designed the vector-intersection virtual line counter. When we identified the pipeline bottleneck where overhead vehicle detectors missed frontal motorcycles, I redesigned the system to use a dedicated spatial helmet-first path where the helmet model self-bootstraps motorcycle detection and rider bounding. I also built the FastAPI backend, the SQLite database layer, the React 19 dashboard, and authored the 52-test automated test suite."*

### Ruchi (Dataset Research)
> *"My focus was on dataset discovery, annotation auditing, and ground-truth validation. I sourced our Indian traffic dataset (901 images across 6 classes) and our motorcycle helmet dataset (205 images across 5 classes) from Roboflow. I inspected the annotation label files to study multi-rider co-occurrences, and segregated our 30-image ground-truth verification set used to prove our 93.3% recall and 0% false positive rate. I also assisted with background literature review and presentation materials."*

### Afzal (Quality Assurance)
> *"My role focused on quality assurance, system testing, and operational documentation. I verified our installation and runtime dependencies, conducted manual functional testing across all FastAPI endpoints, inspected generated visual evidence snapshots to ensure annotation clarity, tested cross-browser video player playback, and helped coordinate our demonstration script and presentation documentation."*

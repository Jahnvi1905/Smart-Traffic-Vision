# Smart Traffic Vision 🚦

An AI-powered Computer Vision traffic monitoring and safety violation intelligence platform. Features real-time vehicle classification, ByteTrack multi-object tracking, dynamic virtual line counting with class aggregation, number plate localization and OCR, automated safety violation auditing (helmet, wrong-way, virtual stop-line), evidence archiving, a high-performance FastAPI backend, and a modern dark traffic-control dashboard built with React, Vite, and TypeScript.

## Project Structure

```text
CV_Project_Traffic/
├── frontend/              # Modern React + Vite + TypeScript Dashboard (Phase 3)
│   ├── src/
│   │   ├── api/           # Typed API client & response types
│   │   ├── components/    # Reusable control center UI components (StatCard, Modal, etc.)
│   │   ├── pages/         # Command Center views
│   │   │   ├── OverviewPage.tsx        # High-level KPIs & category breakdown
│   │   │   ├── TrafficAnalysisPage.tsx # Video/photo player with synchronized telemetry
│   │   │   ├── SafetyViolationsPage.tsx# Review lifecycle & evidence inspection
│   │   │   ├── PlateScannerPage.tsx    # License plate localization & OCR
│   │   │   ├── RecordsReportsPage.tsx  # Database ledger & CSV export
│   │   │   └── DiagnosticsPage.tsx     # Hardware, model weights & DB status
│   │   ├── index.css      # Dark control-room design tokens & styles
│   │   └── App.tsx        # Shell navigation & route controller
│   ├── package.json
│   └── vite.config.ts
├── api/                   # FastAPI Backend (Phase 2)
│   ├── main.py            # Application entrypoint & CORS
│   ├── schemas.py         # Pydantic request/response models
│   ├── dependencies.py    # Database & model dependency providers
│   ├── utils.py           # Safe media URLs, file validation & streams
│   └── routes/            # Modular route controllers
│       ├── health.py      # /api/health & /api/diagnostics
│       ├── stats.py       # /api/stats/overview
│       ├── traffic.py     # /api/traffic/analyze-video & analyze-image
│       ├── plates.py      # /api/plates/scan
│       ├── violations.py  # /api/violations & /api/violations/{id}/review
│       └── reports.py     # /api/reports/export
├── counting/              # Vehicle counting & per-frame telemetry module
├── database/              # SQLite database manager & event logging
├── detection/             # YOLO vehicle detector module
├── ocr/                   # License plate detector & EasyOCR engine
├── violations/            # Safety violation suite (helmet, wrong-way, red-light)
├── data/                  # Media and database storage
│   ├── input/             # Source input media (sample.mp4, etc.)
│   ├── output/            # Processed videos, plates, and telemetry
│   └── output/violations/ # High-contrast annotated violation evidence
├── models/                # Trained YOLO weights (.pt)
├── tests/                 # Automated test suites (25 unit tests)
│   ├── test_phase1_pipeline.py
│   ├── test_plate_and_violations.py
│   └── test_api.py
├── app.py                 # Streamlit dashboard interface (preserved)
└── requirements.txt       # Python dependencies
```

## How to Run

### 1. Backend Environment Setup

Activate your Python virtual environment and install dependencies:
```bash
python -m pip install -r requirements.txt
```

### 2. Launch FastAPI Backend

Run the FastAPI backend server using Uvicorn:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc Documentation: `http://127.0.0.1:8000/redoc`

### 3. Launch React Control Center Dashboard (Recommended)

In a separate terminal, install dependencies and start the Vite dev server:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser. All API requests and static media requests are proxied directly to the FastAPI backend.

To build the production bundle:
```bash
cd frontend
npm run build
```

### 4. Launch Streamlit Application (Legacy Dashboard)

```bash
streamlit run app.py
```

### 5. Run Automated Test Suite

Execute all 25 automated tests across core computer vision pipelines and FastAPI endpoints:
```bash
python -m unittest discover -s tests
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Backend health check and operational status |
| `GET` | `/api/diagnostics` | System diagnostics (model weights, SQLite connection, directories, OpenH264) |
| `GET` | `/api/stats/overview` | Aggregated traffic volume, safety violation totals, category counts, and recent activity |
| `POST` | `/api/traffic/analyze-video` | Upload video for detection, ByteTrack tracking, counting, and per-frame telemetry |
| `POST` | `/api/traffic/analyze-image` | Upload single photo for YOLO vehicle classification and localized bounding boxes |
| `POST` | `/api/plates/scan` | License plate localization and EasyOCR text extraction |
| `GET` | `/api/violations` | Searchable, filterable, paginated violation records (COUNT events excluded) |
| `PATCH` | `/api/violations/{id}/review` | Update violation review status (`pending`, `approved`, `rejected`) |
| `GET` | `/api/reports/export` | Download filtered violation records as CSV |
| `GET` | `/media/output/{filename}` | Serve processed output videos and images |
| `GET` | `/media/violations/{filename}` | Serve high-contrast annotated violation snapshots |

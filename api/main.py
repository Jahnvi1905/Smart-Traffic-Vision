import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import health, stats, traffic, plates, violations, reports

app = FastAPI(
    title="Smart Traffic Vision API",
    version="1.0.0",
    description="High-performance backend API for Smart Traffic Vision monitoring, analytics, and safety enforcement."
)

# ── CORS Configuration for Local React & Vite Development ────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ensure Required Directories Exist ─────────────────────────────────────────
OUTPUT_DIR = Path("data/output")
VIOLATIONS_DIR = Path("data/output/violations")
INPUT_DIR = Path("data/input")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIOLATIONS_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Mount Safe Static Media Paths ─────────────────────────────────────────────
# Mount the violations directory first so /media/violations routes cleanly
app.mount("/media/violations", StaticFiles(directory=str(VIOLATIONS_DIR)), name="violations")
app.mount("/media/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# ── Include API Routers ────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(stats.router)
app.include_router(traffic.router)
app.include_router(plates.router)
app.include_router(violations.router)
app.include_router(reports.router)


@app.get("/")
def root():
    return {
        "message": "Smart Traffic Vision API is running.",
        "docs": "/docs",
        "health": "/api/health",
        "diagnostics": "/api/diagnostics"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)

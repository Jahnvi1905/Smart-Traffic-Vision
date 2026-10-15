from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "motorcycle_helmet" / "data.yaml"
WEIGHTS = ROOT / "yolo11n.pt"

if not DATA.exists():
    raise FileNotFoundError(f"Dataset config not found: {DATA}")
if not WEIGHTS.exists():
    raise FileNotFoundError(f"Pretrained weights not found: {WEIGHTS}")

model = YOLO(str(WEIGHTS))
results = model.train(
    data=str(DATA),
    epochs=15,
    imgsz=320,
    batch=8,
    device="cpu",
    workers=0,
    project=str(ROOT / "runs" / "helmet"),
    name="train_15ep",
    exist_ok=True,
    patience=10,
    save=True,
)
print("Helmet training 15 epochs completed.")


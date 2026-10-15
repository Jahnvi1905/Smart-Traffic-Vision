"""Train YOLO11n on the repaired india_traffic_detection dataset.

Features:
- Auto-detects GPU (CUDA) or falls back to CPU.
- Uses balanced settings (reasonable epochs/imgsz) to avoid excessive training times.
- Checks if a suitable completed checkpoint already exists before training.
- Saves best weights under runs/vehicle/train/weights/best.pt and models/vehicle_detector.pt.
- Evaluates the best checkpoint on the validation set (Precision, Recall, mAP50, mAP50-95).
- Performs a quick inference test on sample traffic inputs and saves annotated output.
"""

from pathlib import Path
import shutil
import sys
import time
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "india_traffic_detection" / "data.yaml"
PRETRAINED_WEIGHTS = ROOT / "yolo11n.pt"

PROJECT_DIR = ROOT / "runs" / "vehicle"
RUN_NAME = "train"
TRAIN_RUN_DIR = PROJECT_DIR / RUN_NAME
BEST_WEIGHTS_PATH = TRAIN_RUN_DIR / "weights" / "best.pt"
MODELS_DIR = ROOT / "models"
DEST_MODEL_PATH = MODELS_DIR / "vehicle_detector.pt"


def main():
    print("=" * 78)
    print("YOLO11n VEHICLE MODEL TRAINING & EVALUATION")
    print("=" * 78)

    # 1. Validate dataset configuration & weights
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Repaired dataset config not found: {DATA_YAML}")
    if not PRETRAINED_WEIGHTS.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {PRETRAINED_WEIGHTS}")

    print(f"Dataset config   : {DATA_YAML}")
    print(f"Base weights     : {PRETRAINED_WEIGHTS}")

    # 2. Hardware selection
    has_cuda = torch.cuda.is_available()
    device = 0 if has_cuda else "cpu"
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    print(f"Compute device   : {device_name} (device={device})")

    # Hardware-adjusted hyperparams
    if has_cuda:
        epochs = 15
        batch_size = 16
        img_size = 640
    else:
        epochs = 5
        batch_size = 8
        img_size = 384

    print(f"Training config  : epochs={epochs}, batch={batch_size}, imgsz={img_size}")

    # 3. Check for existing checkpoint
    training_duration_sec = 0.0
    if BEST_WEIGHTS_PATH.exists():
        print(f"\n[INFO] Found existing completed checkpoint at: {BEST_WEIGHTS_PATH}")
        print("[INFO] Skipping retraining to preserve existing model as requested.")
        model = YOLO(str(BEST_WEIGHTS_PATH))
    else:
        print(f"\n[INFO] Starting training runs -> {TRAIN_RUN_DIR}")
        model = YOLO(str(PRETRAINED_WEIGHTS))
        start_time = time.time()

        model.train(
            data=str(DATA_YAML),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            workers=0,
            project=str(PROJECT_DIR),
            name=RUN_NAME,
            exist_ok=True,
            save=True,
            pretrained=True,
            val=True,
            plots=True
        )

        training_duration_sec = time.time() - start_time
        print(f"\n[SUCCESS] Training completed in {training_duration_sec:.1f} seconds ({training_duration_sec / 60:.2f} mins).")

    # Ensure best weights exist
    if not BEST_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Best weights not found after training: {BEST_WEIGHTS_PATH}")

    # Save a reference copy to models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BEST_WEIGHTS_PATH, DEST_MODEL_PATH)
    print(f"[INFO] Saved primary vehicle model checkpoint to: {DEST_MODEL_PATH}")

    # 4. Detailed Evaluation on Validation Set
    print("\n" + "=" * 78)
    print("VALIDATION EVALUATION (Best Checkpoint)")
    print("=" * 78)

    best_model = YOLO(str(BEST_WEIGHTS_PATH))
    metrics = best_model.val(
        data=str(DATA_YAML),
        split="val",
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=0
    )

    p = metrics.box.mp
    r = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    print("\n--- VALIDATION SUMMARY METRICS ---")
    print(f"Precision (P)   : {p:.4f} ({p * 100:.2f}%)")
    print(f"Recall (R)      : {r:.4f} ({r * 100:.2f}%)")
    print(f"mAP@50          : {map50:.4f} ({map50 * 100:.2f}%)")
    print(f"mAP@50-95       : {map50_95:.4f} ({map50_95 * 100:.2f}%)")
    if training_duration_sec > 0:
        print(f"Training Time   : {training_duration_sec:.1f}s ({training_duration_sec / 60:.2f}m)")
    print(f"Best Model Path : {BEST_WEIGHTS_PATH.resolve()}")

    print("\n--- PER-CLASS METRICS ---")
    class_names = metrics.names
    for i, cname in class_names.items():
        if i < len(metrics.box.p):
            cp = metrics.box.p[i]
            cr = metrics.box.r[i]
            cmap50 = metrics.box.ap50[i]
            cmap = metrics.box.ap[i]
            print(f"  Class {i} [{cname:<14}]: P={cp:.3f}, R={cr:.3f}, mAP50={cmap50:.3f}, mAP50-95={cmap:.3f}")

    # 5. Quick Inference Test on Sample Traffic Input
    print("\n" + "=" * 78)
    print("SAMPLE INFERENCE TEST")
    print("=" * 78)

    # Prefer a sample from the test split, or data/input
    test_img_dir = ROOT / "datasets" / "india_traffic_detection" / "test" / "images"
    test_images = list(test_img_dir.glob("*.jpg"))
    sample_img_path = test_images[0] if test_images else ROOT / "data" / "input" / "sample.jpg"

    print(f"Testing inference on: {sample_img_path.name}")
    results = best_model.predict(
        source=str(sample_img_path),
        conf=0.30,
        imgsz=img_size,
        device=device,
        save=True,
        project=str(PROJECT_DIR),
        name="sample_inference",
        exist_ok=True
    )

    pred = results[0]
    boxes = pred.boxes
    print(f"Detected {len(boxes)} vehicles in test frame:")
    for box in boxes:
        cls_id = int(box.cls[0].item())
        cls_name = pred.names[cls_id]
        conf = float(box.conf[0].item())
        xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]
        print(f"  - {cls_name} (ID {cls_id}): conf={conf:.2f}, bbox={xyxy}")

    saved_pred_path = PROJECT_DIR / "sample_inference" / sample_img_path.name
    print(f"\nInference visualization saved to: {saved_pred_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()

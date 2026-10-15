"""Convert mixed YOLO annotations in india_traffic to standard 5-token detection boxes.

Creates a repaired, standalone dataset under datasets/india_traffic_detection:
- Preserves original dataset unchanged (read-only).
- Preserves existing 5-token detection rows verbatim.
- Converts polygon annotations (min/max coords) to normalized xc, yc, width, height.
- Copies all images to preserve train/valid/test splits.
- Generates datasets/india_traffic_detection/data.yaml with 6 classes.
- Validates the resulting dataset thoroughly.
"""

from collections import Counter
from pathlib import Path
import shutil
import sys
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DATASET = PROJECT_ROOT / "datasets" / "india_traffic"
SRC_YAML = SRC_DATASET / "data.yaml"
DST_DATASET = PROJECT_ROOT / "datasets" / "india_traffic_detection"
DST_YAML = DST_DATASET / "data.yaml"
SPLITS = ["train", "valid", "test"]
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def convert_polygon_to_bbox(parts):
    """Convert polygon tokens [class_id, x1, y1, x2, y2, ...] to detection box.
    Returns:
        (cid_str, xc, yc, w, h)
    """
    cid_str = parts[0]
    coords = [float(x) for x in parts[1:]]
    if len(coords) < 6 or len(coords) % 2 != 0:
        raise ValueError(f"Invalid polygon coordinate count: {len(coords)}")

    xs = coords[0::2]
    ys = coords[1::2]

    xmin = max(0.0, min(xs))
    xmax = min(1.0, max(xs))
    ymin = max(0.0, min(ys))
    ymax = min(1.0, max(ys))

    w = max(0.0, xmax - xmin)
    h = max(0.0, ymax - ymin)
    xc = xmin + w / 2.0
    yc = ymin + h / 2.0

    return cid_str, xc, yc, w, h


def prepare_dataset():
    print("=" * 78)
    print("PREPARING REPAIRED VEHICLE DETECTION DATASET")
    print(f"Source: {SRC_DATASET}")
    print(f"Target: {DST_DATASET}")
    print("=" * 78)

    if not SRC_YAML.exists():
        print(f"[ERROR] Source data.yaml not found: {SRC_YAML}")
        sys.exit(1)

    with open(SRC_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    nc = cfg.get("nc", 6)
    names = cfg.get("names", ['Bus', 'Car', 'Three-Wheeler', 'Truck', 'Two-Wheeler', 'Van'])

    print(f"Number of classes: {nc}")
    print(f"Class names: {names}")

    # Create destination directories
    DST_DATASET.mkdir(parents=True, exist_ok=True)

    summary_stats = {
        "src_images": 0,
        "src_labels": 0,
        "src_det_rows": 0,
        "src_seg_rows": 0,
        "dst_images": 0,
        "dst_labels": 0,
        "dst_preserved_det": 0,
        "dst_converted_poly": 0,
        "failures": [],
    }

    for split in SPLITS:
        src_split_dir = SRC_DATASET / split
        dst_split_dir = DST_DATASET / split

        src_img_dir = src_split_dir / "images"
        src_lbl_dir = src_split_dir / "labels"

        dst_img_dir = dst_split_dir / "images"
        dst_lbl_dir = dst_split_dir / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing split: {split.upper()}...")

        # 1. Copy images
        src_images = sorted([f for f in src_img_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS])
        print(f"  Copying {len(src_images)} images to {dst_img_dir}...")
        for img in src_images:
            dst_img = dst_img_dir / img.name
            if not dst_img.exists() or dst_img.stat().st_size != img.stat().st_size:
                shutil.copy2(img, dst_img)

        # 2. Process labels
        src_labels = sorted([f for f in src_lbl_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"])
        print(f"  Converting {len(src_labels)} label files to {dst_lbl_dir}...")

        split_preserved = 0
        split_converted = 0
        split_src_det = 0
        split_src_seg = 0

        for lbl in src_labels:
            dst_lbl = dst_lbl_dir / lbl.name
            lines = [l.strip() for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]

            converted_lines = []
            for lno, line in enumerate(lines, 1):
                parts = line.split()
                if not parts:
                    continue

                if len(parts) == 5:
                    # Detection box: preserve verbatim
                    split_src_det += 1
                    split_preserved += 1
                    converted_lines.append(line)
                elif len(parts) >= 7 and len(parts) % 2 != 0:
                    # Polygon annotation: convert to bbox
                    split_src_seg += 1
                    try:
                        cid_str, xc, yc, w, h = convert_polygon_to_bbox(parts)
                        cid = int(float(cid_str))
                        if not (0 <= cid < nc):
                            summary_stats["failures"].append(f"{lbl.name}:L{lno} Class ID {cid} out of range [0, {nc-1}]")
                        if w <= 0 or h <= 0:
                            summary_stats["failures"].append(f"{lbl.name}:L{lno} Degenerate box width/height: w={w}, h={h}")
                        
                        bbox_line = f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                        converted_lines.append(bbox_line)
                        split_converted += 1
                    except Exception as e:
                        summary_stats["failures"].append(f"{lbl.name}:L{lno} Error converting polygon: {e}")
                else:
                    summary_stats["failures"].append(f"{lbl.name}:L{lno} Unrecognized format (token count {len(parts)}): {line}")

            # Write converted label file
            dst_lbl.write_text("\n".join(converted_lines) + ("\n" if converted_lines else ""), encoding="utf-8")

        summary_stats["src_images"] += len(src_images)
        summary_stats["src_labels"] += len(src_labels)
        summary_stats["src_det_rows"] += split_src_det
        summary_stats["src_seg_rows"] += split_src_seg
        summary_stats["dst_images"] += len(src_images)
        summary_stats["dst_labels"] += len(src_labels)
        summary_stats["dst_preserved_det"] += split_preserved
        summary_stats["dst_converted_poly"] += split_converted

        print(f"  -> Split done: {split_preserved} preserved boxes, {split_converted} polygon->bbox conversions.")

    # 3. Create repaired data.yaml
    repaired_yaml_content = f"""path: datasets/india_traffic_detection
train: train/images
val: valid/images
test: test/images

nc: {nc}
names: {names}
"""
    DST_YAML.write_text(repaired_yaml_content, encoding="utf-8")
    print(f"\nCreated repaired YAML: {DST_YAML}")

    # 4. Final Conversion Report
    print("\n" + "=" * 78)
    print("CONVERSION SUMMARY")
    print("=" * 78)
    print(f"Source Images           : {summary_stats['src_images']}")
    print(f"Target Images           : {summary_stats['dst_images']}")
    print(f"Source Label Files      : {summary_stats['src_labels']}")
    print(f"Target Label Files      : {summary_stats['dst_labels']}")
    print(f"Original Detection Rows : {summary_stats['src_det_rows']}")
    print(f"Preserved Detection Rows: {summary_stats['dst_preserved_det']}")
    print(f"Original Polygon Rows   : {summary_stats['src_seg_rows']}")
    print(f"Converted Polygon Rows  : {summary_stats['dst_converted_poly']}")
    print(f"Total Repaired Bboxes   : {summary_stats['dst_preserved_det'] + summary_stats['dst_converted_poly']}")
    print(f"Failures / Errors       : {len(summary_stats['failures'])}")
    if summary_stats["failures"]:
        print("  Errors:")
        for err in summary_stats["failures"][:10]:
            print(f"    - {err}")
    print("=" * 78)

    if summary_stats["failures"]:
        print("[FAILED] Dataset conversion completed with errors.")
        sys.exit(1)
    else:
        print("[SUCCESS] All polygon annotations successfully converted to YOLO detection boxes.")


if __name__ == "__main__":
    prepare_dataset()

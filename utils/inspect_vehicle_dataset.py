"""Inspect India Traffic vehicle dataset splits and labels (read-only).

Checks:
- Image dimensions & integrity
- Image/label counts & pairing (missing images or labels)
- Empty label files
- Invalid rows (malformed tokens, invalid numbers, coordinate format errors)
- Detection vs. Segmentation format breakdown and mixed-format files
- Class IDs outside the configured range in data.yaml
"""

from collections import Counter, defaultdict
from pathlib import Path
import sys
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML_PATH = PROJECT_ROOT / "datasets" / "india_traffic" / "data.yaml"
SPLITS = ["train", "valid", "test"]
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inspect_dataset(custom_yaml=None):
    target_yaml = Path(custom_yaml).resolve() if custom_yaml else DATA_YAML_PATH
    print("=" * 78)
    print(f"VEHICLE DATASET INSPECTION: {target_yaml.parent.name}")
    print("=" * 78)

    if not target_yaml.exists():
        print(f"[ERROR] data.yaml not found at: {target_yaml}")
        sys.exit(1)

    with open(target_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    nc = data_cfg.get("nc", 0)
    names = data_cfg.get("names", [])
    configured_path = data_cfg.get("path", target_yaml.parent.name)
    dataset_base = (PROJECT_ROOT / configured_path) if not Path(configured_path).is_absolute() else Path(configured_path)
    if not dataset_base.exists():
        dataset_base = target_yaml.parent

    print(f"Config path  : {target_yaml}")
    print(f"Dataset root : {dataset_base}")
    print(f"Classes (nc) : {nc}")
    print(f"Class names  : {names}")
    print("-" * 78)

    overall_stats = {
        "images": 0,
        "labels": 0,
        "missing_labels": 0,
        "missing_images": 0,
        "empty_labels": 0,
        "invalid_rows": 0,
        "oob_class_rows": 0,
        "detection_rows": 0,
        "segmentation_rows": 0,
        "mixed_files": 0,
        "det_only_files": 0,
        "seg_only_files": 0,
    }

    for split in SPLITS:
        split_dir = dataset_base / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        print(f"\n[{split.upper()} SPLIT] -> {split_dir}")

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"  [ERROR] Missing images or labels directory in {split_dir}")
            continue

        # Collect files
        image_files = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS]
        label_files = [f for f in lbl_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]

        img_stem_map = {f.stem: f for f in image_files}
        lbl_stem_map = {f.stem: f for f in label_files}

        img_stems = set(img_stem_map.keys())
        lbl_stems = set(lbl_stem_map.keys())

        # Check missing pairs
        missing_label_stems = sorted(img_stems - lbl_stems)
        missing_image_stems = sorted(lbl_stems - img_stems)

        # Check image dimensions
        dims_counter = Counter()
        corrupt_images = []
        for img_path in image_files:
            try:
                with Image.open(img_path) as im:
                    dims_counter[im.size] += 1
            except Exception as e:
                corrupt_images.append((img_path.name, str(e)))

        # Inspect labels
        empty_label_files = []
        invalid_rows = []
        oob_class_rows = []

        split_det_rows = 0
        split_seg_rows = 0
        split_mixed_files = []
        split_det_only_files = []
        split_seg_only_files = []
        class_counter = Counter()

        for lbl_path in label_files:
            try:
                raw_text = lbl_path.read_text(encoding="utf-8")
            except Exception as e:
                invalid_rows.append((lbl_path.name, 0, f"Cannot read file: {e}"))
                continue

            lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            if not lines:
                empty_label_files.append(lbl_path.name)
                continue

            file_has_det = False
            file_has_seg = False

            for line_no, line in enumerate(lines, start=1):
                tokens = line.split()
                if not tokens:
                    continue

                # Parse class ID
                try:
                    cid_float = float(tokens[0])
                    cid = int(cid_float)
                    if cid_float != cid:
                        raise ValueError("Non-integer class ID")
                except ValueError:
                    invalid_rows.append((lbl_path.name, line_no, f"Invalid class ID: '{tokens[0]}' in '{line}'"))
                    continue

                # Check coords
                try:
                    coords = [float(t) for t in tokens[1:]]
                except ValueError:
                    invalid_rows.append((lbl_path.name, line_no, f"Non-numeric coordinates in '{line}'"))
                    continue

                # Class range check
                if cid < 0 or cid >= nc:
                    oob_class_rows.append((lbl_path.name, line_no, cid, line))
                else:
                    class_counter[cid] += 1

                # Check format
                num_coords = len(coords)
                if num_coords == 4:
                    # Detection box: xc, yc, w, h
                    split_det_rows += 1
                    file_has_det = True
                elif num_coords >= 6 and num_coords % 2 == 0:
                    # Segmentation polygon: x1, y1, x2, y2, ...
                    split_seg_rows += 1
                    file_has_seg = True
                else:
                    invalid_rows.append(
                        (lbl_path.name, line_no, f"Unexpected coordinate count ({num_coords}) in '{line}'")
                    )

            if file_has_det and file_has_seg:
                split_mixed_files.append(lbl_path.name)
            elif file_has_det:
                split_det_only_files.append(lbl_path.name)
            elif file_has_seg:
                split_seg_only_files.append(lbl_path.name)

        # Update totals
        overall_stats["images"] += len(image_files)
        overall_stats["labels"] += len(label_files)
        overall_stats["missing_labels"] += len(missing_label_stems)
        overall_stats["missing_images"] += len(missing_image_stems)
        overall_stats["empty_labels"] += len(empty_label_files)
        overall_stats["invalid_rows"] += len(invalid_rows)
        overall_stats["oob_class_rows"] += len(oob_class_rows)
        overall_stats["detection_rows"] += split_det_rows
        overall_stats["segmentation_rows"] += split_seg_rows
        overall_stats["mixed_files"] += len(split_mixed_files)
        overall_stats["det_only_files"] += len(split_det_only_files)
        overall_stats["seg_only_files"] += len(split_seg_only_files)

        # Print Split Results
        print(f"  Images count        : {len(image_files)}")
        print(f"  Labels count        : {len(label_files)}")
        dim_str = ", ".join(f"{w}x{h} ({cnt})" for (w, h), cnt in sorted(dims_counter.items()))
        print(f"  Image dimensions    : {dim_str if dim_str else 'None'}")
        if corrupt_images:
            print(f"  [!] Corrupt images  : {len(corrupt_images)} (e.g. {corrupt_images[:3]})")

        # Missing pairs
        print(f"  Missing labels      : {len(missing_label_stems)}")
        if missing_label_stems:
            examples = [f"{s}.jpg" for s in missing_label_stems[:3]]
            print(f"      Example missing label images: {examples}")

        print(f"  Missing images      : {len(missing_image_stems)}")
        if missing_image_stems:
            examples = [f"{s}.txt" for s in missing_image_stems[:3]]
            print(f"      Example missing image labels: {examples}")

        # Empty labels
        print(f"  Empty label files   : {len(empty_label_files)}")
        if empty_label_files:
            print(f"      Example empty labels: {empty_label_files[:3]}")

        # Invalid rows
        print(f"  Invalid rows        : {len(invalid_rows)}")
        if invalid_rows:
            print("      Example invalid rows:")
            for item in invalid_rows[:3]:
                print(f"        - {item[0]}:L{item[1]} -> {item[2]}")

        # Class IDs
        print(f"  Class IDs out-of-range: {len(oob_class_rows)}")
        if oob_class_rows:
            print("      Example out-of-range rows:")
            for item in oob_class_rows[:3]:
                print(f"        - {item[0]}:L{item[1]} (Class ID {item[2]})")

        # Mixed format report
        print(f"  Format breakdown    : {split_det_rows} detection boxes, {split_seg_rows} segmentation polygons")
        print(f"  Label file formats  : {len(split_seg_only_files)} seg-only, {len(split_det_only_files)} det-only, {len(split_mixed_files)} mixed")
        if split_mixed_files:
            print(f"      Example files with mixed formats: {split_mixed_files[:3]}")

        # Class counts
        cls_summary = ", ".join(
            f"{cid} ({names[cid] if cid < len(names) else 'Unknown'}): {class_counter[cid]}"
            for cid in sorted(class_counter.keys())
        )
        print(f"  Class distribution  : {cls_summary if cls_summary else 'No valid annotations'}")

    print("\n" + "=" * 78)
    print("DATASET-WIDE SUMMARY")
    print("=" * 78)
    print(f"Total Images               : {overall_stats['images']}")
    print(f"Total Label Files          : {overall_stats['labels']}")
    print(f"Missing Image-Label Pairs  : {overall_stats['missing_labels'] + overall_stats['missing_images']} (missing labels: {overall_stats['missing_labels']}, missing images: {overall_stats['missing_images']})")
    print(f"Empty Label Files          : {overall_stats['empty_labels']}")
    print(f"Invalid Rows               : {overall_stats['invalid_rows']}")
    print(f"Out-of-range Class IDs     : {overall_stats['oob_class_rows']}")
    print(f"Total Annotations          : {overall_stats['detection_rows'] + overall_stats['segmentation_rows']} ({overall_stats['segmentation_rows']} segmentation, {overall_stats['detection_rows']} detection)")
    print(f"Files with Mixed Formats   : {overall_stats['mixed_files']} / {overall_stats['labels']} label files")
    print("=" * 78)


if __name__ == "__main__":
    yaml_arg = sys.argv[1] if len(sys.argv) > 1 else None
    inspect_dataset(yaml_arg)

import os
import glob
import yaml
import zipfile


def inspect_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_yolo_split(split_dir, num_classes):
    img_dir = os.path.join(split_dir, "images")
    lbl_dir = os.path.join(split_dir, "labels")

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    images = glob.glob(os.path.join(img_dir, "*.*")) if os.path.exists(img_dir) else []
    images = [img for img in images if img.lower().endswith(image_extensions)]

    labels = glob.glob(os.path.join(lbl_dir, "*.txt")) if os.path.exists(lbl_dir) else []

    img_stems = {os.path.splitext(os.path.basename(f))[0]: f for f in images}
    lbl_stems = {os.path.splitext(os.path.basename(f))[0]: f for f in labels}

    missing_labels = [stem for stem in img_stems if stem not in lbl_stems]
    missing_images = [stem for stem in lbl_stems if stem not in img_stems]

    empty_labels = 0
    invalid_boxes = 0
    class_distribution = {}

    for stem, lbl_file in lbl_stems.items():
        with open(lbl_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            empty_labels += 1
            continue

        for line_num, line in enumerate(lines, 1):
            tokens = line.split()
            # YOLO format requires 5 elements: class_id, x_center, y_center, width, height
            if len(tokens) != 5:
                invalid_boxes += 1
                continue

            try:
                cid = int(tokens[0])
                xc, yc, w, h = map(float, tokens[1:])

                if not (0 <= cid < num_classes):
                    invalid_boxes += 1
                    continue

                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    invalid_boxes += 1
                    continue

                class_distribution[cid] = class_distribution.get(cid, 0) + 1

            except ValueError:
                invalid_boxes += 1

    return {
        "images_count": len(images),
        "labels_count": len(labels),
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
        "empty_labels": empty_labels,
        "invalid_boxes": invalid_boxes,
        "class_distribution": class_distribution
    }


def validate_dataset(dataset_path, dataset_name):
    print("=" * 70)
    print(f"Dataset Validation: {dataset_name}")
    print(f"Path: {dataset_path}")
    print("=" * 70)

    yaml_path = os.path.join(dataset_path, "data.yaml")
    yaml_data = inspect_yaml(yaml_path)

    if not yaml_data:
        print(f"[ERROR] data.yaml not found at '{yaml_path}'")
        return False

    names = yaml_data.get("names", [])
    nc = yaml_data.get("nc", len(names))
    print(f"[INFO] Number of Classes (nc): {nc}")
    print(f"[INFO] Class Names: {names}")

    roboflow_info = yaml_data.get("roboflow", {})
    if roboflow_info:
        print(f"[INFO] Roboflow Source: {roboflow_info.get('project')} (v{roboflow_info.get('version')})")

    total_images = 0
    total_labels = 0
    total_invalid = 0
    total_empty = 0

    print("\n--- Split Verification ---")
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(dataset_path, split)
        if not os.path.exists(split_dir):
            print(f"  [{split.upper():<5}] Not found!")
            continue

        stats = validate_yolo_split(split_dir, nc)
        total_images += stats["images_count"]
        total_labels += stats["labels_count"]
        total_invalid += stats["invalid_boxes"]
        total_empty += stats["empty_labels"]

        print(f"  [{split.upper():<5}] Images: {stats['images_count']:<4} | Labels: {stats['labels_count']:<4} | "
              f"Missing Lbl: {stats['missing_labels']} | Invalid Box: {stats['invalid_boxes']} | Empty: {stats['empty_labels']}")

    print("\n--- Summary ---")
    print(f"  Total Images: {total_images}")
    print(f"  Total Labels: {total_labels}")
    print(f"  Invalid Box Lines: {total_invalid}")
    print(f"  Background / Empty Labels: {total_empty}")

    return True


def check_downloads_zip():
    downloads_zip = r"C:\Users\ASUS\Downloads\India_Traffic_Analysis_v2.v2i.yolov11.zip"
    if os.path.exists(downloads_zip):
        print("\n" + "=" * 70)
        print("Inspection of Archive in Downloads: India_Traffic_Analysis_v2.v2i.yolov11.zip")
        print("=" * 70)
        with zipfile.ZipFile(downloads_zip) as z:
            imgs = [f for f in z.namelist() if f.endswith(('.jpg', '.jpeg', '.png'))]
            lbls = [f for f in z.namelist() if f.endswith('.txt') and 'README' not in f]
            print(f"  Archive Size     : {os.path.getsize(downloads_zip):,} bytes")
            print(f"  Total Images     : {len(imgs)}")
            print(f"  Total Labels     : {len(lbls)}")
            for split in ['train', 'valid', 'test']:
                s_imgs = [f for f in imgs if f.startswith(f'{split}/')]
                s_lbls = [f for f in lbls if f.startswith(f'{split}/')]
                print(f"  [{split.upper():<5}] Images: {len(s_imgs):<4} | Labels: {len(s_lbls):<4}")

            if "data.yaml" in z.namelist():
                content = z.read("data.yaml").decode("utf-8")
                print("\n  Archive data.yaml:\n" + "  " + content.replace("\n", "\n  "))


def main():
    ds_base = "datasets"
    ds_helmet = os.path.join(ds_base, "motorcycle_helmet")
    ds_traffic = os.path.join(ds_base, "india_traffic")

    validate_dataset(ds_helmet, "Motorcycle and Helmet")
    validate_dataset(ds_traffic, "India Traffic (Current Directory Content)")
    check_downloads_zip()


if __name__ == "__main__":
    main()

import os
import sys
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ocr.plate_detector import PlateDetector


def main():
    # Support both .jpg and .jpeg extensions seamlessly
    candidates = [
        os.path.join("data", "input", "plate_test.jpg"),
        os.path.join("data", "input", "plate_test.jpeg")
    ]
    input_image_path = None
    for p in candidates:
        if os.path.exists(p):
            input_image_path = p
            break

    output_annotated_path = os.path.join("data", "output", "plate_detection_clear_test.jpeg")
    output_crops_dir = os.path.join("data", "output", "crops")
    model_path = os.path.join("models", "plate_detector.pt")

    print("=" * 60)
    print("Test Plate Detection on Clear Image")
    print("=" * 60)
    print(f"[INFO] Input image path : {input_image_path}")
    print(f"[INFO] Output annotated : {output_annotated_path}")
    print(f"[INFO] Model path       : {model_path}")

    # 1. Validate input file
    if input_image_path is None or not os.path.exists(input_image_path):
        print(f"[ERROR] Could not find test image at 'data/input/plate_test.jpg' or 'data/input/plate_test.jpeg'")
        sys.exit(1)

    # 2. Validate model file
    if not os.path.exists(model_path):
        print(f"[ERROR] License plate model not found at: '{model_path}'")
        sys.exit(1)

    # 3. Read image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"[ERROR] Failed to load image from '{input_image_path}'")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"[INFO] Image loaded successfully: {w}x{h} (Width x Height)")

    # 4. Initialize PlateDetector
    detector = PlateDetector(model_path=model_path, conf_threshold=0.25)
    if not detector.load_model():
        print("[ERROR] Failed to load plate detector model.")
        sys.exit(1)

    # 5. Detect plates
    detections = detector.detect_plates(image)
    num_plates = len(detections)

    print("\n" + "-" * 40)
    print(f"[RESULT] Detected Plates Count: {num_plates}")
    print("-" * 40)

    if num_plates == 0:
        print("[INFO] No license plates detected above confidence threshold.")
    else:
        for idx, det in enumerate(detections, 1):
            conf = det["conf"]
            box = det["box"]
            print(f"  Plate #{idx}: Confidence = {conf:.4f} ({conf * 100:.2f}%) | Bounding Box = {box}")

    # 6. Extract and save plate crops if any found
    crops = detector.crop_plates(image, detections)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs(output_crops_dir, exist_ok=True)

    saved_crop_paths = []
    for idx, crop in enumerate(crops, 1):
        if crop is not None and crop.size > 0:
            # Save in both data/output/plate_crop_{idx}.jpeg and data/output/crops/plate_crop_{idx}.jpeg
            crop_path_primary = os.path.join("data", "output", f"plate_crop_{idx}.jpeg")
            crop_path_folder = os.path.join(output_crops_dir, f"plate_crop_{idx}.jpeg")
            cv2.imwrite(crop_path_primary, crop)
            cv2.imwrite(crop_path_folder, crop)
            saved_crop_paths.append(crop_path_primary)
            print(f"[INFO] Saved plate crop #{idx} to: '{crop_path_primary}' ({crop.shape[1]}x{crop.shape[0]} px)")

    # 7. Save annotated result
    annotated = detector.draw_detections(image, detections)
    success = cv2.imwrite(output_annotated_path, annotated)
    if success:
        print(f"[SUCCESS] Annotated result image saved to: '{output_annotated_path}'")
    else:
        print(f"[ERROR] Failed to save annotated image to '{output_annotated_path}'")
        sys.exit(1)

    print("=" * 60)
    print(f"[SUMMARY] Found {num_plates} plate(s), saved {len(saved_crop_paths)} crop(s).")
    print("=" * 60)


if __name__ == "__main__":
    main()

import os
import sys
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ocr.plate_detector import PlateDetector


def test_plate_detector(image_path: str = None):
    input_image_path = image_path or (sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "input", "sample.jpg"))
    output_image_path = os.path.join("data", "output", "plate_detection_test.jpg")
    model_path = os.path.join("models", "plate_detector.pt")

    print("=" * 60)
    print("Standalone Number Plate Detection Test")
    print("=" * 60)
    print(f"[INFO] Input image : {input_image_path}")
    print(f"[INFO] Output image: {output_image_path}")
    print(f"[INFO] Model path  : {model_path}")

    # 1. Check input image exists
    if not os.path.exists(input_image_path):
        print(f"[ERROR] Input image not found at: '{input_image_path}'")
        print(f"        Absolute path: '{os.path.abspath(input_image_path)}'")
        sys.exit(1)

    # 2. Check model file exists
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: '{model_path}'")
        print(f"        Absolute path: '{os.path.abspath(model_path)}'")
        sys.exit(1)

    # 3. Load image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"[ERROR] Failed to read image from '{input_image_path}'. Corrupt or invalid format.")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"[INFO] Loaded image successfully: {w}x{h} (Width x Height)")

    # 4. Initialize PlateDetector
    detector = PlateDetector(model_path=model_path, conf_threshold=0.25)
    if not detector.load_model():
        print("[ERROR] Failed to initialize plate detector model.")
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
            box = det["box"]
            conf = det["conf"]
            print(f"  Plate #{idx}: Confidence = {conf:.4f} | Box = {box}")

    # 6. Test cropping function
    crops = detector.crop_plates(image, detections)
    valid_crops = [c for c in crops if c is not None]
    print(f"[INFO] Successfully extracted {len(valid_crops)} plate crop(s).")

    # 7. Draw annotations and save output image
    annotated_image = detector.draw_detections(image, detections)

    output_dir = os.path.dirname(output_image_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    success = cv2.imwrite(output_image_path, annotated_image)
    if success:
        print(f"[SUCCESS] Annotated test image saved to: '{output_image_path}'")
    else:
        print(f"[ERROR] Failed to save annotated image to '{output_image_path}'.")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    test_plate_detector()

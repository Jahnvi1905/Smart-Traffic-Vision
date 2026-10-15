import os
import sys
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ocr.plate_ocr import PlateOCR


def main():
    crop_path = os.path.join("data", "output", "plate_crop_1.jpeg")
    output_dir = os.path.join("data", "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 10B — Number Plate OCR Preprocessing Comparison Experiment")
    print("=" * 70)
    print(f"[INFO] Source Crop Image Path: {crop_path}")

    # 1. Gracefully check for crop file existence
    if not os.path.exists(crop_path):
        print(f"[ERROR] Crop file not found at: '{crop_path}'")
        print(f"        Absolute path checked: '{os.path.abspath(crop_path)}'")
        print("        Please run 'utils/test_plate_clear.py' first to generate the plate crop.")
        sys.exit(1)

    # 2. Read original image
    original = cv2.imread(crop_path)
    if original is None:
        print(f"[ERROR] Failed to load crop image from '{crop_path}'")
        sys.exit(1)

    h, w = original.shape[:2]
    print(f"[INFO] Original Crop Dimensions: {w}x{h} px\n")

    # 3. Generate Preprocessed Variants
    # A. Original
    img_original = original.copy()
    path_original = os.path.join(output_dir, "plate_crop_original.jpeg")
    cv2.imwrite(path_original, img_original)

    # B. Enlarged using OpenCV INTER_CUBIC interpolation (2x scale)
    img_enlarged = cv2.resize(original, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    path_enlarged = os.path.join(output_dir, "plate_crop_enlarged.jpeg")
    cv2.imwrite(path_enlarged, img_enlarged)

    # C. Grayscale
    img_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    path_gray = os.path.join(output_dir, "plate_crop_gray.jpeg")
    cv2.imwrite(path_gray, img_gray)

    # D. Adaptive Threshold
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    path_thresh = os.path.join(output_dir, "plate_crop_thresh.jpeg")
    cv2.imwrite(path_thresh, img_thresh)

    methods = [
        ("Original (Raw BGR)", img_original, path_original),
        ("Enlarged (2x Bicubic)", img_enlarged, path_enlarged),
        ("Grayscale", img_gray, path_gray),
        ("Adaptive Threshold", img_thresh, path_thresh),
    ]

    # 4. Initialize OCR engine once
    print("[INFO] Initializing EasyOCR Engine...")
    ocr = PlateOCR(languages=['en'], gpu=False)
    if not ocr.load_model():
        print("[ERROR] Failed to load EasyOCR model.")
        sys.exit(1)

    # 5. Execute OCR on each method and compare
    results = []
    print("\n" + "-" * 70)
    print(f"{'Preprocessing Method':<24} | {'Recognized Text':<18} | {'Confidence':<12} | Status")
    print("-" * 70)

    for name, img_variant, save_path in methods:
        text, conf = ocr.read_plate(img_variant)
        results.append({
            "name": name,
            "text": text if text else "(None)",
            "conf": conf,
            "path": save_path
        })
        status = "Success" if text else "Failed"
        conf_str = f"{conf:.4f} ({conf*100:.2f}%)" if conf > 0 else "N/A"
        print(f"{name:<24} | {text:<18} | {conf_str:<12} | {status}")

    print("-" * 70)

    # 6. Summary and Analysis
    print("\nVisual Inspection Artifacts Saved:")
    for res in results:
        print(f"  - {res['name']:<22} -> '{res['path']}'")

    best_result = max(results, key=lambda x: x["conf"])
    print("\nEmpirical Findings:")
    print(f"  * Best Performing Method : {best_result['name']}")
    print(f"  * Highest Confidence Score: {best_result['conf']:.4f} ({best_result['conf']*100:.2f}%) with text '{best_result['text']}'")
    print("=" * 70)


if __name__ == "__main__":
    main()

import os
import sys
import cv2
import numpy as np
import easyocr


class PlateOCR:
    """
    License plate OCR engine utilizing EasyOCR to extract alphanumeric plate text
    and confidence scores from cropped plate images.
    """

    def __init__(self, languages: list = None, gpu: bool = False):
        """
        Initialize the OCR engine.

        Args:
            languages: List of language codes for EasyOCR (default: ['en']).
            gpu: Whether to use GPU acceleration (default: False for compatibility).
        """
        self.languages = languages or ['en']
        self.gpu = gpu
        self.reader = None

    def load_model(self) -> bool:
        """Loads the EasyOCR reader model."""
        try:
            if self.reader is None:
                # verbose=False avoids Windows charmap encoding errors during model loading/reporting
                self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to initialize EasyOCR reader: {e}")
            return False

    def read_plate(self, plate_crop) -> tuple:
        """
        Recognizes text from a cropped license plate image.

        Args:
            plate_crop: BGR or grayscale numpy image array, or filepath string.

        Returns:
            tuple: (recognized_text: str, confidence: float)
                   Returns ("", 0.0) if no text is recognized or on error.
        """
        # 1. Handle file path string input
        if isinstance(plate_crop, str):
            if not os.path.exists(plate_crop):
                print(f"[ERROR] Crop file does not exist at: '{plate_crop}'")
                return "", 0.0
            image = cv2.imread(plate_crop)
            if image is None:
                print(f"[ERROR] Failed to read crop image from: '{plate_crop}'")
                return "", 0.0
        elif isinstance(plate_crop, np.ndarray):
            if plate_crop.size == 0:
                print("[WARN] Received empty crop image array.")
                return "", 0.0
            image = plate_crop
        else:
            print(f"[ERROR] Unsupported input type for plate_crop: {type(plate_crop)}")
            return "", 0.0

        # 2. Ensure model is loaded
        if self.reader is None and not self.load_model():
            return "", 0.0

        try:
            # Perform OCR inference
            ocr_results = self.reader.readtext(image)

            if not ocr_results:
                return "", 0.0

            # Collect text parts and calculate confidence
            text_parts = []
            confidences = []

            for bbox, text, conf in ocr_results:
                clean_text = text.strip()
                if clean_text:
                    text_parts.append(clean_text)
                    confidences.append(float(conf))

            if not text_parts:
                return "", 0.0

            combined_text = " ".join(text_parts)
            avg_confidence = float(sum(confidences) / len(confidences))

            return combined_text, avg_confidence

        except Exception as e:
            print(f"[ERROR] An error occurred during plate OCR recognition: {e}")
            return "", 0.0


# Module-level convenience function
_DEFAULT_OCR_INSTANCE = None

def recognize_plate(plate_crop, gpu: bool = False) -> tuple:
    """
    Convenience function to recognize text and confidence from a plate crop.
    """
    global _DEFAULT_OCR_INSTANCE
    if _DEFAULT_OCR_INSTANCE is None:
        _DEFAULT_OCR_INSTANCE = PlateOCR(gpu=gpu)
    return _DEFAULT_OCR_INSTANCE.read_plate(plate_crop)

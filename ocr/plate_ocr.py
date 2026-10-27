import os
import re
import sys
from typing import Dict, Tuple, Optional, Any
import cv2
import numpy as np
import easyocr


# Conservative boundary/bracket punctuation artifacts commonly produced by edge contours, borders, or screws
STRIP_PUNCTUATION = " \t\n\r[](){}<>|~_`'\";:.,*+=!?-/\\"


def normalize_plate_text(raw_text: str) -> str:
    """
    Conservatively normalizes raw OCR plate text by:
    - Stripping leading and trailing bracket, border, or punctuation artifacts.
    - Stripping trailing bracket/edge noise on individual tokens (e.g. '7602]' -> '7602').
    - Normalizing whitespace and converting to uppercase.

    Conservative principles:
    - Does NOT fabricate, guess, or substitute ambiguous characters (e.g. no arbitrary '0' vs 'O' guessing).
    - Preserves true alphanumeric content while stripping camera/edge boundary artifacts.
    """
    if not raw_text:
        return ""

    text = str(raw_text).strip()
    # Normalize delimiter characters (hyphens, dots, underscores) between plate blocks
    text = re.sub(r"[-._]", " ", text)
    text = text.strip(STRIP_PUNCTUATION)

    # Process tokens to clean boundary noise while preserving characters
    tokens = text.split()
    cleaned_tokens = []
    for token in tokens:
        clean_tok = token.strip(STRIP_PUNCTUATION)
        if clean_tok:
            cleaned_tokens.append(clean_tok.upper())

    return " ".join(cleaned_tokens)


def validate_plate_output(
    raw_text: str,
    confidence: float,
    min_confidence_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Performs conservative validation of OCR output:
    - Normalizes text conservatively (stripping boundary brackets and noise).
    - Preserves raw OCR output for audit and debugging.
    - Evaluates confidence against minimum threshold (default: 0.50).
    - Classifies confidence level ('high', 'medium', 'low', 'unreadable') without silent guessing.

    Args:
        raw_text: Raw string emitted by OCR reader.
        confidence: Average confidence score from OCR reader.
        min_confidence_threshold: Threshold below which a result is flagged as low confidence.

    Returns:
        dict containing:
            - raw_text: unmodified OCR string
            - normalized_text: conservatively normalized alphanumeric string
            - confidence: rounded float
            - is_low_confidence: bool flag
            - confidence_status: 'high' | 'medium' | 'low' | 'unreadable'
            - is_valid_format: bool
    """
    normalized = normalize_plate_text(raw_text)
    conf = float(confidence) if confidence is not None else 0.0

    if not normalized or conf <= 0.0:
        return {
            "raw_text": raw_text or "",
            "normalized_text": "",
            "confidence": 0.0,
            "is_low_confidence": True,
            "confidence_status": "unreadable",
            "is_valid_format": False
        }

    # Check general plate alphanumeric pattern
    compact = re.sub(r"[^A-Za-z0-9]", "", normalized)
    indian_pattern = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")
    is_standard_indian = bool(indian_pattern.match(compact))
    is_valid_length = 4 <= len(compact) <= 14
    is_valid_format = is_standard_indian or is_valid_length

    # Confidence classification
    if conf >= 0.75:
        confidence_status = "high"
        is_low_conf = False
    elif conf >= min_confidence_threshold:
        confidence_status = "medium"
        is_low_conf = False
    else:
        confidence_status = "low"
        is_low_conf = True

    # If length is suspiciously short (< 4 alphanumeric characters), mark as low confidence
    if len(compact) < 4:
        is_low_conf = True
        if confidence_status != "unreadable":
            confidence_status = "low"

    return {
        "raw_text": raw_text,
        "normalized_text": normalized,
        "confidence": round(conf, 3),
        "is_low_confidence": is_low_conf,
        "confidence_status": confidence_status,
        "is_valid_format": is_valid_format
    }


class PlateOCR:
    """
    License plate OCR engine utilizing EasyOCR to extract alphanumeric plate text
    and confidence scores from cropped plate images.
    Features conservative normalization, raw output preservation, and confidence auditing.
    """

    def __init__(self, languages: list = None, gpu: bool = False, min_confidence: float = 0.50):
        """
        Initialize the OCR engine.

        Args:
            languages: List of language codes for EasyOCR (default: ['en']).
            gpu: Whether to use GPU acceleration (default: False for compatibility).
            min_confidence: Threshold below which plate readings are flagged as low confidence.
        """
        self.languages = languages or ['en']
        self.gpu = gpu
        self.min_confidence = min_confidence
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

    def read_plate_detailed(self, plate_crop) -> Dict[str, Any]:
        """
        Recognizes text from a cropped license plate image and returns full validation details
        including raw OCR output, normalized text, and confidence status.

        Args:
            plate_crop: BGR or grayscale numpy image array, or filepath string.

        Returns:
            Dict containing raw_text, normalized_text, confidence, is_low_confidence,
            confidence_status, and is_valid_format.
        """
        # 1. Handle file path string input
        if isinstance(plate_crop, str):
            if not os.path.exists(plate_crop):
                print(f"[ERROR] Crop file does not exist at: '{plate_crop}'")
                return validate_plate_output("", 0.0, self.min_confidence)
            image = cv2.imread(plate_crop)
            if image is None:
                print(f"[ERROR] Failed to read crop image from: '{plate_crop}'")
                return validate_plate_output("", 0.0, self.min_confidence)
        elif isinstance(plate_crop, np.ndarray):
            if plate_crop.size == 0:
                print("[WARN] Received empty crop image array.")
                return validate_plate_output("", 0.0, self.min_confidence)
            image = plate_crop
        else:
            print(f"[ERROR] Unsupported input type for plate_crop: {type(plate_crop)}")
            return validate_plate_output("", 0.0, self.min_confidence)

        # 2. Ensure model is loaded
        if self.reader is None and not self.load_model():
            return validate_plate_output("", 0.0, self.min_confidence)

        try:
            ocr_results = self.reader.readtext(image)

            if not ocr_results:
                return validate_plate_output("", 0.0, self.min_confidence)

            text_parts = []
            confidences = []

            for bbox, text, conf in ocr_results:
                clean_t = text.strip()
                if clean_t:
                    text_parts.append(clean_t)
                    confidences.append(float(conf))

            if not text_parts:
                return validate_plate_output("", 0.0, self.min_confidence)

            raw_combined = " ".join(text_parts)
            avg_confidence = float(sum(confidences) / len(confidences))

            return validate_plate_output(raw_combined, avg_confidence, self.min_confidence)

        except Exception as e:
            print(f"[ERROR] An error occurred during plate OCR recognition: {e}")
            return validate_plate_output("", 0.0, self.min_confidence)

    def read_plate(self, plate_crop) -> Tuple[str, float]:
        """
        Backward-compatible plate reader returning (normalized_text, confidence).
        Preserves existing caller contracts while applying conservative normalization.

        Args:
            plate_crop: BGR or grayscale numpy image array, or filepath string.

        Returns:
            tuple: (normalized_text: str, confidence: float)
        """
        details = self.read_plate_detailed(plate_crop)
        return details["normalized_text"], details["confidence"]


# Module-level convenience function
_DEFAULT_OCR_INSTANCE = None

def recognize_plate(plate_crop, gpu: bool = False) -> Tuple[str, float]:
    """
    Convenience function to recognize text and confidence from a plate crop.
    """
    global _DEFAULT_OCR_INSTANCE
    if _DEFAULT_OCR_INSTANCE is None:
        _DEFAULT_OCR_INSTANCE = PlateOCR(gpu=gpu)
    return _DEFAULT_OCR_INSTANCE.read_plate(plate_crop)

import os
import cv2
import numpy as np
from ultralytics import YOLO


class PlateDetector:
    """
    License plate detector utilizing YOLO weights fine-tuned for plate localization.
    """

    def __init__(self, model_path: str = "models/plate_detector.pt", conf_threshold: float = 0.25):
        """
        Initialize PlateDetector.

        Args:
            model_path: Path to the trained/fine-tuned license plate YOLO model (.pt).
            conf_threshold: Minimum detection confidence threshold.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None

    def load_model(self) -> bool:
        """
        Loads the YOLO license plate detection model.
        Returns True if successful, False otherwise.
        """
        if not os.path.exists(self.model_path):
            print(f"[ERROR] License plate model file not found at: '{self.model_path}'")
            print(f"        Absolute path checked: '{os.path.abspath(self.model_path)}'")
            return False

        try:
            print(f"[INFO] Loading license plate model from: '{self.model_path}'...")
            self.model = YOLO(self.model_path)
            print("[INFO] License plate model loaded successfully.")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO plate model from '{self.model_path}': {e}")
            self.model = None
            return False

    def detect_plates(self, image: np.ndarray) -> list:
        """
        Detects license plates in the provided BGR image.

        Args:
            image: OpenCV BGR image (np.ndarray).

        Returns:
            list: List of detection dictionaries:
                  [{"box": [x1, y1, x2, y2], "conf": float, "class_name": str}, ...]
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            print("[WARN] Invalid or empty image provided to detect_plates.")
            return []

        if self.model is None and not self.load_model():
            return []

        detections = []
        try:
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            if not results or len(results) == 0:
                return []

            boxes = results[0].boxes
            for box in boxes:
                conf = float(box.conf[0].item())
                if conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                class_name = self.model.names.get(cls_id, "license_plate")

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "conf": conf,
                    "class_name": class_name
                })

        except Exception as e:
            print(f"[ERROR] Error during license plate detection: {e}")

        return detections

    def crop_plates(self, image: np.ndarray, detections: list) -> list:
        """
        Crops detected license plate regions from the original image.

        Args:
            image: Source BGR image (np.ndarray).
            detections: List of detection dictionaries from detect_plates.

        Returns:
            list: List of cropped BGR image arrays.
        """
        if image is None or not detections:
            return []

        h, w = image.shape[:2]
        crops = []

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            # Ensure coordinates stay within image boundaries
            x1_clamped = max(0, min(x1, w - 1))
            y1_clamped = max(0, min(y1, h - 1))
            x2_clamped = max(0, min(x2, w))
            y2_clamped = max(0, min(y2, h))

            if x2_clamped > x1_clamped and y2_clamped > y1_clamped:
                crop = image[y1_clamped:y2_clamped, x1_clamped:x2_clamped].copy()
                crops.append(crop)
            else:
                crops.append(None)

        return crops

    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """
        Draws bounding boxes and confidence score labels on a copy of the image.

        Args:
            image: Source BGR image.
            detections: List of detection dictionaries.

        Returns:
            np.ndarray: Annotated copy of the image.
        """
        if image is None:
            return None

        annotated = image.copy()
        color = (0, 255, 255)  # Yellow for license plates

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            conf = det["conf"]
            label = f"Plate {conf:.2f}"

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Label tag
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            tag_y1 = max(0, y1 - th - 10)
            tag_y2 = y1
            tag_x1 = x1
            tag_x2 = min(annotated.shape[1], x1 + tw + 10)

            cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), color, cv2.FILLED)
            cv2.putText(
                annotated,
                label,
                (tag_x1 + 5, tag_y2 - 5),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
                lineType=cv2.LINE_AA
            )

        return annotated

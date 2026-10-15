import os
import cv2
from ultralytics import YOLO

# Target COCO classes for vehicle detection
TARGET_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Extended BGR colors for visualization
CLASS_COLORS = {
    "car": (0, 255, 0),           # Vibrant Green
    "motorcycle": (255, 200, 0),  # Amber / Cyan-ish
    "two-wheeler": (255, 200, 0), # Amber
    "three-wheeler": (0, 215, 255),# Gold / Yellow-Orange
    "bus": (0, 165, 255),         # Vibrant Orange
    "truck": (200, 50, 255),      # Magenta / Violet
    "van": (255, 105, 180)        # Hot Pink / Rose
}


class VehicleDetector:
    """
    Vehicle detector utilizing Ultralytics YOLO for identifying
    cars, motorcycles/two-wheelers, three-wheelers, buses, trucks, and vans.
    """

    def __init__(self, model_path: str = "models/vehicle_detector.pt", conf_threshold: float = 0.35):
        """
        Initialize the vehicle detector.

        Args:
            model_path: Path to the YOLO weights or model identifier.
            conf_threshold: Minimum detection confidence threshold.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.target_classes = TARGET_CLASSES
        self.target_class_ids = list(TARGET_CLASSES.keys())
        self.model = None

    def load_model(self) -> bool:
        """Loads the YOLO model and dynamically inspects class names."""
        try:
            print(f"[INFO] Loading YOLO model from '{self.model_path}'...")
            self.model = YOLO(self.model_path)
            
            # If model has custom classes (e.g. 6-class india_traffic model)
            if hasattr(self.model, "names") and len(self.model.names) <= 15:
                self.target_classes = {int(k): str(v).lower() for k, v in self.model.names.items()}
                self.target_class_ids = list(self.target_classes.keys())
            else:
                self.target_classes = TARGET_CLASSES
                self.target_class_ids = list(TARGET_CLASSES.keys())

            print(f"[INFO] Model loaded successfully. Active classes: {self.target_classes}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model: {e}")
            return False

    def get_color(self, class_name: str) -> tuple:
        return CLASS_COLORS.get(class_name.lower(), (0, 255, 0))

    def draw_detections(self, frame, boxes):
        """
        Draws stylish bounding boxes and class labels with tracking ID and confidence scores on the frame.
        """
        for box in boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in self.target_classes:
                continue

            conf = float(box.conf[0].item())
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_name = self.target_classes[cls_id]
            color = self.get_color(class_name)

            # Include tracking ID if available
            track_id = int(box.id[0].item()) if box.id is not None else None
            if track_id is not None:
                label = f"ID:{track_id} {class_name.title()} {conf:.2f}"
            else:
                label = f"{class_name.title()} {conf:.2f}"

            # Draw bounding box and centroid
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 4, color, cv2.FILLED)

            # Draw label tag with background box for high legibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            tag_y1 = max(0, y1 - text_h - 8)
            tag_y2 = y1
            tag_x1 = x1
            tag_x2 = min(frame.shape[1], x1 + text_w + 8)

            # Draw background tag
            cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, cv2.FILLED)
            text_color = (0, 0, 0)
            cv2.putText(
                frame,
                label,
                (tag_x1 + 4, tag_y2 - 4),
                font,
                font_scale,
                text_color,
                font_thickness,
                lineType=cv2.LINE_AA
            )

        return frame

    def process_video(
        self,
        input_path: str = "data/input/sample.mp4",
        output_path: str = "data/output/vehicle_counting.mp4",
        enable_tracking: bool = True,
        tracker: str = "bytetrack.yaml",
        counter = None,
        violation_manager = None,
        max_frames: int = None,
        frame_skip: int = 1,
        progress_callback = None
    ) -> bool:
        """
        Runs vehicle detection and tracking on an input video with CPU-safe controls.

        Args:
            input_path: Path to the input video file.
            output_path: Path to the destination annotated video file.
            enable_tracking: Whether to enable multi-object tracking (ByteTrack).
            tracker: Tracker configuration file or name (e.g. 'bytetrack.yaml').
            counter: Optional VehicleCounter instance for virtual line counting.
            max_frames: Optional upper limit on processed frames (safe for CPU).
            frame_skip: Process every Nth frame (1 = every frame).
            progress_callback: Optional callable(frame_idx, total_frames, current_frame, counter).

        Returns:
            bool: True if processing succeeded, False otherwise.
        """
        mode_str = f"Detection + ByteTrack Tracking ({tracker})" if enable_tracking else "Detection Only"
        if counter is not None:
            mode_str += " + Vehicle Counting"
        print("=" * 60)
        print(f"[INFO] Starting vehicle processing pipeline: {mode_str}")
        print(f"[INFO] Input video : {input_path}")
        print(f"[INFO] Output video: {output_path}")
        print("=" * 60)

        if not os.path.exists(input_path):
            print(f"[ERROR] Input video not found at: '{input_path}'")
            return False

        if self.model is None and not self.load_model():
            return False

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"[ERROR] Unable to open video stream from '{input_path}'.")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        limit_frames = min(total_video_frames, max_frames) if max_frames else total_video_frames
        print(f"[INFO] Resolution: {width}x{height} | FPS: {fps:.2f} | Frames to process: {limit_frames}")

        # Automatically adjust counting line to the video's actual resolution
        if counter is not None:
            counter.adjust_line(width, height)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Ensure width and height are even for H.264 / MP4 compatibility
        width_even = width - (width % 2)
        height_even = height - (height % 2)
        effective_fps = max(5.0, float(fps) / max(1, frame_skip))

        writer = None
        for codec_tag in ['avc1', 'H264', 'mp4v']:
            fourcc = cv2.VideoWriter_fourcc(*codec_tag)
            writer = cv2.VideoWriter(output_path, fourcc, effective_fps, (width_even, height_even))
            if writer.isOpened():
                print(f"[INFO] Initialized VideoWriter using codec '{codec_tag}'")
                break

        if writer is None or not writer.isOpened():
            print(f"[ERROR] Could not open video writer for destination: '{output_path}'")
            cap.release()
            return False

        frame_idx = 0
        written_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                if frame_skip > 1 and (frame_idx % frame_skip != 0):
                    continue

                # Run inference / tracking
                if enable_tracking:
                    results = self.model.track(
                        frame,
                        persist=True,
                        tracker=tracker,
                        classes=self.target_class_ids,
                        conf=self.conf_threshold,
                        verbose=False
                    )
                else:
                    results = self.model(
                        frame,
                        classes=self.target_class_ids,
                        conf=self.conf_threshold,
                        verbose=False
                    )

                # Annotate detections and update counter (works with ByteTrack or fallback tracking)
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if counter is not None:
                        counter.update(boxes)
                    if violation_manager is not None:
                        violation_manager.process_frame(frame, boxes, self.target_classes, frame_idx=frame_idx)
                    frame = self.draw_detections(frame, boxes)

                # Draw counter HUD and line if active
                if counter is not None:
                    frame = counter.draw(frame)

                # Ensure dimensions match VideoWriter
                if frame.shape[1] != width_even or frame.shape[0] != height_even:
                    frame = cv2.resize(frame, (width_even, height_even))

                writer.write(frame)
                written_count += 1

                if progress_callback:
                    progress_callback(frame_idx, limit_frames, frame, counter)

                if frame_idx % 25 == 0 or frame_idx == limit_frames:
                    pct = (frame_idx / limit_frames * 100) if limit_frames > 0 else 0
                    print(f"[INFO] Processed frame {frame_idx}/{limit_frames} ({pct:.1f}%)")

        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during processing: {e}")
            return False
        finally:
            cap.release()
            writer.release()

        print("=" * 60)
        print("[INFO] Processing finished successfully!")
        if counter is not None:
            print(f"[INFO] Final Category Counts: {counter.class_counts}")
            print(f"[INFO] Total Counted Vehicles: {counter.total_count}")
        print(f"[INFO] Frames written: {written_count}")
        print(f"[INFO] Output saved to: {output_path}")
        print("=" * 60)
        return True


if __name__ == "__main__":
    detector = VehicleDetector()
    detector.process_video()

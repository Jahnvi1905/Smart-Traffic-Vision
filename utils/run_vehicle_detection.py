import os
import sys

# Ensure project root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detection.vehicle_detector import VehicleDetector
from counting.vehicle_counter import VehicleCounter


def main():
    input_video = os.path.join("data", "input", "sample.mp4")
    output_video = os.path.join("data", "output", "vehicle_counting.mp4")

    print("[RUNNER] Initializing Vehicle Detection, Tracking & Counting Pipeline...")
    detector = VehicleDetector(model_path="yolo11n.pt", conf_threshold=0.35)
    
    # Configurable virtual counting line (can be adjusted for any camera angle/resolution)
    counter = VehicleCounter(line_start=(50, 1200), line_end=(1030, 1200))

    success = detector.process_video(
        input_path=input_video,
        output_path=output_video,
        enable_tracking=True,
        tracker="bytetrack.yaml",
        counter=counter
    )
    
    if not success:
        print("[RUNNER] Pipeline failed or encountered an error.")
        sys.exit(1)
    else:
        print(f"[RUNNER] Pipeline finished successfully! Total vehicles counted: {counter.total_count}")
        print(f"[RUNNER] Counts by category: {counter.class_counts}")


if __name__ == "__main__":
    main()

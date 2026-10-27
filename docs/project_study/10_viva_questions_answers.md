# Document 10: Master Viva Question & Answer Bank (100+ Questions)

This comprehensive question bank is divided into 12 distinct technical categories. Each question includes a direct, scientifically rigorous answer formatted for speaking aloud during an engineering capstone viva or technical interview.

---

## Category 1: Project Overview & System Architecture (Questions 1–10)

### Q1: What is the primary problem your project solves?
**Answer**: Smart Traffic Vision automates urban traffic monitoring and safety enforcement using Computer Vision. It eliminates reliance on fatigued human CCTV operators by automatically performing multi-class vehicle detection, persistent multi-object tracking, bidirectional line-crossing counting, automated violation detection (unhelmeted riders, wrong-way driving, red-light intrusions), and license plate OCR.

### Q2: Why is automated traffic surveillance especially critical for countries like India?
**Answer**: Developing nations experience heterogeneous traffic featuring diverse vehicle categories (cars, auto-rickshaws, buses, motorcycles, trucks) moving without strict lane discipline. Manual policing cannot scale to monitor thousands of high-density intersections simultaneously.

### Q3: What are the three core architectural tiers of your system?
**Answer**: Our system is strictly decoupled into three layers:
1. **Computer Vision Layer**: Python, OpenCV, and Ultralytics YOLO running object detection, ByteTrack tracking, and violation logic.
2. **Backend Service Layer**: FastAPI and Uvicorn providing asynchronous REST endpoints and serving static evidence media.
3. **Frontend Presentation Layer**: A React 19 single-page application built with Vite and TypeScript rendering KPI metrics, synchronized telemetry charts, and evidence review modals.

### Q4: Why did you decouple the frontend from the Python backend instead of using an all-in-one framework?
**Answer**: Decoupling the frontend ensures that computationally intensive computer vision tasks run independently on the backend without blocking the user interface. It also allows multiple client dashboards or mobile clients to consume the same standardized REST API.

### Q5: What database is used in your project and why?
**Answer**: We use SQLite3 via a dedicated `DatabaseManager` class. SQLite is an embedded, zero-configuration, ACID-compliant relational database that stores records directly in a local file (`data/traffic_violations.db`), making the entire system portable and self-contained for edge deployment.

### Q6: What happens if the backend server crashes during video processing?
**Answer**: The video processing loop operates inside try-except blocks. If an unhandled exception occurs, an HTTP 500 error is returned with the trace, temporary files in `data/input/` are preserved for debugging, and already-committed SQLite events remain safe due to ACID transaction rollbacks.

### Q7: What is the difference between synchronous and asynchronous operations in your backend?
**Answer**: Synchronous operations block the execution thread until finished. Asynchronous operations in FastAPI use Python's `asyncio` event loop to handle network I/O, database queries, and client requests concurrently without spawning heavy OS threads for every connection.

### Q8: What video codec is used for saving processed streams and why?
**Answer**: We use the Cisco OpenH264 codec (`avc1` FourCC) via `openh264-2.5.0-win64.dll`. H.264 is natively supported across all modern web browsers (Chrome, Edge, Firefox, Safari), allowing processed MP4 videos to stream directly inside HTML5 `<video>` tags.

### Q9: What is the role of the `docs/` and `data/` directories?
**Answer**: `data/` stores persistent media assets: `data/input/` holds uploaded videos, `data/output/` holds processed videos, `data/output/violations/` holds tamper-evident JPEG evidence snapshots, and `data/traffic_violations.db` stores the SQLite database. `docs/` contains complete technical study and presentation guides.

### Q10: How does your system support real-time camera streaming?
**Answer**: OpenCV's `cv2.VideoCapture()` natively accepts RTSP (Real-Time Streaming Protocol) URLs (e.g. `rtsp://camera_ip:554/stream`) as well as video file paths. The frame-by-frame loop processes live camera feeds using the exact same detection, tracking, and violation logic.

---

## Category 2: Python, OpenCV & Core Video Fundamentals (Questions 11–20)

### Q11: How is a digital image represented in memory in Python?
**Answer**: A digital image is represented as a contiguous three-dimensional NumPy array of shape `(Height, Width, Channels)` with data type `uint8`, where pixel values range from 0 to 255.

### Q12: Why does OpenCV use BGR instead of RGB?
**Answer**: OpenCV was developed in 1999 when early camera sensors, frame grabbers, and Windows bitmap structures stored pixel channels in Blue-Green-Red order. OpenCV maintains BGR for legacy backward compatibility.

### Q13: How do you crop an image in NumPy using bounding box coordinates $[x_1, y_1, x_2, y_2]$?
**Answer**: Because NumPy uses row-major indexing `[row, column]`, the crop is extracted using spatial slicing: `crop = frame[y1:y2, x1:x2]`.

### Q14: What is the purpose of `cv2.addWeighted()` in your vehicle counter?
**Answer**: `cv2.addWeighted()` performs alpha-blending between two image arrays: $\text{dst} = \alpha \cdot \text{src1} + \beta \cdot \text{src2} + \gamma$. It is used in `counting/vehicle_counter.py` to render translucent dark HUD background banners without obscuring the traffic underneath.

### Q15: What is FPS and why is it important for trajectory tracking?
**Answer**: FPS stands for Frames Per Second. It dictates the temporal resolution between consecutive frames: $\Delta t = 1 / \text{FPS}$. Knowing the FPS allows us to convert frame indices into physical seconds (`timestamp_s = frame_idx / fps`) for telemetry synchronization and velocity calculations.

### Q16: How do you draw a bounding box and label in OpenCV?
**Answer**: A rectangle is drawn using `cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)`. Text labels are rendered above the box using `cv2.putText(image, text, (x1, y1 - 10), font, scale, color, thickness, cv2.LINE_AA)`.

### Q17: What does `cv2.LINE_AA` mean?
**Answer**: `cv2.LINE_AA` specifies an anti-aliased line drawing algorithm using Gaussian filtering, which eliminates jagged pixel staircasing on text and geometric shapes.

### Q18: What is the difference between lossless and lossy compression in video processing?
**Answer**: Lossless compression preserves exact raw pixel data without degradation but produces enormous file sizes. Lossy compression (such as H.264/AVC) removes imperceptible high-frequency visual information to achieve 50x–100x compression ratios suitable for streaming.

### Q19: Why do video resolutions need to have even dimensions for H.264 encoding?
**Answer**: H.264 uses $16 \times 16$ macroblocks or $8 \times 8$ transform blocks with 4:2:0 chroma subsampling. Chroma channels are halved horizontally and vertically, meaning an odd pixel width or height cannot be evenly subsampled, causing video encoder crashes. Our code explicitly resizes frames to even dimensions before writing.

### Q20: What is the Global Interpreter Lock (GIL) in Python and how does it impact OpenCV?
**Answer**: The GIL prevents multiple native OS threads from executing Python bytecodes simultaneously. However, OpenCV C++ functions and PyTorch tensor operations release the GIL during execution, allowing underlying C/C++ routines to utilize multi-core CPU parallelism.

---

## Category 3: YOLO & Deep Learning Object Detection (Questions 21–35)

### Q21: What does YOLO stand for and how does it differ from Faster R-CNN?
**Answer**: YOLO stands for "You Only Look Once". Faster R-CNN is a two-stage detector that first generates candidate region proposals (RPN) and then classifies each proposal. YOLO is a single-stage detector that predicts bounding boxes and class probabilities directly from the full image in a single neural network forward pass, making it fast enough for real-time video (30+ FPS).

### Q22: What specific YOLO architecture is used in your models?
**Answer**: Our models use the **Ultralytics YOLO11 nano (`YOLO11n`)** architecture, which features an optimized C3k2 convolutional backbone, spatial pyramid pooling fast (SPPF), and decoupled anchor-free detection heads.

### Q23: What is the advantage of the "nano" (n) model variant?
**Answer**: YOLO11n contains only ~2.6 million parameters and produces a tiny ~5.4 MB weight file. It achieves high frame rates on commodity CPU hardware without requiring expensive high-end server GPUs.

### Q24: What is an Anchor-Free detector?
**Answer**: Traditional anchor-based detectors predefine static reference boxes (anchors) of various scales and aspect ratios at each grid cell. Anchor-free detectors directly predict the distance from a grid point to the four boundaries of the object, reducing hyperparameter complexity and accelerating inference.

### Q25: What is a Feature Pyramid Network (FPN) and why does YOLO use it?
**Answer**: An FPN combines high-resolution shallow feature maps (which carry fine spatial detail) with low-resolution deep feature maps (which carry rich semantic context). This multi-scale representation allows YOLO to detect both small distant objects (like license plates or helmets) and large objects (like buses) simultaneously.

### Q26: What is a bounding box coordinate and what formats exist?
**Answer**: A bounding box defines the spatial rectangle enclosing an object. Common formats include:
- `xyxy`: $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ (top-left and bottom-right corners).
- `xywh`: $[x_{\text{center}}, y_{\text{center}}, \text{width}, \text{height}]$.

### Q27: What is Non-Maximum Suppression (NMS)?
**Answer**: NMS is a post-processing algorithm that eliminates redundant, overlapping candidate boxes predicted for the same object. It selects the highest-scoring box and suppresses all neighboring boxes whose Intersection over Union (IoU) exceeds a set threshold.

### Q28: How is Intersection over Union (IoU) calculated?
**Answer**: $\text{IoU} = \frac{\text{Area of Overlap}(A \cap B)}{\text{Area of Union}(A \cup B)}$. An IoU of 1.0 indicates perfect spatial alignment; an IoU of 0.0 indicates zero overlap.

### Q29: What happens if your detection confidence threshold is set too high (e.g. 0.80)?
**Answer**: Higher thresholds increase precision (reducing false alarms) but severely harm recall: faint, distant, or partially occluded vehicles and unhelmeted riders will be missed.

### Q30: What happens if your detection confidence threshold is set too low (e.g. 0.05)?
**Answer**: Lower thresholds maximize recall but generate excessive false positives: random shadows, road textures, or street signs will be erroneously classified as vehicles or helmets.

### Q31: What is the optimal confidence threshold used in your helmet pipeline and why?
**Answer**: We use `conf_thresh = 0.20`. Empirical testing on 30 ground-truth images proved that 0.20 achieves **93.3% recall (14/15)** on unhelmeted riders while maintaining **0% false alarms (0/15)** on compliant riders.

### Q32: What is batch size in deep learning inference?
**Answer**: Batch size is the number of images passed simultaneously through the neural network. In real-time video streaming, a batch size of 1 is used because frames arrive sequentially.

### Q33: What is the receptive field of a convolutional neural network?
**Answer**: The receptive field is the spatial area in the input image that can affect the activation of a specific neuron in a deeper feature layer. Deeper layers have larger receptive fields, allowing them to comprehend full context.

### Q34: How does YOLO handle multiple classes?
**Answer**: YOLO outputs a vector of class probability scores for every predicted bounding box using independent sigmoid activations (binary cross-entropy loss), allowing multi-label detection without assuming mutually exclusive classes.

### Q35: What is inference latency?
**Answer**: Inference latency is the time (in milliseconds) taken by the model to perform a single forward pass on an input tensor. On our system, YOLO11n executes in approximately 15–25 ms per frame.

---

## Category 4: ML Models, Training & Provenance (Questions 36–45)

### Q36: Did you train the YOLO models from scratch?
**Answer**: No. Training modern vision models from scratch requires tens of thousands of annotated images and weeks of GPU compute. We utilized transfer learning, fine-tuning pretrained YOLO11n weights on domain-specific traffic and helmet datasets.

### Q37: What are the exact classes of `models/vehicle_detector.pt`?
**Answer**: Exactly 6 classes: `{0: 'Bus', 1: 'Car', 2: 'Three-Wheeler', 3: 'Truck', 4: 'Two-Wheeler', 5: 'Van'}`.

### Q38: What are the exact classes of `models/helmet_detector.pt`?
**Answer**: Exactly 5 classes: `{0: 'Helmet', 1: 'fd', 2: 'motorcycle', 3: 'nohelmet', 4: 'platenumber'}`.

### Q39: What are the exact classes of `models/plate_detector.pt`?
**Answer**: Exactly 1 class: `{0: 'license_plate'}`.

### Q40: What is the source and size of the India Traffic dataset in your repository?
**Answer**: Sourced from Roboflow under `workspace: kaceys-workspace` (`datasets/india_traffic/`). It contains 901 total annotated images (631 train, 180 valid, 90 test) across the 6 vehicle classes.

### Q41: What is the source and size of the Motorcycle Helmet dataset in your repository?
**Answer**: Sourced from Roboflow under `workspace: traffic-violation` (`datasets/motorcycle_helmet/`). It contains 205 total annotated images (163 train, 41 valid, 1 test) across the 5 helmet model classes.

### Q42: What is the file `models/helmet_detector_backup.pt`?
**Answer**: It is a preserved backup snapshot of the helmet model weights (~5.45 MB) sharing the exact same 5-class architecture, maintained for disaster recovery.

### Q43: What is `yolo11n.pt` located in the root directory?
**Answer**: It is the standard base pretrained checkpoint from Ultralytics trained on the 80-class COCO (Common Objects in Context) dataset, used as a reference benchmark.

### Q44: What is transfer learning and why is it beneficial?
**Answer**: Transfer learning reuses visual features (edges, textures, shapes) learned by a model trained on a massive generic dataset (COCO) and fine-tunes the final classification layers on a smaller custom dataset (Indian traffic), achieving high accuracy with minimal training data.

### Q45: What is data augmentation in object detection?
**Answer**: Data augmentation synthetically expands training datasets by applying random geometric and photometric transformations (horizontal flips, scaling, HSV color jitter, mosaic stitching) during training to make the model invariant to lighting and perspective changes.

---

## Category 5: Multi-Object Tracking & ByteTrack (Questions 46–55)

### Q46: What is the difference between Object Detection and Multi-Object Tracking (MOT)?
**Answer**: Object Detection identifies what and where objects are in a single static frame (independent boxes without identity). Tracking links detections across consecutive frames to maintain persistent identities (`track_id = 42`) and reconstruct trajectory paths.

### Q47: Why is tracking necessary for automated traffic violation detection?
**Answer**: Without tracking, an unhelmeted rider appearing across 30 consecutive video frames would trigger 30 separate violation records. Tracking assigns a single `track_id`, enabling deduplication and temporal cooldown so only 1 violation is logged.

### Q48: How does ByteTrack associate objects across frames?
**Answer**: ByteTrack uses a two-stage association pipeline: first, high-confidence detections ($D_{\text{high}}$) are matched against active Kalman tracks using IoU distance; second, remaining unmatched tracks are matched against low-confidence detections ($D_{\text{low}}$) to recover temporarily occluded objects.

### Q49: What is a Kalman Filter and how does ByteTrack use it?
**Answer**: A Kalman filter is an optimal linear estimator that predicts the future state of a dynamic system based on noisy measurements. ByteTrack uses it to estimate the position and velocity of vehicle bounding boxes between frames.

### Q50: What is the Hungarian Algorithm?
**Answer**: The Hungarian (Munkres) algorithm is a combinatorial optimization algorithm that solves the bipartite assignment problem in $O(N^3)$ time, matching predicted track boxes to detected boxes with minimum total IoU cost.

### Q51: What is an ID switch (identity switch)?
**Answer**: An ID switch occurs when a tracker mistakenly swaps the identities of two crossing vehicles or re-assigns a new ID to a previously tracked vehicle after temporary occlusion.

### Q52: How does ByteTrack minimize ID switches compared to SORT?
**Answer**: SORT discards low-scoring detections, immediately dropping occluded vehicles. ByteTrack retains low-scoring boxes in its second matching stage, maintaining track continuity through partial occlusions and reducing ID switches.

### Q53: What is the `bytetrack.yaml` file in your repository?
**Answer**: It is the tracking configuration file used by Ultralytics specifying tracking parameters: `track_high_thresh: 0.5`, `track_low_thresh: 0.1`, `new_track_thresh: 0.6`, `track_buffer: 30` (frames to keep lost tracks alive), and `match_thresh: 0.8`.

### Q54: What happens when a tracked vehicle leaves the camera frame?
**Answer**: The Kalman filter predicts its position outside the frame. Because no detections match it, its `missed` frame count increments. Once `missed > track_buffer` (30 frames), the tracker permanently deletes the track from active memory.

### Q55: What is a pseudo-track ID in your helmet detector?
**Answer**: When the primary vehicle detector misses a motorcycle entirely (no ByteTrack ID available), our dedicated helmet detector's internal spatial tracker tracks the motorcycle centroid across consecutive frames ($< 90\text{px}$ distance) and assigns a stable pseudo-track ID (`#9001`, `#9002`...) to enforce cooldown and database deduplication.

---

## Category 6: Vehicle Counting & Spatial Geometry (Questions 56–63)

### Q56: How does your virtual-line vehicle counter work?
**Answer**: The counter defines a static virtual line between two points $(p_3, p_4)$. For every tracked vehicle, it records previous centroid $p_1$ and current centroid $p_2$. If trajectory segment $p_1 \to p_2$ intersects line $p_3 \to p_4$, the vehicle is counted.

### Q57: What is the Counter-Clockwise (`ccw`) test formula?
**Answer**: $\text{ccw}(a, b, c) = (b_x - a_x)(c_y - a_y) - (b_y - a_y)(c_x - a_x)$. It computes the 2D cross-product to determine whether three points form a counter-clockwise or clockwise turn.

### Q58: How does `segments_intersect()` use `ccw`?
**Answer**: Two segments intersect if and only if the endpoints of each segment lie on opposite sides of the line containing the other segment: $\text{ccw}(p_3, p_4, p_1) \times \text{ccw}(p_3, p_4, p_2) < 0$ AND $\text{ccw}(p_1, p_2, p_3) \times \text{ccw}(p_1, p_2, p_4) < 0$.

### Q59: How do you prevent double-counting vehicles moving slowly on the line?
**Answer**: When a vehicle intersects the line, its `track_id` is stored in an in-memory set (`self.counted_ids`). In all subsequent frames, the counter checks `if track_id in self.counted_ids:` in $O(1)$ time and skips re-counting.

### Q60: What is the difference between cumulative counts and visible counts?
**Answer**: **Cumulative count** is the total number of unique vehicles that have crossed the counting line since the session started. **Visible count** is the number of vehicles physically present inside the camera frame at the current instantaneous moment.

### Q61: What is per-frame count telemetry?
**Answer**: It is a chronological array of metrics recorded at each frame index containing `timestamp_s`, `cumulative_total`, `cumulative_classes`, `visible_total`, and `visible_classes`. In our React dashboard, this data is synchronized with HTML5 video playback to update graphs in real time.

### Q62: Can your counter track bidirectional traffic?
**Answer**: Yes. The vector cross-product sign indicates whether the vehicle crossed from top-to-bottom or bottom-to-top, allowing inbound and outbound counts to be tallied separately.

### Q63: Where is the counting line positioned in the video?
**Answer**: The counting line is configurable. In `api/routes/traffic.py`, it defaults to the lower third of the video height ($y = 0.65 \times \text{height}$) where vehicles are largest and most accurately tracked.

---

## Category 7: Helmet Detection Pipeline & Bottleneck Resolution (Questions 64–75)

### Q64: What was the primary bottleneck in the initial helmet detection pipeline?
**Answer**: The initial pipeline depended on a strict serial chain: `vehicle_detector.pt` had to first detect a two-wheeler, and the code cropped the top 45% of that box to feed the helmet model. Because `vehicle_detector.pt` was trained on overhead CCTV, it missed frontal/ground-level motorcycles or cropped only the wheels, resulting in **zero NO_HELMET events**.

### Q65: How did you fix this bottleneck without retraining or downloading new models?
**Answer**: We discovered that `models/helmet_detector.pt` natively includes a `motorcycle` class (class 2) as well as `Helmet` (0) and `nohelmet` (3). We implemented a **dedicated spatial helmet-first detection path** in `HelmetViolationDetector.detect_frame_violations()` that runs on the entire frame, using the helmet model's own motorcycle detections as spatial anchors without depending on `vehicle_detector.pt`.

### Q66: What is the rider search zone formula used in spatial association?
**Answer**: For a motorcycle bounding box $[mx_1, my_1, mx_2, my_2]$ with width $mw$ and height $mh$, the rider search zone is computed as:
$$rx_1 = \max(0, mx_1 - 0.15 \cdot mw), \quad ry_1 = \max(0, my_1 - 0.25 \cdot mh)$$
$$rx_2 = \min(\text{width}, mx_2 + 0.15 \cdot mw), \quad ry_2 = \min(\text{height}, my_1 + 0.65 \cdot mh)$$

### Q67: How does your pipeline resolve multi-rider scenes (e.g. one rider with helmet, one without)?
**Answer**: Rather than classifying the whole frame globally, each motorcycle defines its own localized rider search zone. Head detections are matched to their respective motorcycle. A compliant rider on Motorcycle A does not suppress an unhelmeted rider on Motorcycle B.

### Q68: What happens if `Helmet` and `nohelmet` are detected on the exact same head?
**Answer**: This represents model ambiguity (competing co-detection). Our code compares confidences: if `nohelmet_conf > helmet_conf`, a violation is confirmed; if `helmet_conf >= nohelmet_conf`, the rider is conservatively treated as compliant to prevent false accusations.

### Q69: What happens if an unhelmeted rider's head is detected but the motorcycle is occluded?
**Answer**: The code enters **Case B** (untracked head): it flags the unhelmeted head, synthesizes a context box downwards, sets `track_id = None`, and uses spatial grid cooldown (`loc_cx_cy`) to log the violation without crashing.

### Q70: What is the cooldown duration in your helmet violation detector?
**Answer**: The default cooldown is `10.0` seconds (configurable). Once a motorcycle or rider position triggers a violation, subsequent alerts for that ID or spatial cell are suppressed until the cooldown timer expires.

### Q71: How does the database prevent duplicate helmet violation rows?
**Answer**: In `DatabaseManager.insert_event()`, when `prevent_duplicates=True` and `track_id` is provided, SQLite checks `SELECT id FROM events WHERE event_type = ? AND track_id = ?`. If a row already exists, the insert is skipped and the existing ID is returned.

### Q72: What visual evidence is generated for a NO_HELMET violation?
**Answer**: An annotated JPEG saved to `data/output/violations/` featuring:
1. A red rectangle enclosing the motorcycle.
2. A distinct orange/amber box enclosing the rider's bare head labeled `NO HELMET`.
3. A top header banner showing event type, track ID, and confidence.
4. A bottom audit footer showing timestamp and model confidence string.

### Q73: What recall did your dedicated helmet detector achieve on ground-truth images?
**Answer**: **93.3% recall (14 out of 15 ground-truth unhelmeted images)** at `conf_thresh = 0.20`.

### Q74: What false positive rate did your detector achieve on helmet-compliant riders?
**Answer**: **0.0% false positive rate (0 false alarms out of 15 ground-truth compliant images)**.

### Q75: How many NO_HELMET events were generated when processing the real test video `helmet_test.mp4`?
**Answer**: Exactly **3 distinct NO_HELMET events** were logged (Events #31, #32, and #33), with zero false alarms on compliant frames.

---

## Category 8: License Plate Recognition & EasyOCR (Questions 76–85)

### Q76: What is ANPR / ALPR?
**Answer**: Automatic Number Plate Recognition (or Automatic License Plate Recognition) is a vision technology that localizes vehicle license plates and converts the pixel imagery into digital alphanumeric text strings.

### Q77: Why did you separate plate detection from plate character recognition?
**Answer**: A full surveillance frame is large (e.g. $1920 \times 1080$), while a license plate occupies a tiny area ($120 \times 40$ pixels). Running OCR on the full frame is slow and error-prone. We first use a lightweight YOLO model (`plate_detector.pt`) to locate the plate coordinates, crop the sub-image, and run EasyOCR exclusively on that small crop.

### Q78: What neural network architectures comprise EasyOCR?
**Answer**: EasyOCR uses two deep learning models:
1. **CRAFT**: A convolutional network that detects character region heatmaps and affinity between adjacent characters.
2. **ResNet-BiLSTM-CTC**: A ResNet convolutional feature extractor paired with a Bidirectional LSTM sequence reader and CTC (Connectionist Temporal Classification) decoding.

### Q79: What is CTC (Connectionist Temporal Classification) decoding?
**Answer**: CTC is a loss and decoding mechanism that allows sequence models to output variable-length character sequences without requiring each character in the image to be pre-segmented into individual bounding boxes.

### Q80: What was the bug identified in raw OCR output during the hardening pass?
**Answer**: EasyOCR frequently emitted bracket punctuation artifacts from plate mounting borders or screws, such as `'MH 20 EE 7602]'` with confidence 0.476.

### Q81: How does `normalize_plate_text()` clean plate readings?
**Answer**: It normalizes whitespace, converts text to uppercase, strips delimiter noise (`-._`), and strips leading/trailing punctuation brackets (`[]{}()<>;:.,*+=!?/\\`) without fabricating or guessing alphanumeric characters.

### Q82: What is your policy on OCR confidence thresholds?
**Answer**: In `ocr/plate_ocr.py`, `min_confidence_threshold = 0.50`. If OCR confidence $< 0.50$, `validate_plate_output()` tags `is_low_confidence = True` and `confidence_status = 'low'`, preserving the raw output for human audit rather than silently guessing characters.

### Q83: What regex is used to validate standard Indian registration plate formats?
**Answer**: `^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$` (e.g. `MH 12 AB 1234` or `DL 01 C 5678`).

### Q84: How are license plates spatially associated with tracked vehicles?
**Answer**: In `PlateAssociationManager.process_frame()`, a plate is linked to tracked vehicle $T_i$ if the plate's geometric center falls inside the vehicle's bounding box, or if their mutual overlap exceeds a 60% plausibility threshold.

### Q85: What happens if EasyOCR cannot read a plate on an offending vehicle?
**Answer**: The violation is still logged with `plate_text = 'UNKNOWN'`, confidence `0.0`, and the full visual evidence snapshot is saved so a human operator can review the vehicle visually.

---

## Category 9: Wrong-Way & Red-Light Violations (Questions 86–92)

### Q86: How does your system detect wrong-way driving?
**Answer**: `WrongWayDetector` tracks the centroid trajectory of each vehicle across a rolling buffer of 10 positions. If net displacement exceeds 25 pixels in the direction opposite to the configured `expected_direction` (e.g. moving UP in a DOWN corridor), a `WRONG_WAY` violation is flagged.

### Q87: How do you prevent stationary vehicles in traffic jams from triggering wrong-way violations?
**Answer**: If total centroid displacement is less than 25 pixels, the vehicle is classified as stationary or idling; no wrong-way alert can trigger.

### Q88: What traffic directions are configurable in `WrongWayDetector`?
**Answer**: Four cardinal directions: `'down'` (+y), `'up'` (-y), `'left'` (-x), and `'right'` (+x).

### Q89: How does your virtual red-light detector work?
**Answer**: A virtual stop-line is configured at the intersection approach. When `signal_state == "RED"`, any tracked vehicle whose centroid transitions across the stop-line from the approach side to the intersection side triggers a `RED_LIGHT` violation.

### Q90: Can a vehicle trigger a red-light violation if the signal state is GREEN?
**Answer**: No. `RedLightDetector.check_crossing()` explicitly gates execution: if `self.signal_state != "RED"`, it immediately returns `None`.

### Q91: Does your project perform physical traffic light bulb detection?
**Answer**: No. We do not claim optical traffic light recognition from camera feeds. The signal state is configured as an explicit operational parameter (`signal_state='RED'`), ensuring zero fabricated citations.

### Q92: Where are red-light and wrong-way evidence images saved?
**Answer**: High-contrast annotated evidence JPEGs are written to `data/output/violations/` named `wrongway_viol_{ts}_t{tid}.jpg` and `redlight_viol_{ts}_t{tid}.jpg`.

---

## Category 10: FastAPI Backend & SQLite Database (Questions 93–102)

### Q93: What is FastAPI and what ASGI server runs it?
**Answer**: FastAPI is a modern, high-performance web framework for building REST APIs with Python 3.8+ based on standard Python type hints. It runs on **Uvicorn**, a lightning-fast ASGI (Asynchronous Server Gateway Interface) server built on `uvloop` and `httptools`.

### Q94: What is Pydantic and how is it used in your backend?
**Answer**: Pydantic is a data validation library that enforces type hints at runtime. In `api/schemas.py`, Pydantic models define request and response payloads, ensuring automatic JSON serialization and returning descriptive 422 Unprocessable Entity errors when inputs are invalid.

### Q95: What HTTP status codes are returned by your API endpoints?
**Answer**:
- `200 OK`: Successful retrieval or update.
- `400 Bad Request`: Invalid parameters (e.g. attempting to review an immutable `COUNT` event).
- `404 Not Found`: Event ID does not exist in database.
- `422 Unprocessable Entity`: Request body violates Pydantic schema validation.
- `500 Internal Server Error`: Unhandled server-side processing failure.

### Q96: What HTTP methods are used across your endpoints?
**Answer**:
- `GET`: For idempotent data retrieval (`/api/health`, `/api/stats/overview`, `/api/violations`, `/api/reports/export`).
- `POST`: For uploading and processing media (`/api/traffic/analyze-video`, `/api/traffic/analyze-image`, `/api/plates/scan`).
- `PATCH`: For partial in-place status updates (`/api/violations/{id}/review`).

### Q97: What is CORS and how did you configure it?
**Answer**: Cross-Origin Resource Sharing (CORS) is a browser security mechanism that restricts web pages from making AJAX requests to a different domain or port. In `api/main.py`, `CORSMiddleware` allows requests from `http://localhost:5173` (Vite dev server).

### Q98: How are COUNT events stored in your SQLite database?
**Answer**: COUNT events are stored with `event_type = 'COUNT'`, `vehicle_class = 'Traffic Stream'`, `plate_text = NULL`, structured category counts serialized as JSON in `count_details`, and total line crossings in `total_count`.

### Q99: Why must COUNT events NEVER be stored with count dictionaries in `plate_text`?
**Answer**: In early legacy versions, count dictionaries were stored in `plate_text` (e.g. `plate_text = "{'car': 10}"`), which broke database schema normalization, corrupted text searches, and caused COUNT summaries to appear as safety violations. We executed a database migration to cleanly isolate count data into dedicated columns.

### Q100: Can an operator approve or reject a COUNT event via `PATCH /api/violations/{id}/review`?
**Answer**: No. `PATCH /api/violations/{id}/review` explicitly checks the event type: if `event_type == 'COUNT'`, it rejects the request with `HTTP 400 Bad Request: 'COUNT events are analytical summary records and cannot be reviewed'`.

### Q101: How does your API support pagination in `/api/violations`?
**Answer**: The endpoint accepts `page` (default 1) and `limit` (default 10) query parameters. It uses SQLite `LIMIT ? OFFSET ?` arithmetic: $\text{offset} = (\text{page} - 1) \times \text{limit}$, returning `total`, `page`, `limit`, `pages`, and `items[]`.

### Q102: How does `GET /api/reports/export` generate CSV downloads?
**Answer**: It queries filtered records from SQLite, converts them into a Pandas DataFrame, serializes them to an in-memory string buffer (`io.StringIO`), and streams the result with header `Content-Disposition: attachment; filename="traffic_violations_export.csv"`.

---

## Category 11: React 19 Frontend Dashboard & Visual Analytics (Questions 103–110)

### Q103: What are the main technologies used in your frontend?
**Answer**: React 19, TypeScript, Vite, Recharts for analytics, Lucide-React for icons, and Vanilla CSS with dark control-room design tokens.

### Q104: Why use TypeScript instead of plain JavaScript?
**Answer**: TypeScript provides compile-time type checking. Interfaces in `frontend/src/api/types.ts` strictly mirror backend Pydantic schemas, catching property typos and null-reference errors before code ever runs in the browser.

### Q105: What is Vite and why is it faster than Webpack (Create React App)?
**Answer**: Vite leverages native browser ES Modules (ESM) to serve source files instantly without bundling during development. For production, it uses highly optimized Rollup bundling, completing our production build in under 1 second (972 ms).

### Q106: How do Recharts components work in your Overview page?
**Answer**: Recharts uses composable SVG components (`<ResponsiveContainer>`, `<BarChart>`, `<PieChart>`, `<Tooltip>`). It binds declaratively to React state: when `/api/stats/overview` returns new category breakdown data, Recharts re-renders the charts smoothly with hardware-accelerated SVG transitions.

### Q107: What is an optimistic UI update and where is it used?
**Answer**: When an operator approves or rejects an infraction in `SafetyViolationsPage.tsx`, the local React state updates immediately to reflect the new badge color before awaiting a full page reload, providing a responsive user experience.

### Q108: How does the video player in `TrafficAnalysisPage.tsx` synchronize with telemetry?
**Answer**: The HTML5 `<video>` element has an `onTimeUpdate` event listener that reads `video.currentTime`. The component matches current playback time against the `telemetry[]` array returned by the API, updating the visible vehicle count and category cards in real time.

### Q109: How does the frontend resolve relative media URLs like `/media/violations/helmet_1.jpg`?
**Answer**: `getFullMediaUrl()` in `api/client.ts` prepends the backend base origin (`http://127.0.0.1:8000`) in development or uses relative paths when proxied by Vite, ensuring images and videos load seamlessly.

### Q110: What is the purpose of `DiagnosticsPage.tsx`?
**Answer**: It queries `GET /api/diagnostics` and displays live hardware memory, disk storage, model file paths, SQLite connection health, and OpenCV OpenH264 DLL status for administrative auditing.

---

## Category 12: Testing, Metrics & Engineering Hardening (Questions 111–118)

### Q111: What testing framework is used and how many tests exist?
**Answer**: We use Python's `unittest` framework executed via `pytest`. There are **52 automated unit, integration, and regression tests**, all of which currently pass (**52/52 PASSED**).

### Q112: What is the difference between a Unit Test and an Integration Test?
**Answer**: A **Unit Test** evaluates a single isolated class or function with mocked dependencies (e.g. testing `ccw()` vector math or mocked YOLO box parsing). An **Integration Test** verifies end-to-end multi-component interaction (e.g. submitting a video through FastAPI, executing the vision pipeline, committing to SQLite, and asserting on database records).

### Q113: What is the F1-Score formula and why is it useful?
**Answer**: $F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$. It provides the harmonic mean between precision and recall, balancing false alarms against missed detections.

### Q114: How did you test that duplicate events are not inserted into the database?
**Answer**: In `test_phase1_pipeline.py` and `test_helmet_e2e.py`, we execute `insert_event(..., prevent_duplicates=True)` twice with identical `track_id` and `event_type`. The test asserts that the second call returns the existing record ID and the total row count in SQLite remains exactly 1.

### Q115: How did you verify that helmet detection does not trigger false positives on compliant riders?
**Answer**: We tested 15 verified ground-truth images of riders wearing helmets (`test_compliant_rider_no_false_positive`). The detector generated **0 false alarms (0.0% false positive rate)** because our spatial logic requires `nohelmet_conf > helmet_conf` before flagging.

### Q116: What is the purpose of `tests/test_hardening.py`?
**Answer**: It verifies our database hardening pass: ensuring `COUNT` events are stored in structured columns, validating legacy schema migration, preventing review actions on COUNT records, and asserting that OCR outputs with confidence $< 0.50$ are tagged as low-confidence.

### Q117: What discrepancy exists between your `README.md` and current test results?
**Answer**: The `README.md` historically stated "Execute all 25 automated tests". Following our subsequent development phases, hardening passes, and dedicated helmet regression additions, the active test suite has expanded to **52 automated tests**.

### Q118: Why is claiming 100% real-world accuracy unscientific for computer vision projects?
**Answer**: Benchmark test suites run on finite sample sets under controlled conditions. Real-world traffic feeds introduce unpredictable environmental perturbations—torrential downpours, lens occlusion, lens distortion, high-speed motion blur, extreme headlight glare, and severe vehicle occlusions—that inevitably lower model confidence in production.


import cv2
import csv
import time
import numpy as np
import tensorflow as tf

from pathlib import Path
from collections import deque
from datetime import datetime

# ==================================================
# 1. CONFIGURATION
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_facial_expression_model.keras"
CASCADE_PATH = BASE_DIR / "models" / "haarcascade_frontalface_default.xml"

LOG_DIR = BASE_DIR / "logs"

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

IMG_SIZE = (48, 48)
SMOOTHING_FRAMES = 10
CONFIDENCE_THRESHOLD = 0.40
MAX_MISSED_FRAMES = 12
LOG_INTERVAL_SECONDS = 1.0

# ==================================================
# 2. LOAD MODEL AND FACE DETECTOR
# ==================================================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not CASCADE_PATH.exists():
    raise FileNotFoundError(f"Face detector not found: {CASCADE_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))

if face_detector.empty():
    raise RuntimeError(f"Could not load face detector: {CASCADE_PATH}")

print("AI model loaded successfully!")
print("Face detector loaded successfully!")

# ==================================================
# 3. FACE TRACKING HELPERS
# ==================================================
def get_center(box):
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


def box_iou(box_a, box_b):
    """Calculate overlap between two face boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_detections_to_tracks(detections, tracks):
    """Match current detections with previous face tracks."""
    candidates = []

    for detection_index, detection in enumerate(detections):
        dcx, dcy = get_center(detection)

        for track_id, track in tracks.items():
            tcx, tcy = get_center(track["box"])

            distance = np.hypot(dcx - tcx, dcy - tcy)

            max_distance = max(
                70,
                0.75 * max(
                    detection[2], detection[3],
                    track["box"][2], track["box"][3]
                )
            )

            overlap = box_iou(detection, track["box"])

            if overlap >= 0.05 or distance <= max_distance:
                cost = distance / max_distance - 0.5 * overlap
                candidates.append(
                    (cost, detection_index, track_id)
                )

    candidates.sort(key=lambda item: item[0])

    matches = {}
    used_detections = set()
    used_tracks = set()

    for _, detection_index, track_id in candidates:
        if detection_index in used_detections:
            continue
        if track_id in used_tracks:
            continue

        matches[detection_index] = track_id
        used_detections.add(detection_index)
        used_tracks.add(track_id)

    return matches


# ==================================================
# 4. SESSION CSV LOGGING
# ==================================================
LOG_DIR.mkdir(parents=True, exist_ok=True)

session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = LOG_DIR / f"session_{session_id}.csv"

fieldnames = [
    "timestamp",
    "face_id",
    "expression",
    "confidence_percent",
    "inference_latency_ms",
    "face_x",
    "face_y",
    "face_width",
    "face_height"
] + [f"score_{emotion}" for emotion in EMOTIONS]

# ==================================================
# 5. START CAMERA AND TRACKING
# ==================================================
tracks = {}
next_track_id = 1

fps = 0.0
previous_frame_time = time.perf_counter()
latest_inference_ms = 0.0

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Camera unavailable. Close other camera apps or try index 1."
    )

print(f"Session CSV: {CSV_PATH}")
print("Multi-face recognition started. Press Q to quit.")

try:
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )
        csv_writer.writeheader()

        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detected_faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(60, 60)
            )

            detections = [
                (int(x), int(y), int(w), int(h))
                for x, y, w, h in detected_faces
            ]

            matches = match_detections_to_tracks(
                detections, tracks
            )

            # Mark existing tracks as temporarily missing.
            for track in tracks.values():
                track["missed"] += 1

            # Process each face independently.
            for detection_index, box in enumerate(detections):
                x, y, w, h = box

                if detection_index in matches:
                    track_id = matches[detection_index]
                else:
                    track_id = next_track_id
                    next_track_id += 1

                    tracks[track_id] = {
                        "box": box,
                        "history": deque(
                            maxlen=SMOOTHING_FRAMES
                        ),
                        "missed": 0,
                        "label": "Analyzing",
                        "confidence": 0.0,
                        "last_logged": 0.0
                    }

                track = tracks[track_id]
                track["box"] = box
                track["missed"] = 0

                # Prepare face image for the trained CNN.
                face_gray = gray[y:y + h, x:x + w]
                face_gray = cv2.resize(face_gray, IMG_SIZE)
                face_input = (
                    face_gray.astype("float32") / 255.0
                )
                face_input = np.expand_dims(
                    face_input, axis=(0, -1)
                )

                # Measure model inference time.
                inference_start = time.perf_counter()

                scores = model.predict(
                    face_input, verbose=0
                )[0]

                elapsed_ms = (
                    time.perf_counter() - inference_start
                ) * 1000

                latest_inference_ms = elapsed_ms

                # Each face has a separate prediction history.
                track["history"].append(scores.copy())

                smoothed_scores = np.mean(
                    np.asarray(track["history"]), axis=0
                )

                best_index = int(np.argmax(smoothed_scores))
                confidence = float(smoothed_scores[best_index])

                if confidence < CONFIDENCE_THRESHOLD:
                    label = "Uncertain"
                else:
                    label = EMOTIONS[best_index].capitalize()

                track["label"] = label
                track["confidence"] = confidence

                # Save one CSV row per face per interval.
                now_seconds = time.monotonic()

                if (
                    now_seconds - track["last_logged"]
                    >= LOG_INTERVAL_SECONDS
                ):
                    row = {
                        "timestamp": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        "face_id": track_id,
                        "expression": label,
                        "confidence_percent": round(
                            confidence * 100, 2
                        ),
                        "inference_latency_ms": round(
                            elapsed_ms, 2
                        ),
                        "face_x": x,
                        "face_y": y,
                        "face_width": w,
                        "face_height": h
                    }

                    for i, emotion in enumerate(EMOTIONS):
                        row[f"score_{emotion}"] = round(
                            float(smoothed_scores[i]) * 100, 2
                        )

                    csv_writer.writerow(row)
                    csv_file.flush()
                    track["last_logged"] = now_seconds

                # Draw face box and label.
                color = (
                    (0, 200, 0)
                    if label != "Uncertain"
                    else (0, 165, 255)
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

                text = (
                    f"Face {track_id}: {label} "
                    f"{confidence * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    text,
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    color,
                    2,
                    cv2.LINE_AA
                )

            # Remove tracks missing for too many frames.
            expired_ids = [
                track_id
                for track_id, track in tracks.items()
                if track["missed"] > MAX_MISSED_FRAMES
            ]

            for track_id in expired_ids:
                del tracks[track_id]

            # Calculate approximate overall processing FPS.
            current_time = time.perf_counter()
            frame_seconds = current_time - previous_frame_time
            previous_frame_time = current_time

            if frame_seconds > 0:
                instant_fps = 1.0 / frame_seconds

                if fps == 0:
                    fps = instant_fps
                else:
                    fps = 0.9 * fps + 0.1 * instant_fps

            # Performance overlay.
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Latest inference: {latest_inference_ms:.1f} ms",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Active face tracks: {len(tracks)}",
                
import cv2
import csv
import time
import numpy as np
import tensorflow as tf

from pathlib import Path
from collections import deque
from datetime import datetime

# ==================================================
# 1. CONFIGURATION
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_facial_expression_model.keras"
CASCADE_PATH = BASE_DIR / "models" / "haarcascade_frontalface_default.xml"

LOG_DIR = BASE_DIR / "logs"

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

IMG_SIZE = (48, 48)
SMOOTHING_FRAMES = 10
CONFIDENCE_THRESHOLD = 0.40
MAX_MISSED_FRAMES = 12
LOG_INTERVAL_SECONDS = 1.0

# ==================================================
# 2. LOAD MODEL AND FACE DETECTOR
# ==================================================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not CASCADE_PATH.exists():
    raise FileNotFoundError(f"Face detector not found: {CASCADE_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))

if face_detector.empty():
    raise RuntimeError(f"Could not load face detector: {CASCADE_PATH}")

print("AI model loaded successfully!")
print("Face detector loaded successfully!")

# ==================================================
# 3. FACE TRACKING HELPERS
# ==================================================
def get_center(box):
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


def box_iou(box_a, box_b):
    """Calculate overlap between two face boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_detections_to_tracks(detections, tracks):
    """Match current detections with previous face tracks."""
    candidates = []

    for detection_index, detection in enumerate(detections):
        dcx, dcy = get_center(detection)

        for track_id, track in tracks.items():
            tcx, tcy = get_center(track["box"])

            distance = np.hypot(dcx - tcx, dcy - tcy)

            max_distance = max(
                70,
                0.75 * max(
                    detection[2], detection[3],
                    track["box"][2], track["box"][3]
                )
            )

            overlap = box_iou(detection, track["box"])

            if overlap >= 0.05 or distance <= max_distance:
                cost = distance / max_distance - 0.5 * overlap
                candidates.append(
                    (cost, detection_index, track_id)
                )

    candidates.sort(key=lambda item: item[0])

    matches = {}
    used_detections = set()
    used_tracks = set()

    for _, detection_index, track_id in candidates:
        if detection_index in used_detections:
            continue
        if track_id in used_tracks:
            continue

        matches[detection_index] = track_id
        used_detections.add(detection_index)
        used_tracks.add(track_id)

    return matches


# ==================================================
# 4. SESSION CSV LOGGING
# ==================================================
LOG_DIR.mkdir(parents=True, exist_ok=True)

session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = LOG_DIR / f"session_{session_id}.csv"

fieldnames = [
    "timestamp",
    "face_id",
    "expression",
    "confidence_percent",
    "inference_latency_ms",
    "face_x",
    "face_y",
    "face_width",
    "face_height"
] + [f"score_{emotion}" for emotion in EMOTIONS]

# ==================================================
# 5. START CAMERA AND TRACKING
# ==================================================
tracks = {}
next_track_id = 1

fps = 0.0
previous_frame_time = time.perf_counter()
latest_inference_ms = 0.0

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Camera unavailable. Close other camera apps or try index 1."
    )

print(f"Session CSV: {CSV_PATH}")
print("Multi-face recognition started. Press Q to quit.")

try:
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )
        csv_writer.writeheader()

        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detected_faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(60, 60)
            )

            detections = [
                (int(x), int(y), int(w), int(h))
                for x, y, w, h in detected_faces
            ]

            matches = match_detections_to_tracks(
                detections, tracks
            )

            # Mark existing tracks as temporarily missing.
            for track in tracks.values():
                track["missed"] += 1

            # Process each face independently.
            for detection_index, box in enumerate(detections):
                x, y, w, h = box

                if detection_index in matches:
                    track_id = matches[detection_index]
                else:
                    track_id = next_track_id
                    next_track_id += 1

                    tracks[track_id] = {
                        "box": box,
                        "history": deque(
                            maxlen=SMOOTHING_FRAMES
                        ),
                        "missed": 0,
                        "label": "Analyzing",
                        "confidence": 0.0,
                        "last_logged": 0.0
                    }

                track = tracks[track_id]
                track["box"] = box
                track["missed"] = 0

                # Prepare face image for the trained CNN.
                face_gray = gray[y:y + h, x:x + w]
                face_gray = cv2.resize(face_gray, IMG_SIZE)
                face_input = (
                    face_gray.astype("float32") / 255.0
                )
                face_input = np.expand_dims(
                    face_input, axis=(0, -1)
                )

                # Measure model inference time.
                inference_start = time.perf_counter()

                scores = model.predict(
                    face_input, verbose=0
                )[0]

                elapsed_ms = (
                    time.perf_counter() - inference_start
                ) * 1000

                latest_inference_ms = elapsed_ms

                # Each face has a separate prediction history.
                track["history"].append(scores.copy())

                smoothed_scores = np.mean(
                    np.asarray(track["history"]), axis=0
                )

                best_index = int(np.argmax(smoothed_scores))
                confidence = float(smoothed_scores[best_index])

                if confidence < CONFIDENCE_THRESHOLD:
                    label = "Uncertain"
                else:
                    label = EMOTIONS[best_index].capitalize()

                track["label"] = label
                track["confidence"] = confidence

                # Save one CSV row per face per interval.
                now_seconds = time.monotonic()

                if (
                    now_seconds - track["last_logged"]
                    >= LOG_INTERVAL_SECONDS
                ):
                    row = {
                        "timestamp": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        "face_id": track_id,
                        "expression": label,
                        "confidence_percent": round(
                            confidence * 100, 2
                        ),
                        "inference_latency_ms": round(
                            elapsed_ms, 2
                        ),
                        "face_x": x,
                        "face_y": y,
                        "face_width": w,
                        "face_height": h
                    }

                    for i, emotion in enumerate(EMOTIONS):
                        row[f"score_{emotion}"] = round(
                            float(smoothed_scores[i]) * 100, 2
                        )

                    csv_writer.writerow(row)
                    csv_file.flush()
                    track["last_logged"] = now_seconds

                # Draw face box and label.
                color = (
                    (0, 200, 0)
                    if label != "Uncertain"
                    else (0, 165, 255)
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

                text = (
                    f"Face {track_id}: {label} "
                    f"{confidence * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    text,
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    color,
                    2,
                    cv2.LINE_AA
                )

            # Remove tracks missing for too many frames.
            expired_ids = [
                track_id
                for track_id, track in tracks.items()
                if track["missed"] > MAX_MISSED_FRAMES
            ]

            for track_id in expired_ids:
                del tracks[track_id]

            # Calculate approximate overall processing FPS.
            current_time = time.perf_counter()
            frame_seconds = current_time - previous_frame_time
            previous_frame_time = current_time

            if frame_seconds > 0:
                instant_fps = 1.0 / frame_seconds

                if fps == 0:
                    fps = instant_fps
                else:
                    fps = 0.9 * fps + 0.1 * instant_fps

            # Performance overlay.
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Latest inference: {latest_inference_ms:.1f} ms",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Active face tracks: {len(tracks)}",
                (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "Facial Expression Recognition - Multi-Face",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")
    print(f"Session CSV saved to: {CSV_PATH}")

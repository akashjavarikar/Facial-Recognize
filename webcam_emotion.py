#python
from pathlib import Path
from collections import deque
from datetime import datetime
import csv
import time

import cv2
import numpy as np
import tensorflow as tf

# ---------------- PERFORMANCE SETTINGS ----------------

cv2.setNumThreads(4)

# ---------------- CONFIGURATION ----------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_facial_expression_model.keras"
CASCADE_PATH = PROJECT_DIR / "models" / "haarcascade_frontalface_default.xml"
LOG_DIR = PROJECT_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_LABELS = [
    "Angry", "Disgust", "Fear", "Happy",
    "Neutral", "Sad", "Surprise"
]

FACE_SIZE = (48, 48)
CONFIDENCE_THRESHOLD = 40.0
SMOOTHING_FRAMES = 10
LOG_INTERVAL_SECONDS = 1.0
TRACK_TIMEOUT_FRAMES = 15

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Detect faces every 2 frames to reduce CPU usage.
DETECTION_INTERVAL = 2

# ---------------- LOAD MODEL ----------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not CASCADE_PATH.exists():
    raise FileNotFoundError(f"Face detector not found: {CASCADE_PATH}")

print("Loading emotion model...")

model = tf.keras.models.load_model(MODEL_PATH)
face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))

if face_detector.empty():
    raise RuntimeError("Could not load Haar cascade XML.")

if model.output_shape[-1] != len(EMOTION_LABELS):
    raise ValueError("Model output count does not match emotion labels.")

print("Model loaded:", MODEL_PATH.name)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("OpenCV threads:", cv2.getNumThreads())
print("Face detection interval:", DETECTION_INTERVAL)

# ---------------- SESSION CSV ----------------

session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S.csv")
csv_path = LOG_DIR / session_name

CSV_COLUMNS = [
    "timestamp",
    "face_id",
    "expression",
    "confidence_percent",
    "inference_latency_ms",
    "fps",
    "face_x",
    "face_y",
    "face_w",
    "face_h",
] + [f"score_{label.lower()}" for label in EMOTION_LABELS]

tracks = {}
next_face_id = 1


# ---------------- FACE TRACKING ----------------

def calculate_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection

    return intersection / union if union > 0 else 0.0


def get_track_id(box, frame_number):
    global next_face_id

    x, y, w, h = box
    center_x = x + w / 2
    center_y = y + h / 2

    best_id = None
    best_score = -1.0

    for face_id, track in tracks.items():
        if frame_number - track["last_seen"] > TRACK_TIMEOUT_FRAMES:
            continue

        ox, oy, ow, oh = track["box"]
        old_cx = ox + ow / 2
        old_cy = oy + oh / 2

        distance = np.hypot(center_x - old_cx, center_y - old_cy)
        max_distance = max(w, h, ow, oh) * 1.5

        if distance < max_distance:
            score = calculate_iou(box, track["box"])
            score += 0.5 * (1.0 - distance / max_distance)

            if score > best_score:
                best_score = score
                best_id = face_id

    if best_id is None:
        best_id = next_face_id
        next_face_id += 1

        tracks[best_id] = {
            "box": box,
            "last_seen": frame_number,
            "scores": deque(maxlen=SMOOTHING_FRAMES),
            "last_logged": 0.0,
        }

    tracks[best_id]["box"] = box
    tracks[best_id]["last_seen"] = frame_number

    return best_id


# ---------------- CAMERA ----------------

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not camera.isOpened():
    camera.release()
    raise RuntimeError("Camera could not open. Try camera index 1.")

csv_file = open(csv_path, "w", newline="", encoding="utf-8")
csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
csv_writer.writeheader()
csv_file.flush()

print("CSV:", csv_path)
print("Webcam started. Press Q to quit.")

frame_number = 0
fps = 0.0

# Cached detections are reused between detection frames.
cached_faces = []

try:
    while True:
        frame_start = time.perf_counter()

        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        frame_number += 1
        frame = cv2.flip(frame, 1)

        if (
            frame.shape[1] != CAMERA_WIDTH
            or frame.shape[0] != CAMERA_HEIGHT
        ):
            frame = cv2.resize(
                frame,
                (CAMERA_WIDTH, CAMERA_HEIGHT)
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run face detection on the first frame and every Nth frame.
        if (
            frame_number == 1
            or frame_number % DETECTION_INTERVAL == 0
        ):
            detected_faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(35, 35),
            )

            cached_faces = [
                tuple(map(int, face))
                for face in detected_faces
            ]

        # Use the latest detected boxes on intermediate frames.
        faces = cached_faces

        # ---------------- PREPARE FACE BATCH ----------------

        face_items = []

        for (x, y, w, h) in faces:
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Ensure face coordinates are inside the image.
            x = max(0, x)
            y = max(0, y)
            w = min(w, gray.shape[1] - x)
            h = min(h, gray.shape[0] - y)

            if w <= 0 or h <= 0:
                continue

            box = (x, y, w, h)
            face_id = get_track_id(box, frame_number)

            face_gray = gray[y:y + h, x:x + w]
            face_gray = cv2.resize(face_gray, FACE_SIZE)

            face_input = face_gray.astype(np.float32) / 255.0

            face_items.append(
                (box, face_id, face_input.reshape(48, 48, 1))
            )

        # ---------------- BATCH PREDICTION ----------------

        batch_latency_ms = 0.0
        predictions = []

        if face_items:
            batch_input = np.stack(
                [item[2] for item in face_items],
                axis=0,
            )

            inference_start = time.perf_counter()

            batch_predictions = model.predict(
                batch_input,
                verbose=0,
            )

            batch_latency_ms = (
                time.perf_counter() - inference_start
            ) * 1000.0

            predictions = list(batch_predictions)

        now = time.time()

        # ---------------- DISPLAY AND LOG RESULTS ----------------

        for item, probabilities in zip(face_items, predictions):
            box, face_id, _ = item
            x, y, w, h = box

            probabilities = np.asarray(
                probabilities,
                dtype=np.float32,
            )

            if not np.all(np.isfinite(probabilities)):
                continue

            track = tracks[face_id]
            track["scores"].append(probabilities)

            smoothed = np.mean(
                np.stack(list(track["scores"])),
                axis=0,
            )

            best_index = int(np.argmax(smoothed))
            confidence = float(smoothed[best_index] * 100.0)

            emotion = (
                EMOTION_LABELS[best_index]
                if confidence >= CONFIDENCE_THRESHOLD
                else "Uncertain"
            )

            color = (
                (0, 200, 0)
                if emotion != "Uncertain"
                else (0, 165, 255)
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2,
            )

            label = (
                f"ID {face_id}: {emotion} "
                f"({confidence:.1f}%)"
            )

            cv2.putText(
                frame,
                label,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            if now - track["last_logged"] >= LOG_INTERVAL_SECONDS:
                row = {
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "face_id": face_id,
                    "expression": emotion,
                    "confidence_percent": round(confidence, 2),
                    "inference_latency_ms": round(
                        batch_latency_ms, 2
                    ),
                    "fps": round(fps, 2),
                    "face_x": x,
                    "face_y": y,
                    "face_w": w,
                    "face_h": h,
                }

                for index, name in enumerate(EMOTION_LABELS):
                    row[f"score_{name.lower()}"] = round(
                        float(smoothed[index] * 100.0),
                        2,
                    )

                csv_writer.writerow(row)
                track["last_logged"] = now

        # ---------------- FPS ----------------

        frame_elapsed = time.perf_counter() - frame_start

        if frame_elapsed > 0:
            current_fps = 1.0 / frame_elapsed

            fps = (
                current_fps
                if fps == 0
                else 0.9 * fps + 0.1 * current_fps
            )

        # ---------------- CLEAN STALE TRACKS ----------------

        stale_ids = [
            face_id
            for face_id, track in tracks.items()
            if frame_number - track["last_seen"] > TRACK_TIMEOUT_FRAMES
        ]

        for face_id in stale_ids:
            del tracks[face_id]

        # ---------------- OVERLAY ----------------

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Faces: {len(face_items)}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Facial Expression Recognition", frame)

        if frame_number % 30 == 0:
            csv_file.flush()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    csv_file.flush()
    csv_file.close()

print("Webcam session ended.")
print("CSV saved:", csv_path)


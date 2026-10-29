#python
from pathlib import Path
from collections import deque
from datetime import datetime
import csv
import time

import cv2
import numpy as np
import tensorflow as tf

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

print("Model loaded:", MODEL_PATH.name)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

if model.output_shape[-1] != len(EMOTION_LABELS):
    raise ValueError(
        "Model output count does not match the configured emotion labels."
    )

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

csv_file = open(csv_path, "w", newline="", encoding="utf-8")
csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
csv_writer.writeheader()
csv_file.flush()

print("CSV:", csv_path)

# ---------------- TRACKING ----------------

tracks = {}
next_face_id = 1


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
        old_center_x = ox + ow / 2
        old_center_y = oy + oh / 2

        distance = np.hypot(
            center_x - old_center_x,
            center_y - old_center_y
        )

        max_distance = max(w, h, ow, oh) * 1.5

        if distance < max_distance:
            iou = calculate_iou(box, track["box"])
            score = iou + 0.5 * (1.0 - distance / max_distance)

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

if not camera.isOpened():
    csv_file.close()
    raise RuntimeError(
        "Camera could not open. Check camera permissions or try index 1."
    )

frame_number = 0
previous_frame_time = time.perf_counter()
fps = 0.0

print("Webcam started. Press Q to quit.")

try:
    while True:
        frame_start = time.perf_counter()

        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        frame_number += 1
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(35, 35),
        )

        for (x, y, w, h) in faces:
            box = (int(x), int(y), int(w), int(h))
            face_id = get_track_id(box, frame_number)

            face_gray = gray[y:y + h, x:x + w]
            face_gray = cv2.resize(face_gray, FACE_SIZE)

            face_input = face_gray.astype(np.float32) / 255.0
            face_input = face_input.reshape(1, 48, 48, 1)

            # Measure model inference time for this face.
            inference_start = time.perf_counter()
            probabilities = model.predict(face_input, verbose=0)[0]
            inference_latency_ms = (
                time.perf_counter() - inference_start
            ) * 1000.0

            probabilities = np.asarray(
                probabilities, dtype=np.float32
            )

            if not np.all(np.isfinite(probabilities)):
                continue

            track = tracks[face_id]
            track["scores"].append(probabilities)

            smoothed = np.mean(
                np.stack(list(track["scores"])),
                axis=0
            )

            best_index = int(np.argmax(smoothed))
            confidence = float(smoothed[best_index] * 100.0)

            if confidence < CONFIDENCE_THRESHOLD:
                emotion = "Uncertain"
            else:
                emotion = EMOTION_LABELS[best_index]

            color = (
                (0, 200, 0)
                if emotion != "Uncertain"
                else (0, 165, 255)
            )

            cv2.rectangle(
                frame, (x, y), (x + w, y + h), color, 2
            )

            label = f"ID {face_id}: {emotion} ({confidence:.1f}%)"

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

            # Log each tracked face once per interval.
            now = time.time()

            if now - track["last_logged"] >= LOG_INTERVAL_SECONDS:
                row = {
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "face_id": face_id,
                    "expression": emotion,
                    "confidence_percent": round(confidence, 2),
                    "inference_latency_ms": round(
                        inference_latency_ms, 2
                    ),
                    "fps": "",
                    "face_x": x,
                    "face_y": y,
                    "face_w": w,
                    "face_h": h,
                }

                for index, name in enumerate(EMOTION_LABELS):
                    row[f"score_{name.lower()}"] = round(
                        float(smoothed[index] * 100.0), 2
                    )

                csv_writer.writerow(row)
                csv_file.flush()
                track["last_logged"] = now

        # FPS is based on the whole frame-processing loop.
        frame_elapsed = time.perf_counter() - frame_start
        if frame_elapsed > 0:
            current_fps = 1.0 / frame_elapsed
            fps = current_fps if fps == 0 else (
                0.9 * fps + 0.1 * current_fps
            )

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
            f"Faces: {len(faces)}",
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

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    csv_file.close()

print("Webcam session ended.")
print("CSV saved:", csv_path)

#python
from pathlib import Path
import json
import time

import cv2
import numpy as np
import tensorflow as tf

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "facial_expression_model_v3.keras"
LABELS_PATH = BASE_DIR / "models" / "emotion_labels_v3.json"
CASCADE_PATH = (
    Path(cv2.__file__).resolve().parent
    / "data"
    / "haarcascade_frontalface_default.xml"
)

DEBUG_DIR = BASE_DIR / "debug_faces"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load emotion labels
# --------------------------------------------------
with open(LABELS_PATH, "r", encoding="utf-8") as file:
    EMOTION_LABELS = json.load(file)

# --------------------------------------------------
# 3. Load model
# --------------------------------------------------
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

if model.output_shape[-1] != len(EMOTION_LABELS):
    raise ValueError(
        "Model output and emotion labels do not match."
    )

print("Model loaded successfully.")
print("Emotion labels:", EMOTION_LABELS)
print("Model input shape:", model.input_shape)

# --------------------------------------------------
# 4. Load face detector
# --------------------------------------------------
face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))

if face_detector.empty():
    raise RuntimeError(
        f"Could not load face detector: {CASCADE_PATH}"
    )

print("Face detector loaded successfully.")

# --------------------------------------------------
# 5. Start webcam
# --------------------------------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera permissions "
        "or try another camera index."
    )

previous_time = time.perf_counter()
fps = 0.0

# Save a debug face only once every few seconds
last_save_time = 0.0
save_interval = 3.0

print("Version 3 webcam started. Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            # Extract face safely
            face_crop = gray[y:y + h, x:x + w]

            if face_crop.size == 0:
                continue

            # --------------------------------------------------
            # 6. Preprocess: grayscale, 48x48, normalize
            # --------------------------------------------------
            face_resized = cv2.resize(
                face_crop,
                (48, 48),
                interpolation=cv2.INTER_AREA
            )

            face_input = face_resized.astype(
                np.float32
            ) / 255.0

            # Shape: (1, 48, 48, 1)
            face_input = np.expand_dims(
                face_input,
                axis=(0, -1)
            )

            # Save the exact resized face for inspection
            current_time = time.perf_counter()

            if current_time - last_save_time >= save_interval:
                cv2.imwrite(
                    str(DEBUG_DIR / "latest_face.png"),
                    face_resized
                )
                last_save_time = current_time

            # --------------------------------------------------
            # 7. Predict emotion
            # --------------------------------------------------
            probabilities = model.predict(
                face_input,
                verbose=0
            )[0]

            best_index = int(np.argmax(probabilities))
            confidence = float(probabilities[best_index])

            predicted_emotion = EMOTION_LABELS[best_index]

            if confidence < 0.25:
                emotion = "Uncertain"
            else:
                emotion = predicted_emotion

            # Print probabilities for debugging
            print("\n--- Face Prediction ---")

            for name, score in zip(
                EMOTION_LABELS,
                probabilities
            ):
                print(f"{name:10s}: {score * 100:5.1f}%")

            print(
                f"Predicted: {emotion} "
                f"(confidence: {confidence * 100:.1f}%)"
            )

            # --------------------------------------------------
            # 8. Draw rectangle and prediction
            # --------------------------------------------------
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            display_text = (
                f"{emotion} ({confidence * 100:.1f}%)"
            )

            cv2.putText(
                frame,
                display_text,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        # --------------------------------------------------
        # 9. Calculate FPS
        # --------------------------------------------------
        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        previous_time = current_time

        if elapsed > 0:
            instant_fps = 1.0 / elapsed

            if fps == 0:
                fps = instant_fps
            else:
                fps = 0.9 * fps + 0.1 * instant_fps

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Main webcam window
        cv2.imshow(
            "Facial Expression Recognition - Version 3",
            frame
        )

        # Debug window: exact 48x48 model input, enlarged for viewing
        if len(faces) > 0:
            debug_view = cv2.resize(
                face_resized,
                (192, 192),
                interpolation=cv2.INTER_NEAREST
            )
            cv2.imshow("Debug Face Input (48x48 enlarged)", debug_view)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

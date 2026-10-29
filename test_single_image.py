
from pathlib import Path
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "dataset" / "feb2013" / "test"
MODEL_PATH = BASE_DIR / "models" / "facial_expression_model_v3.keras"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

class_names = sorted(
    folder.name for folder in TEST_DIR.iterdir()
    if folder.is_dir()
)

print("Classes:", class_names)

for emotion in ["happy", "surprise", "sad"]:
    folder = TEST_DIR / emotion
    images = list(folder.glob("*.png"))

    if not images:
        images = list(folder.glob("*.jpg"))

    if not images:
        print(f"No image found for {emotion}")
        continue

    image_path = images[0]

    image = tf.keras.utils.load_img(
        image_path,
        color_mode="grayscale",
        target_size=(48, 48)
    )

    array = tf.keras.utils.img_to_array(image)
    array = array.astype("float32") / 255.0
    array = np.expand_dims(array, axis=0)

    probabilities = model.predict(array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    print(f"\nActual label: {emotion}")
    print(f"Image: {image_path.name}")
    print(f"Predicted: {class_names[predicted_index]}")

    for name, score in zip(class_names, probabilities):
        print(f"{name:10s}: {score * 100:5.1f}%")

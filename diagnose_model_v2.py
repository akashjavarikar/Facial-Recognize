
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "dataset" / "feb2013" / "test"
MODEL_PATH = BASE_DIR / "models" / "facial_expression_model_v2.keras"

print("Loading Version 2 model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    label_mode="int",
    shuffle=False
)

class_names = test_ds.class_names
true_counts = Counter()
pred_counts = Counter()
confidence_values = []
correct = 0
total = 0

for images, labels in test_ds:
    images = tf.cast(images, tf.float32) / 255.0
    probabilities = model.predict(images, verbose=0)

    predictions = np.argmax(probabilities, axis=1)
    actual = labels.numpy()

    true_counts.update(actual.tolist())
    pred_counts.update(predictions.tolist())

    confidence_values.extend(
        np.max(probabilities, axis=1).tolist()
    )

    correct += int(np.sum(predictions == actual))
    total += len(actual)

print("\nModel output shape:", model.output_shape)
print(f"Test accuracy: {100 * correct / total:.2f}%")
print(
    "Average prediction confidence: "
    f"{100 * np.mean(confidence_values):.2f}%"
)

print("\nCLASS DISTRIBUTION")
print("-" * 55)
print(f"{'Expression':<15}{'Actual':>12}{'Predicted':>15}")

for i, name in enumerate(class_names):
    print(
        f"{name:<15}"
        f"{true_counts[i]:>12}"
        f"{pred_counts[i]:>15}"
    )

print("\nDiagnostic completed.")

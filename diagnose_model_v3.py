
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "dataset" / "feb2013" / "test"
MODEL_PATH = BASE_DIR / "models" / "facial_expression_model_v3.keras"

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
y_true = []
y_pred = []

for images, labels in test_ds:
    images = tf.cast(images, tf.float32) / 255.0
    probabilities = model.predict(images, verbose=0)

    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(probabilities, axis=1).tolist())

print("\nClass names:", class_names)
print("Model output shape:", model.output_shape)

print("\nACTUAL TEST LABEL COUNTS")
for i, name in enumerate(class_names):
    print(f"{name}: {sum(y == i for y in y_true)}")

print("\nPREDICTED LABEL COUNTS")
for i, name in enumerate(class_names):
    print(f"{name}: {sum(y == i for y in y_pred)}")

print("\nTEST CLASSIFICATION REPORT")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

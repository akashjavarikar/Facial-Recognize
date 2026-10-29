
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "dataset" / "feb2013" / "test"
MODEL_PATH = BASE_DIR / "models" / "facial_expression_model_v2.keras"
OUTPUT_DIR = BASE_DIR / "models"

IMG_SIZE = (48, 48)
BATCH_SIZE = 64

print("Loading Version 2 model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = test_ds.class_names
print("\nClass order:", class_names)

y_true = []
y_pred = []

for images, labels in test_ds:
    images = tf.cast(images, tf.float32) / 255.0
    probabilities = model.predict(images, verbose=0)

    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(probabilities, axis=1).tolist())

print("\nVersion 2 Test Accuracy: "
      f"{np.mean(np.array(y_true) == np.array(y_pred)) * 100:.2f}%")

print("Version 2 Test Macro F1: "
      f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Version 2 - Test Confusion Matrix")
plt.tight_layout()

output_path = OUTPUT_DIR / "confusion_matrix_v2.png"
plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nConfusion matrix saved: {output_path}")

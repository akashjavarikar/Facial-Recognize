
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "dataset" / "feb2013" / "test"
MODEL_PATH = BASE_DIR / "models" / "best_facial_expression_model.keras"

IMG_SIZE = (48, 48)
BATCH_SIZE = 64

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = test_ds.class_names

# Normalize pixel values
test_ds = test_ds.map(
    lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
)

# Evaluate model
loss, accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Collect actual and predicted labels
y_true = np.concatenate([
    np.argmax(labels.numpy(), axis=1)
    for _, labels in test_ds
])

predictions = model.predict(test_ds, verbose=1)
y_pred = np.argmax(predictions, axis=1)

# Classification report
print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    labels=range(len(class_names)),
    target_names=class_names,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names))
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Facial Expression Recognition - Confusion Matrix")
plt.tight_layout()

output_path = BASE_DIR / "models" / "confusion_matrix.png"
plt.savefig(output_path, dpi=150)
plt.show()

print(f"\nConfusion matrix saved to: {output_path}")

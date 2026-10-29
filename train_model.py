
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# -----------------------------
# 1. Project paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "feb2013" / "train"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "facial_expression_model_v2.keras"
LABELS_PATH = MODEL_DIR / "emotion_labels_v2.json"

# -----------------------------
# 2. Configuration
# -----------------------------
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 40
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# -----------------------------
# 3. Load datasets
# -----------------------------
train_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    shuffle=True
)

val_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    shuffle=False
)

class_names = train_raw.class_names
num_classes = len(class_names)

print("\nEmotion class order:", class_names)

# Save exact label mapping for future inference
with open(LABELS_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2)

# -----------------------------
# 4. Class weights
# -----------------------------
class_counts = np.zeros(num_classes, dtype=np.int64)

for _, labels in train_raw:
    class_counts += np.bincount(
        labels.numpy(), minlength=num_classes
    )

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=np.repeat(
        np.arange(num_classes),
        class_counts
    )
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(weights)
}

print("\nTraining class counts:")
for name, count in zip(class_names, class_counts):
    print(f"{name}: {count}")

print("\nClass weights:", class_weights)

# -----------------------------
# 5. Preprocessing and augmentation
# -----------------------------
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.10),
], name="data_augmentation")

AUTOTUNE = tf.data.AUTOTUNE

def prepare_train(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    images = augmentation(images, training=True)
    labels = tf.one_hot(labels, num_classes)
    return images, labels

def prepare_val(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    labels = tf.one_hot(labels, num_classes)
    return images, labels

train_ds = train_raw.map(
    prepare_train, num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

val_ds = val_raw.map(
    prepare_val, num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# -----------------------------
# 6. Build CNN model
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),

    layers.Conv2D(32, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.45),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 7. Best checkpoint and callbacks
# -----------------------------
checkpoint = callbacks.ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

print("\nStarting Version 2 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

print("\nTraining finished.")
print("Best Version 2 model:", MODEL_PATH)
print("Emotion labels:", LABELS_PATH)
print("Original model files have not been overwritten.")

# -----------------------------
# 8. Validation macro F1
# -----------------------------
best_model = tf.keras.models.load_model(MODEL_PATH)

y_true = []
y_pred = []

for images, labels in val_raw:
    images = tf.cast(images, tf.float32) / 255.0
    probabilities = best_model.predict(images, verbose=0)

    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(probabilities, axis=1).tolist())

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

print(f"\nValidation Macro F1: {macro_f1:.4f}")
print("\nVersion 2 training complete.")
print("Note: Final performance must be measured on the untouched test set.")

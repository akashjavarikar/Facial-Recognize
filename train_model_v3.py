
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import f1_score, classification_report

# 1. Paths
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "feb2013" / "train"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "facial_expression_model_v3.keras"
LABELS_PATH = MODEL_DIR / "emotion_labels_v3.json"

# 2. Configuration
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 40
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# 3. Load training and validation data
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

print("\nClass order:", class_names)

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

# 4. Mild augmentation; no class weights in this experiment
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.05),
], name="mild_augmentation")

AUTOTUNE = tf.data.AUTOTUNE

def prepare_train(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    images = augmentation(images, training=True)
    return images, labels

def prepare_val(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

train_ds = train_raw.map(
    prepare_train,
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

val_ds = val_raw.map(
    prepare_val,
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# 5. CNN model
model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),

    layers.Conv2D(32, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(2),
    layers.Dropout(0.20),

    layers.Conv2D(64, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),

    layers.Conv2D(128, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(2),
    layers.Dropout(0.30),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.40),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Callbacks
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

# 7. Train without class weighting
print("\nStarting Version 3 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

print("\nTraining finished.")
print("Best model:", MODEL_PATH)
print("Labels:", LABELS_PATH)

# 8. Evaluate macro F1 on validation data
best_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

y_true = []
y_pred = []

for images, labels in val_raw:
    images = tf.cast(images, tf.float32) / 255.0
    probabilities = best_model.predict(images, verbose=0)

    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(probabilities, axis=1).tolist())

print("\nValidation Macro F1:",
      round(f1_score(
          y_true, y_pred,
          average="macro",
          zero_division=0
      ), 4))

print("\nValidation classification report:\n")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0
))

print("\nVersion 3 complete.")
print("The original models have not been overwritten.")

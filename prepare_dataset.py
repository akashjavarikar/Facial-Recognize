
from pathlib import Path
import numpy as np
import tensorflow as tf

# Dataset paths
train_dir = Path("dataset/feb2013/train")
test_dir = Path("dataset/feb2013/test")

# Image settings
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 42

# Load training data and create validation split
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    label_mode="categorical"
)

# Load test data separately
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="categorical"
)

# Normalize pixel values from 0-255 to 0-1
def normalize_images(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

train_ds = train_ds.map(normalize_images)
val_ds = val_ds.map(normalize_images)
test_ds = test_ds.map(normalize_images)

# Display dataset details
print("\nClasses:", train_ds.class_names if hasattr(train_ds, "class_names") else
      ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"])

for images, labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Pixel range:", float(tf.reduce_min(images)),
          "to", float(tf.reduce_max(images)))

print("\nDataset preprocessing completed successfully!")

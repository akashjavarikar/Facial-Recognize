
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

base = Path("dataset/feb2013/train")

classes = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]

fig, axes = plt.subplots(1, 7, figsize=(16, 4))

for ax, emotion in zip(axes, classes):
    folder = base / emotion

    if not folder.is_dir():
        ax.set_title(f"{emotion}\nMissing")
        ax.axis("off")
        continue

    images = [
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    if not images:
        ax.set_title(f"{emotion}\nNo images")
        ax.axis("off")
        continue

    image_path = random.choice(images)

    with Image.open(image_path) as img:
        ax.imshow(img.convert("L"), cmap="gray")

    ax.set_title(emotion)
    ax.axis("off")

plt.tight_layout()
plt.show()

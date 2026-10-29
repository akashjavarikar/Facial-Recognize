
from pathlib import Path
from PIL import Image

base = Path("dataset/feb2013/train")
classes = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]

for emotion in classes:
    folder = base / emotion

    if not folder.is_dir():
        print(f"{emotion}: folder missing")
        continue

    images = [
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    if not images:
        print(f"{emotion}: no images")
        continue

    with Image.open(images[0]) as img:
        print(
            f"{emotion}: size={img.size}, "
            f"mode={img.mode}"
        )

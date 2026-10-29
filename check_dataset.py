
from pathlib import Path

base = Path("dataset/feb2013")

classes = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]

for split in ["train", "test"]:
    folder = base / split
    print(f"\n--- {split.upper()} DATASET ---")

    if not folder.is_dir():
        print(f"Folder not found: {folder}")
        continue

    for name in classes:
        class_folder = folder / name

        if not class_folder.is_dir():
            print(f"{name}: Folder missing")
            continue

        count = sum(
            1 for p in class_folder.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        )

        print(f"{name}: {count} images")

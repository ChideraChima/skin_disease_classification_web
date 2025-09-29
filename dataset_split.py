import os
import shutil
import random
from pathlib import Path


def create_deterministic_split(
    source_dir: str = "skin_disease_dataset",
    target_dir: str = "dataset",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    random.seed(seed)

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    for split in ["train", "val", "test"]:
        for cls in classes:
            Path(os.path.join(target_dir, split, cls)).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        images.sort()

        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            for fname in split_files:
                src = os.path.join(cls_path, fname)
                dst = os.path.join(target_dir, split_name, cls, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)


if __name__ == "__main__":
    create_deterministic_split()



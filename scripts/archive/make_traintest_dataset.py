"""
Create Dataset-TrainTest: combine train + test splits for training.

Rationale: We have 624 labeled test images that we're not using during training.
The val set (604 images) remains held out for evaluation.
This gives us 3388 training images vs 2764 (22% more data).

This is a data pipeline change — using all available labeled data for training.
"""

import os
import shutil
from pathlib import Path

SRC_DIR = Path("/workspace/autoresearch/Dataset-YOLO")
DST_DIR = Path("/workspace/autoresearch/Dataset-TrainTest")


def create_traintest_dataset():
    """Create Dataset-TrainTest with train+test as training data."""
    print("Creating Dataset-TrainTest (train + test combined for training)...")

    # Create directory structure
    for subdir in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        (DST_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Combine train + test → new train split (symlinks)
    n_imgs = 0
    n_lbls = 0
    for src_split in ['train', 'test']:
        src_img_dir = SRC_DIR / 'images' / src_split
        src_lbl_dir = SRC_DIR / 'labels' / src_split

        for img_path in sorted(src_img_dir.glob('*')):
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                dst = DST_DIR / 'images' / 'train' / img_path.name
                if not dst.exists():
                    os.symlink(str(img_path.resolve()), str(dst))
                n_imgs += 1

        for lbl_path in sorted(src_lbl_dir.glob('*.txt')):
            dst = DST_DIR / 'labels' / 'train' / lbl_path.name
            if not dst.exists():
                os.symlink(str(lbl_path.resolve()), str(dst))
            n_lbls += 1

    print(f"  train: {n_imgs} images, {n_lbls} labels (from train+test splits)")

    # Val split → new val (symlinks, unchanged)
    n_val_imgs = 0
    n_val_lbls = 0
    for img_path in sorted((SRC_DIR / 'images' / 'val').glob('*')):
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            dst = DST_DIR / 'images' / 'val' / img_path.name
            if not dst.exists():
                os.symlink(str(img_path.resolve()), str(dst))
            n_val_imgs += 1

    for lbl_path in sorted((SRC_DIR / 'labels' / 'val').glob('*.txt')):
        dst = DST_DIR / 'labels' / 'val' / lbl_path.name
        if not dst.exists():
            os.symlink(str(lbl_path.resolve()), str(dst))
        n_val_lbls += 1

    print(f"  val:   {n_val_imgs} images, {n_val_lbls} labels (held out for eval)")

    # Count class instances in merged train
    train_lbl_dir = DST_DIR / 'labels' / 'train'
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for lbl_path in train_lbl_dir.glob('*.txt'):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
    print(f"\n  Train class counts: B1={counts[0]}, B2={counts[1]}, B3={counts[2]}, B4={counts[3]}")

    # Write data.yaml
    yaml_content = f"""path: {DST_DIR}
train: images/train
val: images/val

nc: 4
names:
  - B1
  - B2
  - B3
  - B4
"""
    with open(DST_DIR / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    print(f"\n  data.yaml written: {DST_DIR / 'data.yaml'}")
    print("Done.")


if __name__ == '__main__':
    if DST_DIR.exists():
        print(f"Removing existing {DST_DIR}...")
        shutil.rmtree(DST_DIR)
    create_traintest_dataset()

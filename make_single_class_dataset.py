"""
Create a single-class dataset for two-stage pipeline.

Stage 1 of two-stage pipeline:
- All 4 classes (B1, B2, B3, B4) → class 0 "TBS" (oil palm bunch)
- Train a high-accuracy single-class detector
- Expected: mAP50-95 >> 0.389 (from project history)

Stage 2 (classifier) will be built separately.
"""

import os
import shutil
from pathlib import Path

SRC = Path("/workspace/autoresearch/Dataset-YOLO")
DST = Path("/workspace/autoresearch/Dataset-SingleClass")


def read_label(label_path):
    """Read YOLO label file."""
    boxes = []
    if Path(label_path).exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4])))
    return boxes


def write_label_single_class(label_path, boxes):
    """Write YOLO label with all classes → 0."""
    with open(label_path, 'w') as f:
        for box in boxes:
            f.write(f"0 {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def setup_dirs():
    """Create output directory structure."""
    for split in ['train', 'val', 'test']:
        (DST / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DST / 'labels' / split).mkdir(parents=True, exist_ok=True)


def process_split(split):
    """Copy images and remap all class IDs to 0."""
    src_img = SRC / 'images' / split
    src_lbl = SRC / 'labels' / split
    dst_img = DST / 'images' / split
    dst_lbl = DST / 'labels' / split

    n = 0
    for img_path in sorted(src_img.glob('*')):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue

        shutil.copy2(img_path, dst_img / img_path.name)

        lbl_path = src_lbl / (img_path.stem + '.txt')
        boxes = read_label(lbl_path)
        write_label_single_class(dst_lbl / (img_path.stem + '.txt'), boxes)
        n += 1

    print(f"  {split}: {n} images processed")
    return n


def create_data_yaml():
    """Create data.yaml for single-class dataset."""
    yaml_content = """path: /workspace/autoresearch/Dataset-SingleClass
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: TBS
"""
    with open(DST / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    print(f"  Created {DST / 'data.yaml'}")


if __name__ == '__main__':
    print("Creating single-class dataset (all → TBS)...")
    setup_dirs()

    for split in ['train', 'val', 'test']:
        process_split(split)

    create_data_yaml()
    print(f"\nDone! Dataset saved to {DST}")
    print("Next: train YOLOv9c single-class, then build classifier on crops.")

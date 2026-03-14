"""
Create a class-balanced training dataset by oversampling images containing B1 and B4.

B3 dominates training (5634 instances) vs B1=1540, B4=2343.
Strategy: copy all original images, then add 2x copies of images with B1 or B4
using horizontal/vertical flip augmentations for diversity.

Output: Dataset-Balanced/ with same val/test as Dataset-YOLO/
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path

SRC = Path("/workspace/autoresearch/Dataset-YOLO")
DST = Path("/workspace/autoresearch/Dataset-Balanced")

# Class indices
B1_IDX = 0
B2_IDX = 1
B3_IDX = 2
B4_IDX = 3


def read_label(label_path):
    """Read YOLO label file, return list of (class_id, cx, cy, w, h)."""
    boxes = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4])))
    return boxes


def write_label(label_path, boxes):
    """Write YOLO label file."""
    with open(label_path, 'w') as f:
        for box in boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def flip_boxes_horizontal(boxes, img_w=None):
    """Flip boxes horizontally: cx_new = 1 - cx_old."""
    return [(cls_id, 1.0 - cx, cy, w, h) for cls_id, cx, cy, w, h in boxes]


def flip_boxes_vertical(boxes, img_h=None):
    """Flip boxes vertically: cy_new = 1 - cy_old."""
    return [(cls_id, cx, 1.0 - cy, w, h) for cls_id, cx, cy, w, h in boxes]


def contains_class(boxes, class_ids):
    return any(b[0] in class_ids for b in boxes)


def count_class_instances(boxes, class_id):
    return sum(1 for b in boxes if b[0] == class_id)


def setup_dirs():
    """Create output directory structure."""
    for split in ['train', 'val', 'test']:
        (DST / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DST / 'labels' / split).mkdir(parents=True, exist_ok=True)


def copy_split(split):
    """Copy val and test splits unchanged."""
    src_img = SRC / 'images' / split
    src_lbl = SRC / 'labels' / split
    dst_img = DST / 'images' / split
    dst_lbl = DST / 'labels' / split

    for img_path in src_img.glob('*'):
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            shutil.copy2(img_path, dst_img / img_path.name)
            lbl_path = src_lbl / (img_path.stem + '.txt')
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl / lbl_path.name)

    n = len(list(dst_img.glob('*')))
    print(f"  {split}: copied {n} images")


def process_train():
    """Copy train with oversampling for B1 and B4 images."""
    src_img = SRC / 'images' / 'train'
    src_lbl = SRC / 'labels' / 'train'
    dst_img = DST / 'images' / 'train'
    dst_lbl = DST / 'labels' / 'train'

    total = 0
    extra = 0

    # Stats
    stats = {'B1_images': 0, 'B4_images': 0, 'total_images': 0, 'original_B1': 0,
             'original_B2': 0, 'original_B3': 0, 'original_B4': 0}

    all_img_paths = sorted(src_img.glob('*'))

    for img_path in all_img_paths:
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue

        lbl_path = src_lbl / (img_path.stem + '.txt')
        boxes = read_label(lbl_path)

        # Count class instances
        for b in boxes:
            if b[0] == B1_IDX: stats['original_B1'] += 1
            elif b[0] == B2_IDX: stats['original_B2'] += 1
            elif b[0] == B3_IDX: stats['original_B3'] += 1
            elif b[0] == B4_IDX: stats['original_B4'] += 1

        # Always copy the original
        dst_img_path = dst_img / img_path.name
        shutil.copy2(img_path, dst_img_path)
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl / lbl_path.name)
        total += 1

        # Oversample images containing B1 or B4 (minority classes)
        has_b1 = contains_class(boxes, [B1_IDX])
        has_b4 = contains_class(boxes, [B4_IDX])

        if has_b1:
            stats['B1_images'] += 1
        if has_b4:
            stats['B4_images'] += 1

        if has_b1 or has_b4:
            # Read image for augmentation
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Copy 1: horizontal flip
            img_hflip = cv2.flip(img, 1)
            hflip_name = img_path.stem + '_hflip' + img_path.suffix
            cv2.imwrite(str(dst_img / hflip_name), img_hflip)
            flipped_boxes = flip_boxes_horizontal(boxes)
            write_label(dst_lbl / (img_path.stem + '_hflip.txt'), flipped_boxes)
            extra += 1

            # Copy 2: vertical flip (only if has B4, to double B4 even more)
            if has_b4:
                img_vflip = cv2.flip(img, 0)
                vflip_name = img_path.stem + '_vflip' + img_path.suffix
                cv2.imwrite(str(dst_img / vflip_name), img_vflip)
                flipped_v_boxes = flip_boxes_vertical(boxes)
                write_label(dst_lbl / (img_path.stem + '_vflip.txt'), flipped_v_boxes)
                extra += 1

    stats['total_images'] = total
    print(f"  train: {total} original + {extra} augmented copies = {total + extra} total")
    print(f"  Images with B1: {stats['B1_images']}, with B4: {stats['B4_images']}")
    print(f"  Original class distribution: B1={stats['original_B1']}, B2={stats['original_B2']}, B3={stats['original_B3']}, B4={stats['original_B4']}")


def create_data_yaml():
    """Create data.yaml for balanced dataset."""
    yaml_content = """path: /workspace/autoresearch/Dataset-Balanced
train: images/train
val: /workspace/autoresearch/Dataset-YOLO/images/val
test: /workspace/autoresearch/Dataset-YOLO/images/test

nc: 4
names:
  0: B1
  1: B2
  2: B3
  3: B4
"""
    with open(DST / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    print(f"  Created {DST / 'data.yaml'}")


def verify_balanced():
    """Count instances in balanced train set."""
    lbl_dir = DST / 'labels' / 'train'
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    n_files = 0
    for lbl_file in lbl_dir.glob('*.txt'):
        boxes = read_label(lbl_file)
        for b in boxes:
            if b[0] in counts:
                counts[b[0]] += 1
        if boxes:
            n_files += 1
    print(f"  Balanced train label files: {len(list(lbl_dir.glob('*.txt')))}")
    print(f"  Balanced class distribution: B1={counts[0]}, B2={counts[1]}, B3={counts[2]}, B4={counts[3]}")
    return counts


if __name__ == '__main__':
    print("Setting up directories...")
    setup_dirs()

    print("Copying val and test splits unchanged...")
    copy_split('val')
    copy_split('test')

    print("Processing train with B1/B4 oversampling...")
    process_train()

    print("Creating data.yaml...")
    create_data_yaml()

    print("Verifying balanced dataset...")
    counts = verify_balanced()
    print(f"\nDone! Dataset saved to {DST}")

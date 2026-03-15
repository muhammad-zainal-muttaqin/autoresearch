"""
Extract crops from training images using GT bounding boxes.
Used for two-stage pipeline stage 2 (classifier).

Creates:
- Dataset-Crops/train/B1/*.jpg  (B2, B3, B4 same)
- Dataset-Crops/val/B1/*.jpg
- Dataset-Crops/test/*.jpg (for inference)
"""

import os
import shutil
import math
from pathlib import Path
from PIL import Image

SRC = Path("/workspace/autoresearch/Dataset-YOLO")
DST = Path("/workspace/autoresearch/Dataset-Crops")

CLASS_NAMES = {0: "B1", 1: "B2", 2: "B3", 3: "B4"}
PAD_RATIO = 0.2  # Add 20% padding around each crop for context


def read_label(label_path):
    """Read YOLO label, return list of (class_id, cx, cy, w, h)."""
    boxes = []
    if Path(label_path).exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4])))
    return boxes


def setup_dirs():
    """Create output directory structure."""
    for split in ['train', 'val']:
        for cls_name in CLASS_NAMES.values():
            (DST / split / cls_name).mkdir(parents=True, exist_ok=True)


def extract_crops(split):
    """Extract all GT crops from a split."""
    src_img = SRC / 'images' / split
    src_lbl = SRC / 'labels' / split

    counts = {cls_id: 0 for cls_id in CLASS_NAMES}
    n_imgs_processed = 0

    for img_path in sorted(src_img.glob('*')):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue

        lbl_path = src_lbl / (img_path.stem + '.txt')
        boxes = read_label(lbl_path)

        if not boxes:
            continue

        img = Image.open(str(img_path))
        img_w, img_h = img.size

        for i, (cls_id, cx_norm, cy_norm, w_norm, h_norm) in enumerate(boxes):
            if cls_id not in CLASS_NAMES:
                continue

            # Convert to pixel coords
            cx = cx_norm * img_w
            cy = cy_norm * img_h
            bw = w_norm * img_w
            bh = h_norm * img_h

            # Add padding
            pad_w = bw * PAD_RATIO
            pad_h = bh * PAD_RATIO

            x1 = max(0, int(cx - bw/2 - pad_w))
            y1 = max(0, int(cy - bh/2 - pad_h))
            x2 = min(img_w, int(cx + bw/2 + pad_w))
            y2 = min(img_h, int(cy + bh/2 + pad_h))

            # Skip degenerate crops
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            crop = img.crop((x1, y1, x2, y2))

            # Resize to 224x224 for classifier
            crop = crop.resize((224, 224), Image.LANCZOS)

            cls_name = CLASS_NAMES[cls_id]
            out_path = DST / split / cls_name / f"{img_path.stem}_box{i}.jpg"
            crop.save(str(out_path), quality=90)
            counts[cls_id] += 1

        n_imgs_processed += 1

    print(f"  {split}: {n_imgs_processed} images, crops: B1={counts[0]}, B2={counts[1]}, B3={counts[2]}, B4={counts[3]}")
    return counts


if __name__ == '__main__':
    print("Setting up directories...")
    setup_dirs()

    print("Extracting crops from train set...")
    train_counts = extract_crops('train')

    print("Extracting crops from val set...")
    val_counts = extract_crops('val')

    print(f"\nDone! Crops saved to {DST}")
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    print(f"Total crops: train={total_train}, val={total_val}")

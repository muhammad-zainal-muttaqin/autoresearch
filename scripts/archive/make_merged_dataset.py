"""
Create Dataset-Merged3: merge B2 (class 1) and B3 (class 2) into a single class.

Original classes: B1=0, B2=1, B3=2, B4=3
Merged classes:   B1=0, B23=1, B4=2

Hypothesis: By removing the ambiguous B2/B3 boundary, the 3-class detector
should have much better precision/recall on B23, and the model's features
will be more discriminative for B1 vs B4. We then apply a binary B2/B3
classifier to split B23 detections into B2 and B3.

This is a data pipeline change (not just augmentation).
"""

import os
import shutil
from pathlib import Path

SRC_DIR = Path("/workspace/autoresearch/Dataset-YOLO")
DST_DIR = Path("/workspace/autoresearch/Dataset-Merged3")

# Original: B1=0, B2=1, B3=2, B4=3
# New:      B1=0, B23=1, B4=2
CLASS_MAP = {0: 0, 1: 1, 2: 1, 3: 2}  # merge B2+B3 → B23
NEW_CLASSES = ["B1", "B23", "B4"]


def convert_label_file(src_path, dst_path):
    """Convert a YOLO label file to use merged class IDs."""
    lines_out = []
    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                new_cls = CLASS_MAP[cls_id]
                lines_out.append(f"{new_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
    with open(dst_path, 'w') as f:
        f.write('\n'.join(lines_out))
        if lines_out:
            f.write('\n')


def create_merged_dataset():
    """Create Dataset-Merged3 with symlinked images and converted labels."""
    print("Creating Dataset-Merged3 (B2+B3 merged into B23)...")

    for split in ['train', 'val', 'test']:
        src_img_dir = SRC_DIR / 'images' / split
        src_lbl_dir = SRC_DIR / 'labels' / split

        if not src_img_dir.exists():
            continue

        dst_img_dir = DST_DIR / 'images' / split
        dst_lbl_dir = DST_DIR / 'labels' / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Symlink images
        n_imgs = 0
        for img_path in sorted(src_img_dir.glob('*')):
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                dst = dst_img_dir / img_path.name
                if not dst.exists():
                    os.symlink(str(img_path.resolve()), str(dst))
                n_imgs += 1

        # Convert labels
        n_lbls = 0
        for lbl_path in sorted(src_lbl_dir.glob('*.txt')):
            dst = dst_lbl_dir / lbl_path.name
            convert_label_file(lbl_path, dst)
            n_lbls += 1

        print(f"  {split}: {n_imgs} images, {n_lbls} labels")

    # Count class instances in train
    train_lbl_dir = DST_DIR / 'labels' / 'train'
    counts = {0: 0, 1: 0, 2: 0}
    for lbl_path in train_lbl_dir.glob('*.txt'):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
    print(f"\n  Train class counts: B1={counts[0]}, B23={counts[1]}, B4={counts[2]}")
    print(f"  (Original B2+B3 merged. B23 is the dominant class.)")

    # Create data.yaml
    yaml_content = f"""path: {DST_DIR}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  - B1
  - B23
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
    create_merged_dataset()

"""
Tiled Training Dataset for Small Object Detection (B4).

DIFFERENT from SAHI inference. This creates a new training dataset by cutting
each training image into 640x640 tiles with 25% overlap (stride=480).
- Training is done on tiles, so B4 objects appear proportionally larger
- Val/test sets are NOT tiled (evaluation on full-res images)

Hypothesis: B4 mAP50-95 = 0.140 (very poor). B4 objects are small relative to
full image. At 640px training resolution, B4 might be only 20-30px wide.
On tiles, the same B4 object appears as 40-80px width → easier to detect.

This is fundamentally different from augmentation (scale=0.7, etc.) because:
1. The actual image content is at true full resolution
2. Context is preserved within each tile
3. Multiple tiles provide different contexts for edge objects
"""

import os
import shutil
import math
from pathlib import Path
from PIL import Image

SRC = Path("/workspace/autoresearch/Dataset-YOLO")
DST = Path("/workspace/autoresearch/Dataset-Tiled")

TILE_SIZE = 640
STRIDE = 480  # 25% overlap (640 - 480 = 160 overlap)
MIN_LABELS_PER_TILE = 1  # Drop empty tiles to save space
MAX_BACKGROUND_TILES = 2  # Keep some background tiles per image for negative samples


def read_label(label_path):
    """Read YOLO label, return list of (class_id, cx, cy, w, h) all normalized [0,1]."""
    boxes = []
    if Path(label_path).exists():
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


def get_tile_labels(boxes, img_w, img_h, tile_x, tile_y, tile_size):
    """
    Get labels that fall within a tile. Convert to tile-relative coordinates.

    Args:
        boxes: list of (cls, cx, cy, w, h) normalized to full image
        img_w, img_h: full image dimensions
        tile_x, tile_y: top-left corner of tile in pixels
        tile_size: size of tile in pixels

    Returns:
        list of (cls, cx, cy, w, h) normalized to tile
    """
    tile_labels = []

    for cls, cx_norm, cy_norm, w_norm, h_norm in boxes:
        # Convert to pixel coordinates
        cx_px = cx_norm * img_w
        cy_px = cy_norm * img_h
        w_px = w_norm * img_w
        h_px = h_norm * img_h

        # Box boundaries in pixels
        x_min = cx_px - w_px / 2
        x_max = cx_px + w_px / 2
        y_min = cy_px - h_px / 2
        y_max = cy_px + h_px / 2

        # Tile boundaries
        tile_x_max = tile_x + tile_size
        tile_y_max = tile_y + tile_size

        # Check if box center is within tile (center-based inclusion)
        if not (tile_x <= cx_px < tile_x_max and tile_y <= cy_px < tile_y_max):
            continue

        # Clamp box to tile
        x_min_clamp = max(x_min, tile_x)
        x_max_clamp = min(x_max, tile_x_max)
        y_min_clamp = max(y_min, tile_y)
        y_max_clamp = min(y_max, tile_y_max)

        # Compute new width/height after clamping
        new_w = x_max_clamp - x_min_clamp
        new_h = y_max_clamp - y_min_clamp

        # Skip if box is too small after clamping (< 10px in either dim)
        if new_w < 5 or new_h < 5:
            continue

        # Convert to tile-relative normalized coordinates
        new_cx = (x_min_clamp + new_w / 2 - tile_x) / tile_size
        new_cy = (y_min_clamp + new_h / 2 - tile_y) / tile_size
        new_w_norm = new_w / tile_size
        new_h_norm = new_h / tile_size

        # Clamp to [0, 1]
        new_cx = max(0.0, min(1.0, new_cx))
        new_cy = max(0.0, min(1.0, new_cy))
        new_w_norm = max(0.001, min(1.0, new_w_norm))
        new_h_norm = max(0.001, min(1.0, new_h_norm))

        tile_labels.append((cls, new_cx, new_cy, new_w_norm, new_h_norm))

    return tile_labels


def tile_image(img_path, lbl_path, dst_img_dir, dst_lbl_dir, stem):
    """Tile a single image and create corresponding label files."""
    img = Image.open(str(img_path))
    img_w, img_h = img.size

    boxes = read_label(lbl_path)

    n_tiles_x = math.ceil((img_w - TILE_SIZE) / STRIDE) + 1 if img_w > TILE_SIZE else 1
    n_tiles_y = math.ceil((img_h - TILE_SIZE) / STRIDE) + 1 if img_h > TILE_SIZE else 1

    tiles_saved = 0
    bg_tiles_saved = 0

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile_x = tx * STRIDE
            tile_y = ty * STRIDE

            # Clamp to image bounds
            tile_x = min(tile_x, max(0, img_w - TILE_SIZE))
            tile_y = min(tile_y, max(0, img_h - TILE_SIZE))

            # Get tile labels
            tile_labels = get_tile_labels(boxes, img_w, img_h, tile_x, tile_y, TILE_SIZE)

            # Skip empty tiles (unless within background budget)
            if not tile_labels:
                if bg_tiles_saved >= MAX_BACKGROUND_TILES:
                    continue
                bg_tiles_saved += 1

            # Crop tile from image
            tile_img = img.crop((tile_x, tile_y, tile_x + TILE_SIZE, tile_y + TILE_SIZE))

            # Handle images smaller than tile size
            if tile_img.size != (TILE_SIZE, TILE_SIZE):
                padded = Image.new('RGB', (TILE_SIZE, TILE_SIZE), (114, 114, 114))
                padded.paste(tile_img, (0, 0))
                tile_img = padded

            # Save tile
            tile_name = f"{stem}_tx{tx}_ty{ty}"
            tile_img.save(str(dst_img_dir / f"{tile_name}.jpg"), quality=95)
            write_label(dst_lbl_dir / f"{tile_name}.txt", tile_labels)
            tiles_saved += 1

    return tiles_saved


def setup_dirs():
    """Create output directory structure."""
    for split in ['train', 'val', 'test']:
        (DST / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DST / 'labels' / split).mkdir(parents=True, exist_ok=True)


def copy_split_unchanged(split):
    """Copy val and test splits unchanged (evaluate on full images)."""
    src_img = SRC / 'images' / split
    src_lbl = SRC / 'labels' / split
    dst_img = DST / 'images' / split
    dst_lbl = DST / 'labels' / split

    n = 0
    for img_path in src_img.glob('*'):
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            shutil.copy2(img_path, dst_img / img_path.name)
            lbl_path = src_lbl / (img_path.stem + '.txt')
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl / lbl_path.name)
            n += 1

    print(f"  {split}: copied {n} images unchanged")


def process_train():
    """Tile all training images."""
    src_img = SRC / 'images' / 'train'
    src_lbl = SRC / 'labels' / 'train'
    dst_img = DST / 'images' / 'train'
    dst_lbl = DST / 'labels' / 'train'

    img_paths = sorted([p for p in src_img.glob('*')
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])

    total_tiles = 0
    for i, img_path in enumerate(img_paths):
        lbl_path = src_lbl / (img_path.stem + '.txt')
        n_tiles = tile_image(img_path, lbl_path, dst_img, dst_lbl, img_path.stem)
        total_tiles += n_tiles

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(img_paths)} images, {total_tiles} tiles so far...")

    print(f"  Done: {len(img_paths)} images → {total_tiles} tiles")
    return total_tiles


def create_data_yaml():
    """Create data.yaml."""
    yaml_content = f"""path: /workspace/autoresearch/Dataset-Tiled
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
    print(f"  data.yaml created at {DST / 'data.yaml'}")


def count_class_instances():
    """Count instances in tiled train set."""
    lbl_dir = DST / 'labels' / 'train'
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for lbl_file in lbl_dir.glob('*.txt'):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    if cls in counts:
                        counts[cls] += 1
    return counts


if __name__ == '__main__':
    print(f"Creating tiled dataset: tile_size={TILE_SIZE}, stride={STRIDE}")
    print(f"  Source: {SRC}")
    print(f"  Destination: {DST}")

    print("\nSetting up directories...")
    setup_dirs()

    print("\nCopying val and test (unchanged — evaluate on full images)...")
    copy_split_unchanged('val')
    copy_split_unchanged('test')

    print("\nTiling training images...")
    n_tiles = process_train()

    print("\nCreating data.yaml...")
    create_data_yaml()

    print("\nVerifying tiled dataset...")
    counts = count_class_instances()
    n_train_imgs = len(list((DST / 'images' / 'train').glob('*')))
    print(f"  Train tiles: {n_train_imgs}")
    print(f"  Class distribution: B1={counts[0]}, B2={counts[1]}, B3={counts[2]}, B4={counts[3]}")

    print(f"\nDone! Dataset saved to {DST}")

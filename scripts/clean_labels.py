"""
Label noise correction via high-confidence model prediction disagreement.

Strategy:
1. Load current best model
2. Run inference on training images
3. For each GT box: find model prediction with IoU > 0.5
4. If model prediction class != GT class AND confidence > threshold:
   - Flag as potential mislabel
5. Focus on B2/B3 confusion (most common)
6. Auto-correct the most confident mismatches
7. Save to Dataset-Cleaned/

Historical evidence: B2→B3 confusion: 208 cases, B3→B2: 85 cases
B2 mAP50-95 = 0.197 (weakest, limited by label noise)
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

SRC = Path("/workspace/autoresearch/Dataset-YOLO")
DST = Path("/workspace/autoresearch/Dataset-Cleaned")
BEST_MODEL = Path("/workspace/autoresearch/runs/autoresearch/train/weights/best.pt")

# Thresholds
CONF_THRESHOLD = 0.70   # Minimum confidence for correction
IOU_THRESHOLD = 0.50    # Minimum IoU between pred box and GT box
MAX_CORRECTIONS = 500   # Max total label corrections

CLASS_NAMES = {0: "B1", 1: "B2", 2: "B3", 3: "B4"}


def iou(box1, box2):
    """Compute IoU between two YOLO format boxes (cx, cy, w, h normalized)."""
    x1_min = box1[0] - box1[2] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def read_label(label_path):
    """Read YOLO label file, return list of [class_id, cx, cy, w, h]."""
    boxes = []
    if Path(label_path).exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append([int(parts[0]), float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4])])
    return boxes


def write_label(label_path, boxes):
    """Write YOLO label file."""
    with open(label_path, 'w') as f:
        for box in boxes:
            f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def setup_dirs():
    """Create output directory structure."""
    for split in ['train', 'val', 'test']:
        (DST / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DST / 'labels' / split).mkdir(parents=True, exist_ok=True)


def copy_split_unchanged(split):
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


def find_label_corrections(model, src_img_dir, src_lbl_dir):
    """Find potential label errors via model prediction disagreement."""
    print(f"  Running inference on {src_img_dir}...")

    img_paths = sorted([p for p in src_img_dir.glob('*')
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])

    corrections = []  # List of (img_path, gt_box_idx, gt_class, pred_class, conf, iou_score)

    for img_path in img_paths:
        lbl_path = src_lbl_dir / (img_path.stem + '.txt')
        gt_boxes = read_label(lbl_path)

        if not gt_boxes:
            continue

        # Run model inference
        results = model.predict(str(img_path), conf=0.1, iou=0.5, verbose=False)

        if not results or len(results[0].boxes) == 0:
            continue

        result = results[0]
        h, w = result.orig_shape

        # Convert model predictions to normalized xywh format
        pred_boxes = []
        for det in result.boxes:
            cls_id = int(det.cls[0])
            conf_score = float(det.conf[0])
            xyxy = det.xyxy[0].tolist()
            cx = (xyxy[0] + xyxy[2]) / 2 / w
            cy = (xyxy[1] + xyxy[3]) / 2 / h
            bw = (xyxy[2] - xyxy[0]) / w
            bh = (xyxy[3] - xyxy[1]) / h
            pred_boxes.append((cls_id, cx, cy, bw, bh, conf_score))

        # Match each GT box to best prediction
        for i, gt_box in enumerate(gt_boxes):
            gt_cls = gt_box[0]
            best_iou = 0
            best_pred = None

            for pred in pred_boxes:
                pred_cls, pcx, pcy, pw, ph, pconf = pred
                gt_coords = gt_box[1:]
                pred_coords = (pcx, pcy, pw, ph)
                iou_score = iou(gt_coords, pred_coords)

                if iou_score > best_iou:
                    best_iou = iou_score
                    best_pred = pred

            # Check if high-confidence mismatch
            if best_pred is not None and best_iou >= IOU_THRESHOLD:
                pred_cls, pcx, pcy, pw, ph, pconf = best_pred
                if pred_cls != gt_cls and pconf >= CONF_THRESHOLD:
                    # Focus on B2/B3 confusion
                    if (gt_cls in [1, 2] and pred_cls in [1, 2]):
                        corrections.append({
                            'img_path': img_path,
                            'lbl_path': lbl_path,
                            'box_idx': i,
                            'gt_cls': gt_cls,
                            'pred_cls': pred_cls,
                            'conf': pconf,
                            'iou': best_iou,
                        })

    # Sort by confidence (most confident disagreements first)
    corrections.sort(key=lambda x: x['conf'], reverse=True)
    return corrections


def apply_corrections(corrections, max_corrections):
    """Apply label corrections to training data."""
    # Group corrections by file
    file_corrections = {}
    applied = 0
    for corr in corrections[:max_corrections]:
        lbl_path = str(corr['lbl_path'])
        if lbl_path not in file_corrections:
            file_corrections[lbl_path] = []
        file_corrections[lbl_path].append(corr)
        applied += 1

    return file_corrections, applied


def main():
    print("Setting up directories...")
    setup_dirs()

    print("Copying val and test (unchanged)...")
    copy_split_unchanged('val')
    copy_split_unchanged('test')

    print("Loading model for inference...")
    model = YOLO(str(BEST_MODEL))

    print("Finding label corrections in training set...")
    corrections = find_label_corrections(
        model,
        SRC / 'images' / 'train',
        SRC / 'labels' / 'train'
    )

    print(f"\nFound {len(corrections)} potential label mismatches (B2/B3 confusion)")

    # Show distribution
    b2_to_b3 = sum(1 for c in corrections if c['gt_cls'] == 1 and c['pred_cls'] == 2)
    b3_to_b2 = sum(1 for c in corrections if c['gt_cls'] == 2 and c['pred_cls'] == 1)
    print(f"  B2→B3 (model says B3, label says B2): {b2_to_b3}")
    print(f"  B3→B2 (model says B2, label says B3): {b3_to_b2}")

    # Show top 10 most confident corrections
    print("\nTop 10 highest-confidence corrections:")
    for i, corr in enumerate(corrections[:10]):
        gt_name = CLASS_NAMES[corr['gt_cls']]
        pred_name = CLASS_NAMES[corr['pred_cls']]
        print(f"  {i+1}. {corr['img_path'].name}: GT={gt_name} → pred={pred_name} "
              f"(conf={corr['conf']:.3f}, iou={corr['iou']:.3f})")

    # Apply corrections to copies
    n_to_correct = min(MAX_CORRECTIONS, len(corrections))
    print(f"\nApplying top {n_to_correct} corrections (sorted by confidence)...")

    file_corrections, n_applied = apply_corrections(corrections, n_to_correct)

    # Copy training data with corrections applied
    src_img_dir = SRC / 'images' / 'train'
    src_lbl_dir = SRC / 'labels' / 'train'
    dst_img_dir = DST / 'images' / 'train'
    dst_lbl_dir = DST / 'labels' / 'train'

    n_files_modified = 0
    n_boxes_corrected = 0

    for img_path in sorted(src_img_dir.glob('*')):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue

        lbl_path = src_lbl_dir / (img_path.stem + '.txt')
        shutil.copy2(img_path, dst_img_dir / img_path.name)

        gt_boxes = read_label(lbl_path)

        if str(lbl_path) in file_corrections:
            # Apply corrections to this file's labels
            for corr in file_corrections[str(lbl_path)]:
                box_idx = corr['box_idx']
                if box_idx < len(gt_boxes):
                    old_cls = gt_boxes[box_idx][0]
                    gt_boxes[box_idx][0] = corr['pred_cls']
                    n_boxes_corrected += 1
            n_files_modified += 1

        write_label(dst_lbl_dir / (img_path.stem + '.txt'), gt_boxes)

    print(f"\nResults:")
    print(f"  Files modified: {n_files_modified}")
    print(f"  Boxes corrected: {n_boxes_corrected}")
    print(f"  Total train images copied: {len(list(dst_img_dir.glob('*')))}")

    # Create data.yaml
    yaml_content = f"""path: /workspace/autoresearch/Dataset-Cleaned
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

    print(f"\nDone! Dataset saved to {DST}")
    print(f"data.yaml: {DST / 'data.yaml'}")


if __name__ == '__main__':
    main()

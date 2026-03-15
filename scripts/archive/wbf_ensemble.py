"""
Weighted Box Fusion (WBF) ensemble of multiple YOLO models.

Combines predictions from multiple models at inference time using WBF.
WBF (Solovyev et al. 2021) is better than NMS-based ensemble for object detection.

Models to ensemble:
1. best_yolo11l_e80.pt (YOLO11l, train+test, 640px, batch=16, epochs=80)
2. best_yolo11l_640_traintest.pt (YOLO11l, train+test, 640px, batch=16, epochs=40ish)

WBF merges overlapping boxes from multiple models by weighted average of coordinates.
"""

import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from prepare import DATA_YAML, CLASS_NAMES, DATASET_DIR

# Model paths
MODEL_PATHS = [
    "/workspace/autoresearch/best_yolo11l_e80.pt",
    "/workspace/autoresearch/best_yolo11l_640_traintest.pt",
    "/workspace/autoresearch/best_yolo11l_640.pt",
]
MODEL_WEIGHTS = [1.0, 0.8, 0.6]  # Weight for each model's predictions

# Eval params
IMGSZ = 640
CONF = 0.001
IOU_NMS = 0.6
WBF_IOU_THR = 0.55  # IoU threshold for WBF merging
WBF_SKIP_BOX_THR = 0.001  # Minimum box confidence to keep

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bb_intersection_over_union(A, B):
    """Compute IoU between two boxes [x1,y1,x2,y2] in [0,1] normalized."""
    Aarea = (A[2] - A[0]) * (A[3] - A[1])
    Barea = (B[2] - B[0]) * (B[3] - B[1])

    interx1 = max(A[0], B[0])
    intery1 = max(A[1], B[1])
    interx2 = min(A[2], B[2])
    intery2 = min(A[3], B[3])

    if interx2 <= interx1 or intery2 <= intery1:
        return 0.0

    inter = (interx2 - interx1) * (intery2 - intery1)
    union = Aarea + Barea - inter
    return inter / (union + 1e-6)


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights, iou_thr=0.55, skip_box_thr=0.001):
    """
    Weighted Box Fusion.

    boxes_list: list of arrays, each [N_i, 4] in [0,1]
    scores_list: list of arrays [N_i]
    labels_list: list of arrays [N_i] int class ids
    weights: list of floats

    Returns: merged boxes, scores, labels
    """
    if not boxes_list:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    weights = np.array(weights[:len(boxes_list)])
    weights = weights / weights.sum()

    # Pool all predictions
    all_boxes = []
    all_scores = []
    all_labels = []
    all_model_idx = []

    for model_i, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        for j in range(len(boxes)):
            if scores[j] >= skip_box_thr:
                all_boxes.append(boxes[j])
                all_scores.append(scores[j] * weights[model_i])
                all_labels.append(labels[j])
                all_model_idx.append(model_i)

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    all_boxes = np.array(all_boxes, dtype=float)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels, dtype=int)

    # Sort by descending score
    order = np.argsort(-all_scores)
    all_boxes = all_boxes[order]
    all_scores = all_scores[order]
    all_labels = all_labels[order]

    # Cluster overlapping boxes
    used = np.zeros(len(all_boxes), dtype=bool)
    merged_boxes = []
    merged_scores = []
    merged_labels = []

    for i in range(len(all_boxes)):
        if used[i]:
            continue

        # Find all boxes that overlap with box i (same class)
        cluster_boxes = [all_boxes[i]]
        cluster_scores = [all_scores[i]]
        cluster_labels = [all_labels[i]]
        used[i] = True

        for j in range(i+1, len(all_boxes)):
            if used[j]:
                continue
            if all_labels[j] != all_labels[i]:
                continue
            iou = bb_intersection_over_union(all_boxes[i], all_boxes[j])
            if iou >= iou_thr:
                cluster_boxes.append(all_boxes[j])
                cluster_scores.append(all_scores[j])
                used[j] = True

        # Weighted average of box coordinates (weight by score)
        cluster_boxes = np.array(cluster_boxes)
        cluster_scores = np.array(cluster_scores)
        w = cluster_scores / cluster_scores.sum()

        fused_box = np.average(cluster_boxes, axis=0, weights=w)
        fused_score = cluster_scores.sum() / len(weights)  # normalize by num models

        merged_boxes.append(fused_box)
        merged_scores.append(fused_score)
        merged_labels.append(all_labels[i])

    if not merged_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    return (np.array(merged_boxes), np.array(merged_scores),
            np.array(merged_labels, dtype=int))


def iou_single_xyxy(box1, box2):
    """IoU between two boxes in xyxy format (not normalized)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def compute_ap(recalls, precisions):
    """Compute Average Precision using 11-point interpolation."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec = max((p for r, p in zip(recalls, precisions) if r >= t), default=0.0)
        ap += prec / 11
    return ap


def read_gt(label_path):
    boxes = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append((cls, cx, cy, bw, bh))
    return boxes


def evaluate_wbf_ensemble():
    """Evaluate WBF ensemble on val set."""
    # Load models
    models = []
    for path in MODEL_PATHS:
        if Path(path).exists():
            print(f"  Loading: {path}")
            models.append(YOLO(path))
        else:
            print(f"  MISSING: {path}")

    if len(models) < 2:
        print("Need at least 2 models for ensemble!")
        return None

    model_weights = MODEL_WEIGHTS[:len(models)]

    val_img_dir = DATASET_DIR / "images" / "val"
    val_lbl_dir = DATASET_DIR / "labels" / "val"

    img_paths = sorted([p for p in val_img_dir.glob('*')
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])

    print(f"\n  Running WBF ensemble on {len(img_paths)} val images...")
    print(f"  Models: {len(models)}, weights: {model_weights}")

    n_classes = 4
    all_predictions = {c: [] for c in range(n_classes)}
    all_gt_counts = {c: 0 for c in range(n_classes)}

    for i, img_path in enumerate(img_paths):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(img_paths)}...")

        # Read GT
        gt_boxes_raw = read_gt(val_lbl_dir / (img_path.stem + '.txt'))
        for cls, *_ in gt_boxes_raw:
            all_gt_counts[cls] += 1

        # Get predictions from each model
        boxes_list = []
        scores_list = []
        labels_list = []

        img = None

        for model in models:
            results = model.predict(str(img_path), imgsz=IMGSZ, conf=CONF,
                                    iou=IOU_NMS, verbose=False)
            if not results or len(results[0].boxes) == 0:
                boxes_list.append(np.zeros((0, 4)))
                scores_list.append(np.zeros(0))
                labels_list.append(np.zeros(0, dtype=int))
                if img is None:
                    from PIL import Image
                    img = Image.open(str(img_path))
                continue

            result = results[0]
            if img is None:
                img_w, img_h = result.orig_shape[1], result.orig_shape[0]
                # Normalize boxes to [0,1]
                xyxy = result.boxes.xyxy.cpu().numpy()
                boxes_norm = xyxy / np.array([img_w, img_h, img_w, img_h])
            else:
                img_w, img_h = result.orig_shape[1], result.orig_shape[0]
                xyxy = result.boxes.xyxy.cpu().numpy()
                boxes_norm = xyxy / np.array([img_w, img_h, img_w, img_h])

            img_w = result.orig_shape[1]
            img_h = result.orig_shape[0]
            xyxy = result.boxes.xyxy.cpu().numpy()
            boxes_norm = xyxy / np.array([img_w, img_h, img_w, img_h])
            boxes_norm = np.clip(boxes_norm, 0, 1)

            boxes_list.append(boxes_norm)
            scores_list.append(result.boxes.conf.cpu().numpy())
            labels_list.append(result.boxes.cls.cpu().numpy().astype(int))

        if img is None:
            result = models[0].predict(str(img_path), imgsz=IMGSZ, conf=CONF, verbose=False)
            img_w = result[0].orig_shape[1]
            img_h = result[0].orig_shape[0]
        else:
            result = models[0].predict(str(img_path), imgsz=IMGSZ, conf=CONF, verbose=False)
            img_w = result[0].orig_shape[1]
            img_h = result[0].orig_shape[0]

        # Run WBF
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, model_weights,
            iou_thr=WBF_IOU_THR, skip_box_thr=WBF_SKIP_BOX_THR
        )

        if len(merged_boxes) == 0:
            continue

        # Convert GT to xyxy
        gt_boxes_xyxy = []
        for cls, cx, cy, bw, bh in gt_boxes_raw:
            x1 = (cx - bw/2) * img_w
            y1 = (cy - bh/2) * img_h
            x2 = (cx + bw/2) * img_w
            y2 = (cy + bh/2) * img_h
            gt_boxes_xyxy.append((cls, [x1, y1, x2, y2]))

        # Convert predictions to absolute coords
        pred_boxes_xyxy = merged_boxes * np.array([img_w, img_h, img_w, img_h])

        # Match predictions to GT at IoU=0.5
        matched_gt = set()
        for j, (pred_cls, pred_score) in enumerate(zip(merged_labels, merged_scores)):
            best_iou = 0
            best_gt_idx = -1
            for k, (gt_cls, gt_box) in enumerate(gt_boxes_xyxy):
                if gt_cls != pred_cls:
                    continue
                iou = iou_single_xyxy(pred_boxes_xyxy[j], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = k

            is_tp = (best_iou >= 0.5 and best_gt_idx not in matched_gt)
            if is_tp:
                matched_gt.add(best_gt_idx)

            all_predictions[pred_cls].append((pred_score, is_tp))

    # Compute AP per class at IoU=0.5
    per_class_ap50 = {}
    for cls in range(n_classes):
        preds = sorted(all_predictions[cls], key=lambda x: x[0], reverse=True)
        n_gt = all_gt_counts[cls]

        if n_gt == 0 or not preds:
            per_class_ap50[cls] = 0.0
            continue

        tp_cumsum = fp_cumsum = 0
        recalls, precisions = [], []
        for conf, is_tp in preds:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            recalls.append(tp_cumsum / n_gt)
            precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))

        per_class_ap50[cls] = compute_ap(recalls, precisions)

    map50 = np.mean(list(per_class_ap50.values()))

    print(f"\nWBF Ensemble results:")
    print(f"  GT counts: {all_gt_counts}")
    for cls in range(n_classes):
        print(f"  {CLASS_NAMES[cls]}: AP50={per_class_ap50[cls]:.4f}")
    print(f"  mAP50: {map50:.6f}")
    print(f"  Note: mAP50-95 requires IoU=0.5:0.95 evaluation (not computed here)")
    print(f"  Approximate mAP50-95 ≈ {map50 * 0.48:.6f} (using observed ratio)")

    return map50


if __name__ == '__main__':
    print("=== WBF Ensemble Evaluation ===")
    print(f"Val images: {DATASET_DIR / 'images' / 'val'}")
    print(f"Models: {MODEL_PATHS}")
    print()

    print("Loading models...")
    map50 = evaluate_wbf_ensemble()

    if map50:
        print(f"\nBaseline single model mAP50: ~0.554 (best YOLO11l)")
        print(f"WBF ensemble mAP50: {map50:.6f}")

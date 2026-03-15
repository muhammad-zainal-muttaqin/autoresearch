"""
Two-stage pipeline evaluation v2 — FIXED VERSION.

Fixes over v1:
1. Computes TRUE mAP50-95 over 10 IoU thresholds (0.50:0.05:0.95) instead of approximation
2. Uses 101-point AP interpolation (COCO-style) instead of 11-point (Pascal VOC)
3. Detector runs at IMGSZ=640 (matching training resolution) instead of 1024
4. Cleaner per-class reporting
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from prepare import DATA_YAML, CLASS_NAMES
from stage2_models import (
    classifier_logits_to_probs,
    load_stage2_classifier,
)

DETECTOR_PATH = Path("/workspace/autoresearch/stage1_detector.pt")
CLASSIFIER_PATH = Path("/workspace/autoresearch/stage2_classifier.pth")
DINOV2_CLASSIFIER_PATH = Path("/workspace/autoresearch/stage2_dinov2_classifier.pth")
ORDINAL_CLASSIFIER_PATH = Path("/workspace/autoresearch/stage2_dinov2_ordinal_classifier.pth")

DETECTOR_CONF = 0.1
DETECTOR_IOU = 0.5
CLASSIFIER_BATCH = 32
IMGSZ = 640  # FIX: match training resolution (was 1024 in v1)
PAD_RATIO = 0.2

IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)  # [0.50, 0.55, ..., 0.95]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_gt_labels(label_dir, img_stem):
    label_path = label_dir / (img_stem + ".txt")
    boxes = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append([int(parts[0]), float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4])])
    return boxes


def iou_single(box1_xyxy, box2_xyxy):
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    a2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    return inter / (a1 + a2 - inter + 1e-6)


def compute_ap_101point(recalls, precisions, n_gt):
    """
    COCO-style 101-point AP interpolation.
    More accurate than 11-point VOC interpolation.
    """
    if n_gt == 0:
        return 0.0
    # Append sentinel values
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])
    # Make precision monotonically decreasing from right
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    # 101 recall thresholds
    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for t in recall_thresholds:
        precs_at_t = precisions[recalls >= t]
        ap += (precs_at_t[0] if len(precs_at_t) > 0 else 0.0)
    ap /= 101
    return ap


def evaluate_pipeline(
    detector,
    classifier,
    classifier_type,
    val_img_dir,
    val_lbl_dir,
    *,
    detector_conf=DETECTOR_CONF,
    detector_iou=DETECTOR_IOU,
    imgsz=IMGSZ,
    pad_ratio=PAD_RATIO,
):
    """
    Evaluate two-stage pipeline.
    Computes true mAP50-95 over 10 IoU thresholds.
    """
    clf_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    n_classes = 4
    n_iou = len(IOU_THRESHOLDS)

    # all_predictions[cls][iou_idx] = list of (conf, is_tp)
    all_predictions = {c: {t: [] for t in range(n_iou)} for c in range(n_classes)}
    all_gt_counts = {c: 0 for c in range(n_classes)}

    img_paths = sorted([p for p in val_img_dir.glob("*")
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"  Evaluating on {len(img_paths)} val images...")

    for i, img_path in enumerate(img_paths):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(img_paths)}...")

        gt_boxes = read_gt_labels(val_lbl_dir, img_path.stem)
        for gb in gt_boxes:
            all_gt_counts[gb[0]] += 1

        results = detector.predict(
            str(img_path),
            imgsz=imgsz,
            conf=detector_conf,
            iou=detector_iou,
            verbose=False,
        )
        if not results or len(results[0].boxes) == 0:
            continue

        result = results[0]
        img = Image.open(str(img_path)).convert("RGB")
        img_w, img_h = img.size

        crops, det_confs, det_boxes_xyxy = [], [], []
        for det in result.boxes:
            conf_score = float(det.conf[0])
            x1, y1, x2, y2 = det.xyxy[0].tolist()
            w_box, h_box = x2 - x1, y2 - y1
            cx1 = max(0, int(x1 - w_box * pad_ratio))
            cy1 = max(0, int(y1 - h_box * pad_ratio))
            cx2 = min(img_w, int(x2 + w_box * pad_ratio))
            cy2 = min(img_h, int(y2 + h_box * pad_ratio))
            if cx2 - cx1 < 5 or cy2 - cy1 < 5:
                continue
            crop = img.crop((cx1, cy1, cx2, cy2))
            crops.append(clf_tf(crop))
            det_confs.append(conf_score)
            det_boxes_xyxy.append([x1, y1, x2, y2])

        if not crops:
            continue

        crop_batch = torch.stack(crops).to(DEVICE)
        with torch.no_grad():
            logits = classifier(crop_batch)
            probs = classifier_logits_to_probs(classifier_type, logits)
            pred_classes = probs.argmax(dim=1).cpu().numpy()
            pred_conf_cls = probs.max(dim=1).values.cpu().numpy()

        combined_confs = [dc * pc for dc, pc in zip(det_confs, pred_conf_cls)]

        # Convert GT to xyxy
        gt_boxes_xyxy = []
        for gb in gt_boxes:
            cls, cx, cy, bw, bh = gb
            gt_boxes_xyxy.append((cls, [
                (cx - bw/2) * img_w, (cy - bh/2) * img_h,
                (cx + bw/2) * img_w, (cy + bh/2) * img_h
            ]))

        # For each IoU threshold
        for t_idx, iou_thresh in enumerate(IOU_THRESHOLDS):
            matched_gt = set()
            for j, (pred_cls, comb_conf, det_box) in enumerate(
                    zip(pred_classes, combined_confs, det_boxes_xyxy)):
                best_iou = 0
                best_gt_idx = -1
                for k, (gt_cls, gt_box) in enumerate(gt_boxes_xyxy):
                    if gt_cls != pred_cls:
                        continue
                    iou_val = iou_single(det_box, gt_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = k

                is_tp = (best_iou >= iou_thresh and best_gt_idx not in matched_gt)
                if is_tp:
                    matched_gt.add(best_gt_idx)
                all_predictions[pred_cls][t_idx].append((comb_conf, is_tp))

    # Compute per-class AP at each IoU threshold
    per_class_ap = {c: [] for c in range(n_classes)}
    for cls in range(n_classes):
        n_gt = all_gt_counts[cls]
        for t_idx in range(n_iou):
            preds = sorted(all_predictions[cls][t_idx], key=lambda x: x[0], reverse=True)
            if not preds or n_gt == 0:
                per_class_ap[cls].append(0.0)
                continue
            tp_cumsum, fp_cumsum = 0, 0
            recalls, precisions = [], []
            for conf, is_tp in preds:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                recalls.append(tp_cumsum / n_gt)
                precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
            per_class_ap[cls].append(compute_ap_101point(
                np.array(recalls), np.array(precisions), n_gt))

    # mAP50 = avg per-class AP at IoU=0.5
    map50 = np.mean([per_class_ap[c][0] for c in range(n_classes)])
    # mAP50-95 = avg over all IoU thresholds and all classes
    map50_95 = np.mean([np.mean(per_class_ap[c]) for c in range(n_classes)])

    print(f"\nTwo-stage pipeline v2 results:")
    print(f"  GT counts: B1={all_gt_counts[0]}, B2={all_gt_counts[1]}, B3={all_gt_counts[2]}, B4={all_gt_counts[3]}")
    print(f"  AP50 per class:")
    for cls, name in enumerate(CLASS_NAMES):
        ap_at_iou = [per_class_ap[cls][t] for t in range(n_iou)]
        print(f"    {name}: AP50={ap_at_iou[0]:.4f}, mean_AP50-95={np.mean(ap_at_iou):.4f}")
    print(f"  mAP50:    {map50:.6f}")
    print(f"  mAP50-95: {map50_95:.6f}")
    print(f"  Baseline (YOLO11l end-to-end): 0.269424")
    print(f"  Delta vs baseline: {map50_95 - 0.269424:+.6f}")

    return map50, map50_95, per_class_ap


def pick_classifier_path(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)
    for candidate in (
        ORDINAL_CLASSIFIER_PATH,
        DINOV2_CLASSIFIER_PATH,
        CLASSIFIER_PATH,
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No stage-2 classifier checkpoint found.")


if __name__ == "__main__":
    import os
    from prepare import DATA_YAML

    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-path", default=str(DETECTOR_PATH))
    parser.add_argument("--classifier-path", default=None)
    parser.add_argument("--det-conf", type=float, default=DETECTOR_CONF)
    parser.add_argument("--det-iou", type=float, default=DETECTOR_IOU)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--pad-ratio", type=float, default=PAD_RATIO)
    args = parser.parse_args()

    os.chdir(DATA_YAML.parent)

    print("Loading detector...")
    detector = YOLO(str(Path(args.detector_path)))

    classifier_path = pick_classifier_path(args.classifier_path)
    print(f"Loading classifier: {classifier_path}")
    loaded = load_stage2_classifier(classifier_path, DEVICE)
    print(f"Classifier type: {loaded.classifier_type}")

    val_img_dir = DATA_YAML.parent / "images" / "val"
    val_lbl_dir = DATA_YAML.parent / "labels" / "val"

    print("\nRunning v2 evaluation...")
    map50, map50_95, per_class_ap = evaluate_pipeline(
        detector,
        loaded.model,
        loaded.classifier_type,
        val_img_dir,
        val_lbl_dir,
        detector_conf=args.det_conf,
        detector_iou=args.det_iou,
        imgsz=args.imgsz,
        pad_ratio=args.pad_ratio,
    )

    print(f"\nFINAL: mAP50={map50:.6f}, mAP50-95={map50_95:.6f}")

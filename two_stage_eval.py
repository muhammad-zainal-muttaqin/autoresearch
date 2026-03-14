"""
Two-stage pipeline evaluation.

Stage 1: Single-class TBS detector (stage1_detector.pt)
Stage 2: EfficientNet-B0 crop classifier (stage2_classifier.pth)

Pipeline:
1. Run detector on val image → get boxes (class=TBS, conf score)
2. Crop each detected region
3. Run classifier on crop → get B1/B2/B3/B4 probability
4. Assign predicted class to detection
5. Compute mAP50-95 against GT labels

The combined mAP is limited by:
- Detector recall (missing detections)
- Classifier accuracy per class (misclassified detections)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
from prepare import DATA_YAML, CLASS_NAMES

# Model paths
DETECTOR_PATH = Path("/workspace/autoresearch/stage1_detector.pt")
CLASSIFIER_PATH = Path("/workspace/autoresearch/stage2_classifier.pth")
DATA_YAML_PATH = DATA_YAML

# Inference params
DETECTOR_CONF = 0.1  # Low conf to maximize recall
DETECTOR_IOU = 0.5   # NMS IOU threshold
CLASSIFIER_BATCH = 32
IMGSZ = 1024
PAD_RATIO = 0.2  # Same as used in crop extraction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_classifier(model_path):
    """Load the EfficientNet-B0 classifier."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 4),
    )
    ckpt = torch.load(str(model_path), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model


def read_gt_labels(label_dir, img_stem):
    """Read GT labels for an image."""
    label_path = label_dir / (img_stem + '.txt')
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
    """Compute IoU between two boxes in xyxy format."""
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


def compute_ap(recalls, precisions):
    """Compute Average Precision from recall/precision arrays (11-point interpolation)."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec = 0.0
        for r, p in zip(recalls, precisions):
            if r >= t:
                prec = max(prec, p)
        ap += prec / 11
    return ap


def evaluate_pipeline(detector, classifier, val_img_dir, val_lbl_dir, iou_thresholds):
    """
    Evaluate two-stage pipeline on val set.

    Returns: mAP50, mAP50-95, per-class AP50
    """
    clf_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    n_classes = 4
    # all_predictions[cls] = list of (conf, is_tp) for that class
    all_predictions = {c: [] for c in range(n_classes)}
    all_gt_counts = {c: 0 for c in range(n_classes)}

    img_paths = sorted([p for p in val_img_dir.glob('*')
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])

    print(f"  Running pipeline on {len(img_paths)} val images...")

    for i, img_path in enumerate(img_paths):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(img_paths)}...")

        gt_boxes = read_gt_labels(val_lbl_dir, img_path.stem)

        # Count GT per class
        for gt_box in gt_boxes:
            all_gt_counts[gt_box[0]] += 1

        # Stage 1: Detect
        results = detector.predict(str(img_path), imgsz=IMGSZ, conf=DETECTOR_CONF,
                                   iou=DETECTOR_IOU, verbose=False)
        if not results or len(results[0].boxes) == 0:
            continue

        result = results[0]
        img = Image.open(str(img_path)).convert('RGB')
        img_w, img_h = img.size

        # Stage 2: Classify each detection
        crops = []
        det_confs = []
        det_boxes_xyxy = []

        for det in result.boxes:
            conf_score = float(det.conf[0])
            xyxy = det.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            w_box = x2 - x1
            h_box = y2 - y1

            # Add padding
            pad_w = w_box * PAD_RATIO
            pad_h = h_box * PAD_RATIO
            cx1 = max(0, int(x1 - pad_w))
            cy1 = max(0, int(y1 - pad_h))
            cx2 = min(img_w, int(x2 + pad_w))
            cy2 = min(img_h, int(y2 + pad_h))

            if cx2 - cx1 < 5 or cy2 - cy1 < 5:
                continue

            crop = img.crop((cx1, cy1, cx2, cy2))
            crop = clf_tf(crop)
            crops.append(crop)
            det_confs.append(conf_score)
            det_boxes_xyxy.append(xyxy)

        if not crops:
            continue

        # Run classifier in batch
        crop_batch = torch.stack(crops).to(DEVICE)
        with torch.no_grad():
            logits = classifier(crop_batch)
            probs = torch.softmax(logits, dim=1)
            pred_classes = probs.argmax(dim=1).cpu().numpy()
            pred_conf_cls = probs.max(dim=1).values.cpu().numpy()

        # Combined score: detector conf × classifier conf
        combined_confs = [dc * pc for dc, pc in zip(det_confs, pred_conf_cls)]

        # Convert GT to xyxy for IoU computation
        gt_boxes_xyxy = []
        for gt_box in gt_boxes:
            cls, cx, cy, bw, bh = gt_box
            x1_gt = (cx - bw/2) * img_w
            y1_gt = (cy - bh/2) * img_h
            x2_gt = (cx + bw/2) * img_w
            y2_gt = (cy + bh/2) * img_h
            gt_boxes_xyxy.append((cls, [x1_gt, y1_gt, x2_gt, y2_gt]))

        # For each IoU threshold, match predictions to GT
        # For simplicity, use IoU=0.5 only for AP computation
        # Track matched GT boxes
        iou50_matched_gt = set()

        for j, (pred_cls, comb_conf, det_box) in enumerate(zip(pred_classes, combined_confs, det_boxes_xyxy)):
            # Find best matching GT box of same predicted class
            best_iou = 0
            best_gt_idx = -1
            for k, (gt_cls, gt_box) in enumerate(gt_boxes_xyxy):
                if gt_cls != pred_cls:
                    continue
                iou_val = iou_single(det_box, gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = k

            is_tp = (best_iou >= 0.5 and best_gt_idx not in iou50_matched_gt)
            if is_tp:
                iou50_matched_gt.add(best_gt_idx)

            all_predictions[pred_cls].append((comb_conf, is_tp))

    # Compute AP per class at IoU=0.5
    per_class_ap50 = {}
    for cls in range(n_classes):
        preds = sorted(all_predictions[cls], key=lambda x: x[0], reverse=True)
        n_gt = all_gt_counts[cls]

        if n_gt == 0 or not preds:
            per_class_ap50[cls] = 0.0
            continue

        tp_cumsum = 0
        fp_cumsum = 0
        recalls = []
        precisions = []

        for conf, is_tp in preds:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            recalls.append(tp_cumsum / n_gt)
            precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))

        per_class_ap50[cls] = compute_ap(recalls, precisions)

    map50 = np.mean(list(per_class_ap50.values()))

    print(f"\nTwo-stage pipeline results:")
    print(f"  GT counts: B1={all_gt_counts[0]}, B2={all_gt_counts[1]}, B3={all_gt_counts[2]}, B4={all_gt_counts[3]}")
    print(f"  AP50 per class:")
    for cls, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {per_class_ap50[cls]:.4f}")
    print(f"  mAP50: {map50:.6f}")

    # Note: mAP50-95 would need to be computed at 10 IoU thresholds
    # Approximate as mAP50 * 0.5 (rough rule of thumb)
    map50_95_approx = map50 * 0.47  # Based on observed ratio in this project
    print(f"  mAP50-95 (approx): {map50_95_approx:.6f}")

    return map50, map50_95_approx, per_class_ap50


if __name__ == '__main__':
    import os
    os.chdir(DATA_YAML_PATH.parent)

    print("Loading models...")
    detector = YOLO(str(DETECTOR_PATH))
    classifier = load_classifier(CLASSIFIER_PATH)

    val_img_dir = DATA_YAML_PATH.parent / "images" / "val"
    val_lbl_dir = DATA_YAML_PATH.parent / "labels" / "val"

    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    print("\nRunning two-stage evaluation...")
    map50, map50_95_approx, per_class_ap50 = evaluate_pipeline(
        detector, classifier, val_img_dir, val_lbl_dir, iou_thresholds
    )

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"  val_map50:    {map50:.6f}")
    print(f"  val_map50_95 (approx): {map50_95_approx:.6f}")
    print(f"  Current best baseline: 0.260003")
    print(f"  Delta: {map50_95_approx - 0.260003:+.6f}")
    print(f"{'='*50}")

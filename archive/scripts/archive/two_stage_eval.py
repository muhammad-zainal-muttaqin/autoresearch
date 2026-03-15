"""
Two-stage pipeline evaluation.

Stage 1: Single-class TBS detector (stage1_detector.pt)
Stage 2: EfficientNet-B0 or DINOv2 crop classifier

Pipeline:
1. Run detector on val image → get boxes (class=TBS, conf score)
2. Crop each detected region
3. Run classifier on crop → get B1/B2/B3/B4 probability
4. Assign predicted class to detection
5. Compute mAP50-95 against GT labels (COCO protocol, 10 IoU thresholds)

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
# Set to stage2_dinov2_classifier.pth to use DINOv2 backbone instead
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
    """Load classifier — auto-detects EfficientNet-B0 or DINOv2 from checkpoint."""
    ckpt = torch.load(str(model_path), map_location=DEVICE, weights_only=False)

    if 'dinov2_model' in ckpt:
        # DINOv2-based classifier
        print(f"  Loading DINOv2 classifier (backbone: {ckpt['dinov2_model']})")
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(ckpt['dinov2_model'])
        hidden_size = backbone.config.hidden_size
        head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4),
        )

        class DINOv2Clf(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                out = self.backbone(pixel_values=x)
                cls_emb = out.last_hidden_state[:, 0, :]
                return self.head(cls_emb)

        model = DINOv2Clf(backbone, head)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        # EfficientNet-B0 classifier (legacy)
        print("  Loading EfficientNet-B0 classifier")
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 4),
        )
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

    mAP50-95 is computed correctly by evaluating at each IoU threshold in
    [0.50, 0.55, 0.60, ..., 0.95] and averaging — same as COCO protocol.
    """
    clf_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    n_classes = 4

    img_paths = sorted([p for p in val_img_dir.glob('*')
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])

    print(f"  Running pipeline on {len(img_paths)} val images...")

    # Collect all detections across the full val set first (one pass)
    # all_detections: list of (img_idx, pred_cls, comb_conf, det_box_xyxy)
    all_detections = []
    # all_gt: list of (img_idx, gt_cls, gt_box_xyxy)
    all_gt = []
    all_gt_counts = {c: 0 for c in range(n_classes)}

    for i, img_path in enumerate(img_paths):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(img_paths)}...")

        gt_boxes = read_gt_labels(val_lbl_dir, img_path.stem)

        img = Image.open(str(img_path)).convert('RGB')
        img_w, img_h = img.size

        # Count GT and store in absolute coords
        for gt_box in gt_boxes:
            cls, cx, cy, bw, bh = gt_box
            all_gt_counts[cls] += 1
            x1_gt = (cx - bw / 2) * img_w
            y1_gt = (cy - bh / 2) * img_h
            x2_gt = (cx + bw / 2) * img_w
            y2_gt = (cy + bh / 2) * img_h
            all_gt.append((i, cls, [x1_gt, y1_gt, x2_gt, y2_gt]))

        # Stage 1: Detect
        results = detector.predict(str(img_path), imgsz=IMGSZ, conf=DETECTOR_CONF,
                                   iou=DETECTOR_IOU, verbose=False)
        if not results or len(results[0].boxes) == 0:
            continue

        result = results[0]

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
        for pred_cls, dc, pc, det_box in zip(pred_classes, det_confs, pred_conf_cls, det_boxes_xyxy):
            combined_conf = float(dc) * float(pc)
            all_detections.append((i, int(pred_cls), combined_conf, det_box))

    # ── Compute mAP at each IoU threshold (COCO protocol) ────────────────────
    # Sort all detections by confidence (descending) once — reuse across thresholds
    all_detections_sorted = sorted(all_detections, key=lambda x: x[2], reverse=True)

    def compute_map_at_iou(iou_thresh):
        """Compute mean AP across 4 classes at a single IoU threshold."""
        # per-class lists of (conf, is_tp) — built via a single pass over sorted dets
        per_class_preds = {c: [] for c in range(n_classes)}
        matched_gt = set()  # (img_idx, gt_idx) pairs already matched

        for img_idx, pred_cls, conf, det_box in all_detections_sorted:
            # Find best matching GT box of same class in same image
            best_iou = 0.0
            best_key = None
            for gt_idx, (g_img_idx, g_cls, g_box) in enumerate(all_gt):
                if g_img_idx != img_idx or g_cls != pred_cls:
                    continue
                key = (img_idx, gt_idx)
                if key in matched_gt:
                    continue
                iou_val = iou_single(det_box, g_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_key = key

            is_tp = best_iou >= iou_thresh and best_key is not None
            if is_tp:
                matched_gt.add(best_key)
            per_class_preds[pred_cls].append((conf, is_tp))

        ap_per_class = {}
        for cls in range(n_classes):
            preds = per_class_preds[cls]
            n_gt = all_gt_counts[cls]
            if n_gt == 0 or not preds:
                ap_per_class[cls] = 0.0
                continue
            tp_cumsum = 0
            fp_cumsum = 0
            recalls = []
            precisions = []
            for _, is_tp in preds:  # already sorted globally
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                recalls.append(tp_cumsum / n_gt)
                precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
            ap_per_class[cls] = compute_ap(recalls, precisions)
        return np.mean(list(ap_per_class.values())), ap_per_class

    # AP50
    map50, per_class_ap50 = compute_map_at_iou(0.5)

    # mAP50-95: average over IoU thresholds 0.50..0.95 step 0.05
    iou_thresholds_coco = np.arange(0.5, 1.0, 0.05)
    map_values = []
    per_class_ap_accum = {c: [] for c in range(n_classes)}
    for iou_t in iou_thresholds_coco:
        m, ap_c = compute_map_at_iou(round(float(iou_t), 2))
        map_values.append(m)
        for c in range(n_classes):
            per_class_ap_accum[c].append(ap_c[c])
    map50_95 = float(np.mean(map_values))
    per_class_ap50_95 = {c: float(np.mean(per_class_ap_accum[c])) for c in range(n_classes)}

    print(f"\nTwo-stage pipeline results:")
    print(f"  GT counts: B1={all_gt_counts[0]}, B2={all_gt_counts[1]}, B3={all_gt_counts[2]}, B4={all_gt_counts[3]}")
    print(f"  AP50 per class:")
    for cls, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {per_class_ap50[cls]:.4f}")
    print(f"  mAP50: {map50:.6f}")
    print(f"  AP50-95 per class:")
    for cls, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {per_class_ap50_95[cls]:.4f}")
    print(f"  mAP50-95 (COCO, 10 thresholds): {map50_95:.6f}")

    return map50, map50_95, per_class_ap50


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
    map50, map50_95, per_class_ap50 = evaluate_pipeline(
        detector, classifier, val_img_dir, val_lbl_dir, iou_thresholds
    )

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"  val_map50:        {map50:.6f}")
    print(f"  val_map50_95:     {map50_95:.6f}")
    print(f"  Current best baseline: 0.269424")
    print(f"  Delta: {map50_95 - 0.269424:+.6f}")
    print(f"{'='*50}")

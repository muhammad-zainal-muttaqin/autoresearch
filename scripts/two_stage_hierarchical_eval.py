"""Evaluate a hierarchical two-stage pipeline with a single-class detector."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare import CLASS_NAMES, DATA_YAML
from stage2_models import load_stage2_classifier
from two_stage_eval_v2 import IOU_THRESHOLDS, compute_ap_101point, iou_single, read_gt_labels


DETECTOR_PATH = Path("/workspace/autoresearch/stage1_detector.pt")
COARSE_CLASSIFIER_PATH = Path("/workspace/autoresearch/stage2_hier_coarse3_dinov2_classifier.pth")
BINARY_CLASSIFIER_PATH = Path("/workspace/autoresearch/stage2_hier_b23_dinov2_classifier.pth")

DETECTOR_CONF = 0.1
DETECTOR_IOU = 0.5
IMGSZ = 640
PAD_RATIO = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_classifier_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def expand_hierarchical_probs(coarse_probs: torch.Tensor, binary_probs: torch.Tensor) -> torch.Tensor:
    final_probs = torch.zeros((coarse_probs.shape[0], 4), device=coarse_probs.device)
    final_probs[:, 0] = coarse_probs[:, 0]
    final_probs[:, 1] = coarse_probs[:, 1] * binary_probs[:, 0]
    final_probs[:, 2] = coarse_probs[:, 1] * binary_probs[:, 1]
    final_probs[:, 3] = coarse_probs[:, 2]
    return final_probs / final_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


def validate_classifier_names(class_names: list[str], expected: list[str], label: str) -> None:
    if class_names != expected:
        raise ValueError(f"{label} class_names mismatch: {class_names} vs expected {expected}")


def evaluate_pipeline(
    detector,
    coarse_loaded,
    binary_loaded,
    val_img_dir: Path,
    val_lbl_dir: Path,
    *,
    detector_conf: float,
    detector_iou: float,
    imgsz: int,
    pad_ratio: float,
):
    clf_tf = build_classifier_transform()
    n_iou = len(IOU_THRESHOLDS)
    all_predictions = {c: {t: [] for t in range(n_iou)} for c in range(4)}
    all_gt_counts = {c: 0 for c in range(4)}

    validate_classifier_names(coarse_loaded.class_names, ["B1", "B23", "B4"], "Coarse classifier")
    validate_classifier_names(binary_loaded.class_names, ["B2", "B3"], "Binary classifier")

    img_paths = sorted([p for p in val_img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"  Evaluating on {len(img_paths)} val images...")

    for idx, img_path in enumerate(img_paths):
        if (idx + 1) % 100 == 0:
            print(f"    {idx + 1}/{len(img_paths)}...")

        gt_boxes = read_gt_labels(val_lbl_dir, img_path.stem)
        for gt_box in gt_boxes:
            all_gt_counts[gt_box[0]] += 1

        results = detector.predict(
            str(img_path),
            imgsz=imgsz,
            conf=detector_conf,
            iou=detector_iou,
            verbose=False,
        )
        if not results or len(results[0].boxes) == 0:
            continue

        img = Image.open(str(img_path)).convert("RGB")
        img_w, img_h = img.size
        crops = []
        det_confs = []
        det_boxes_xyxy = []
        for det in results[0].boxes:
            conf_score = float(det.conf[0])
            x1, y1, x2, y2 = det.xyxy[0].tolist()
            w_box, h_box = x2 - x1, y2 - y1
            cx1 = max(0, int(x1 - w_box * pad_ratio))
            cy1 = max(0, int(y1 - h_box * pad_ratio))
            cx2 = min(img_w, int(x2 + w_box * pad_ratio))
            cy2 = min(img_h, int(y2 + h_box * pad_ratio))
            if cx2 - cx1 < 5 or cy2 - cy1 < 5:
                continue
            crops.append(clf_tf(img.crop((cx1, cy1, cx2, cy2))))
            det_confs.append(conf_score)
            det_boxes_xyxy.append([x1, y1, x2, y2])

        if not crops:
            continue

        crop_batch = torch.stack(crops).to(DEVICE)
        with torch.no_grad():
            coarse_logits = coarse_loaded.model(crop_batch)
            binary_logits = binary_loaded.model(crop_batch)
            coarse_probs = torch.softmax(coarse_logits, dim=1)
            binary_probs = torch.softmax(binary_logits, dim=1)
            final_probs = expand_hierarchical_probs(coarse_probs, binary_probs)
            pred_classes = final_probs.argmax(dim=1).cpu().numpy()
            pred_conf_cls = final_probs.max(dim=1).values.cpu().numpy()

        combined_confs = [dc * pc for dc, pc in zip(det_confs, pred_conf_cls)]
        gt_boxes_xyxy = []
        for gt_cls, cx, cy, bw, bh in gt_boxes:
            gt_boxes_xyxy.append(
                (
                    gt_cls,
                    [
                        (cx - bw / 2) * img_w,
                        (cy - bh / 2) * img_h,
                        (cx + bw / 2) * img_w,
                        (cy + bh / 2) * img_h,
                    ],
                )
            )

        for t_idx, iou_thresh in enumerate(IOU_THRESHOLDS):
            matched_gt = set()
            for pred_cls, comb_conf, det_box in zip(pred_classes, combined_confs, det_boxes_xyxy):
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, (gt_cls, gt_box) in enumerate(gt_boxes_xyxy):
                    if gt_cls != pred_cls:
                        continue
                    iou_val = iou_single(det_box, gt_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = gt_idx
                is_tp = best_iou >= iou_thresh and best_gt_idx not in matched_gt
                if is_tp:
                    matched_gt.add(best_gt_idx)
                all_predictions[pred_cls][t_idx].append((comb_conf, is_tp))

    per_class_ap = {c: [] for c in range(4)}
    for cls_idx in range(4):
        n_gt = all_gt_counts[cls_idx]
        for t_idx in range(n_iou):
            preds = sorted(all_predictions[cls_idx][t_idx], key=lambda x: x[0], reverse=True)
            if not preds or n_gt == 0:
                per_class_ap[cls_idx].append(0.0)
                continue
            tp_cumsum = 0
            fp_cumsum = 0
            recalls = []
            precisions = []
            for _, is_tp in preds:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                recalls.append(tp_cumsum / n_gt)
                precisions.append(tp_cumsum / max(tp_cumsum + fp_cumsum, 1))
            per_class_ap[cls_idx].append(
                compute_ap_101point(np.array(recalls), np.array(precisions), n_gt)
            )

    map50 = np.mean([per_class_ap[c][0] for c in range(4)])
    map50_95 = np.mean([np.mean(per_class_ap[c]) for c in range(4)])

    print("\nHierarchical two-stage results:")
    print(
        "  GT counts: "
        + " ".join(f"{name}={all_gt_counts[idx]}" for idx, name in enumerate(CLASS_NAMES))
    )
    for cls_idx, name in enumerate(CLASS_NAMES):
        ap_curve = per_class_ap[cls_idx]
        print(f"  {name}: AP50={ap_curve[0]:.4f}, mean_AP50-95={np.mean(ap_curve):.4f}")
    print(f"  mAP50:    {map50:.6f}")
    print(f"  mAP50-95: {map50_95:.6f}")
    print(f"  Baseline (YOLO11l end-to-end): 0.269424")
    print(f"  Delta vs baseline: {map50_95 - 0.269424:+.6f}")

    return map50, map50_95, per_class_ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-path", default=str(DETECTOR_PATH))
    parser.add_argument("--coarse-classifier-path", default=str(COARSE_CLASSIFIER_PATH))
    parser.add_argument("--binary-classifier-path", default=str(BINARY_CLASSIFIER_PATH))
    parser.add_argument("--det-conf", type=float, default=DETECTOR_CONF)
    parser.add_argument("--det-iou", type=float, default=DETECTOR_IOU)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--pad-ratio", type=float, default=PAD_RATIO)
    args = parser.parse_args()

    os.chdir(DATA_YAML.parent)
    print("Loading detector...")
    detector = YOLO(str(Path(args.detector_path)))

    print(f"Loading coarse classifier: {args.coarse_classifier_path}")
    coarse_loaded = load_stage2_classifier(Path(args.coarse_classifier_path), DEVICE)
    print(f"  type={coarse_loaded.classifier_type}, classes={coarse_loaded.class_names}")

    print(f"Loading binary classifier: {args.binary_classifier_path}")
    binary_loaded = load_stage2_classifier(Path(args.binary_classifier_path), DEVICE)
    print(f"  type={binary_loaded.classifier_type}, classes={binary_loaded.class_names}")

    val_img_dir = DATA_YAML.parent / "images" / "val"
    val_lbl_dir = DATA_YAML.parent / "labels" / "val"
    print("\nRunning hierarchical evaluation...")
    map50, map50_95, _ = evaluate_pipeline(
        detector,
        coarse_loaded,
        binary_loaded,
        val_img_dir,
        val_lbl_dir,
        detector_conf=args.det_conf,
        detector_iou=args.det_iou,
        imgsz=args.imgsz,
        pad_ratio=args.pad_ratio,
    )
    print(f"\nFINAL: mAP50={map50:.6f}, mAP50-95={map50_95:.6f}")


if __name__ == "__main__":
    main()

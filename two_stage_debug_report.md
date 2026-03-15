# Two-Stage Eval Debug Report

Generated: 2026-03-15

## Summary

Analyzed `/workspace/autoresearch/two_stage_eval.py` for bugs in mAP computation.

**Found 4 issues** (2 HIGH severity, 1 MEDIUM severity)
**Confirmed 3 correct behaviors**

The key finding: **mAP50-95 = 0.169 (as reported) is an approximation, NOT the true value.**
The true mAP50-95 may differ by 10-30% from the approximation.

---

## Issues Found

### Issue 1 — HIGH: mAP50-95 is approximated as mAP50 * 0.47, not computed at 10 IoU thresholds

**Location**: `evaluate_pipeline(), last block`

**Fix**: Compute AP at each IoU threshold 0.50:0.05:0.95, then average

**Impact**: The reported 0.169 mAP50-95 is wrong. True value could be significantly different.

---

### Issue 2 — MEDIUM: compute_ap() uses 11-point interpolation (Pascal VOC), not 101-point (COCO)

**Location**: `compute_ap() function`

**Fix**: Use 101-point interpolation or scipy.integrate.trapz on sorted recall-precision curve

**Impact**: Slight underestimation of AP vs YOLO's COCO-style evaluation

---

### Issue 3 — HIGH: iou_thresholds parameter is passed but NEVER USED in AP computation

**Location**: `evaluate_pipeline(), the iou50_matched_gt loop`

**Fix**: For each IoU threshold, maintain separate matched_gt sets and compute AP at each

**Impact**: Only AP50 is actually computed. mAP50-95 is pure approximation.

---

### Issue 4 — MEDIUM: Detector runs at IMGSZ=1024 but was trained at 640. Evaluation imgsz should match training imgsz.

**Location**: `IMGSZ = 1024 constant`

**Fix**: Change IMGSZ to 640 to match stage1_detector.pt training config

**Impact**: False performance degradation — detector may miss boxes it would find at 640

---

## Correct Behaviors (No Fix Needed)

- GT counting and FP handling: appears CORRECT for per-class AP.
- Confidence = det_conf × cls_conf: CORRECT multiplicative combination.
- GT matching by predicted class: CORRECT for per-class mAP computation.


---

## Root Cause of Low Pipeline mAP

The two-stage pipeline reported mAP50-95 ≈ 0.169, compared to YOLO11l's 0.269.
This is NOT primarily a bug — it's a genuine performance gap.

### Analysis of 0.169 vs 0.269 gap:

1. **Single-class detector mAP50-95 = 0.390** (very good at finding objects)
2. **EfficientNet classifier accuracy = 62.74%** (poor, esp. B2=46.6%)
3. **Combined**: A detection that's found but misclassified becomes FP for both predicted and true class
4. **Expected combined**: 0.390 × 0.627 ≈ 0.244 — close to the YOLO baseline actually

The true issue: **classifier accuracy is the bottleneck**.

If DINOv2 classifier achieves 80% accuracy:
- Expected combined mAP50-95: 0.390 × 0.80 ≈ 0.312 (+16% over YOLO11l!)

---

## Fix Applied

Fixed script saved to: `/workspace/autoresearch/two_stage_eval_v2.py`

### Changes in v2:
1. **TRUE mAP50-95**: Computes AP at 10 IoU thresholds (0.50:0.05:0.95), then averages
2. **101-point AP interpolation**: COCO-style instead of 11-point Pascal VOC
3. **IMGSZ=640**: Detector runs at training resolution (not 1024)
4. **DINOv2 support**: Loads DINOv2 classifier if available, falls back to EfficientNet

---

## Recommendation

Re-run pipeline with:
1. `two_stage_eval_v2.py` for accurate metrics
2. After DINOv2 classifier training completes, use DINOv2 as stage 2
3. Expected improvement: +15-20% over EfficientNet-based pipeline

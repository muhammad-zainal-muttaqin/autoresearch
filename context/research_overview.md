# Research Overview

## Objective

Maximize mAP@0.5 for 4-class oil palm fruit bunch (FFB) detection: B1, B2, B3, B4.

## Current State

- Main baseline: legacy best 4-class run (mAP@0.5 ~ 0.55, mAP@0.5-0.95 ~ 0.269)
- Main ceiling: B2/B3 discrimination — these classes overlap heavily in bounding box size and visual appearance
- Framework: Ultralytics YOLO (yolo11s/yolo11l/yolov9c)
- Canonical split: train 2764, val 604, test 624

## Exploration Priorities

Seed ideas (E1-E4):
1. E1: size features — inject bounding box size as an auxiliary signal
2. E2: position features — use spatial position in the image
3. E3: size + position combined
4. E4: texture features — leverage texture differences between maturity stages

Beyond E1-E4 (when seed ideas are exhausted):
- Contrastive pre-training on FFB crops
- Attention mechanisms for occlusion handling
- Multi-view-inspired augmentation
- Learned augmentation policies
- Maturity-aware anchor design
- Curriculum learning
- Synthetic occlusion augmentation

## Out of Scope

Techniques already in E0 (focal loss, ordinal classification, copy-paste, two-stage, confidence thresholds) are out of scope until the RA completes the relevant phase and results appear in `e0_results.md`.

## Closed Branches

- Long brute-force one-stage training
- Crop-only two-stage pipelines
- Tiled training as previously implemented
- Label smoothing as a standalone fix
- Small knob-turning on already closed branches

## Batch Intent

- Prefer one falsifiable experiment at a time
- Do not reopen closed branches by retuning knobs
- Use modeling.py/pipeline.py only when the experiment is structurally different from plain train.py tuning

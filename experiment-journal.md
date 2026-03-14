# Experiment Journal — Autoresearch TBS Detection

## Experiment 1 — 2026-03-14 (imgsz 1024 batch 8)

**Hypothesis**: If I increase image resolution from 640 to 1024 with batch=8, then val_map50_95 will improve because objects (especially small B4) will be more visible at higher resolution, allowing better feature extraction.

**Change**: IMGSZ=640→1024, BATCH=16→8

**Result**: val_map50_95=0.259880 (delta=+0.002132 from previous best 0.257748) — KEEP

**Per-class** (from run.log):
- B1: mAP50-95=0.441 (excellent, high precision at 0.583, recall 0.884)
- B2: mAP50-95=0.197 (very poor - B2/B3 confusion documented)
- B3: mAP50-95=0.264 (moderate)
- B4: mAP50-95=0.140 (worst - small object problem)

**Analysis**: Higher resolution helps marginally overall (+0.002). B4 remains worst class at 0.140 mAP50-95. B2 confusion is severe at 0.197. The gap between B1 (0.441) and B4 (0.140) is 3x — huge class performance gap. The ceiling of ~0.26 overall is dragged down by B2 (0.197) and B4 (0.140). Even if B1 and B3 were perfect, we'd still be limited.

**Next**: Try class-balanced dataset to address B1/B4 underrepresentation. B3 has 5634 instances vs B1=1540, B4=2343. Oversample B1 and B4 images to balance training distribution.

---

## Experiment 2 — 2026-03-14 (imgsz 1024 BOX 10 CLS 1.5 DFL 2.0)

**Hypothesis**: If I increase BOX loss weight to 10.0, CLS to 1.5, and DFL to 2.0 while keeping imgsz=1024, the model will pay more attention to precise box localization and classification discrimination, helping B4 and B2.

**Change**: BOX=7.5→10.0, CLS=0.5→1.5, DFL=1.5→2.0

**Result**: val_map50_95=0.256593 (delta=-0.003287 from best 0.259880) — DISCARD

**Analysis**: Increasing all loss weights simultaneously hurt performance. The model may have struggled to balance the higher gradients, leading to less stable training. This combined change was too aggressive without a clear mechanism. Not repeating aggressive multi-loss changes.

**Next**: Based on per-class analysis from Experiment 1:
- Root cause 1: B2/B3 confusion (B2 only 0.197 mAP50-95)
- Root cause 2: B4 small object (B4 only 0.140 mAP50-95)
- Strategy: Build class-balanced dataset (oversample B1/B4 images)

---

## Experiment 3 — 2026-03-14 (class-balanced dataset, B1/B4 oversampled)

**Hypothesis**: If B1 and B4 images are oversampled 2-3x in training data to balance the class distribution (currently B3 has 5634 instances vs B1=1540, B4=2343), then B4 and B1 mAP50-95 will increase because the model sees proportionally more B4/B1 examples per epoch, reducing the B3-dominant bias.

**Change**: Create Dataset-Balanced/ by copying all images and adding 2x copies of images containing B1 or B4 (with horizontal flip augmentation to add diversity). Train on this balanced dataset.

**Expected success criterion**: val_map50_95 > 0.262 OR B4 mAP50-95 > 0.18

**Falsification**: If B4 mAP doesn't improve despite more B4 training data, then the bottleneck is not data quantity but rather B4's inherent difficulty (small size, low contrast). In that case, try tiled dataset.


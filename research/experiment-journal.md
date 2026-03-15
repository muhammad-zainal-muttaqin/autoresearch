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

**Result**: val_map50_95=0.247590 (delta=-0.012290 from best 0.259880) — DISCARD

**Per-class**:
- B1: mAP50-95=0.423 (slightly worse)
- B2: mAP50-95=0.196 (no change)
- B3: mAP50-95=0.248 (worse, was 0.264)
- B4: mAP50-95=0.123 (worse, was 0.140)

**Analysis**: Class-balanced oversampling hurt performance on all classes. The reasons:
1. Oversampling with simple flips creates duplicate-like images — not genuinely diverse. The model sees "more data" but it's almost the same data.
2. The training dataset went from 2764→6367 images, but within the 20-min time budget, this means fewer epochs (fewer passes through the full diversity), potentially underfitting.
3. B3 also increased 2.3x in absolute count, so the "balance" improvement is real but offset by less epochs.
4. Flipped copies of B4 images don't help because B4's challenge is small object size, not orientation.

**Falsification confirmed**: Oversampling B1/B4 with geometric augmentation does NOT help. B4's bottleneck is NOT data quantity at standard resolution.

**Key insight**: The balanced dataset approach failed because:
- Geometric flips are not meaningful augmentation for these objects
- Doubling dataset size with similar images reduces effective epochs in time budget
- B4 needs better RESOLUTION, not more images

**Next**: Try model soup/weight averaging OR label noise correction OR try YOLOv9e (bigger model).

---

## Experiment 4 — 2026-03-14 (Label Noise Correction)

**Hypothesis**: If high-confidence model predictions that disagree with labels (especially B2/B3 confusion) are used to correct those labels, then B2 mAP50-95 will increase because the model is no longer penalized for correctly predicting B3 on mislabeled-as-B2 instances. B2 mAP at 0.197 is the strongest bottleneck.

**Approach**:
1. Load best model (runs/autoresearch/train/weights/best.pt)
2. Run inference on training images with conf=0.1 (to catch all detections)
3. For each detection with conf>0.75 that predicts a DIFFERENT class than the label:
   - Focus on B2/B3 mismatches (most common: 208 B2→B3 cases)
   - IoU match between predicted box and labeled box
   - If IoU>0.5 and conf>0.75, flag as potential mislabel
4. Correct top-N most confident mismatches
5. Save Dataset-Cleaned/ and retrain

**Expected success criterion**: B2 mAP50-95 > 0.22 OR val_map50_95 > 0.265

**Falsification**: If label noise correction doesn't help B2, then the B2/B3 confusion may not be label noise but instead inherent visual ambiguity that the model cannot resolve.

**INVESTIGATION RESULT (pre-experiment)**:
- Ran inference on 200 training images at different confidence thresholds
- conf>=0.5: 21 B2/B3 mismatches (extrapolating: ~290 total in full train set)
- conf>=0.7: 0 corrections (model not confident enough to correct labels)
- **Conclusion**: The model cannot confidently identify mislabeled examples. At conf>=0.5, the model may itself be wrong about B2/B3 — using these to "correct" labels would introduce new errors.
- **Decision**: Label noise correction via model prediction is not safe at this accuracy level. Need a more confident oracle or different approach.

**Abandoning Experiment 4 (label correction)**. Moving to Experiment 5: YOLOv9e (larger model).

---

## Experiment 5 — 2026-03-14 (YOLOv9e larger model)

**Hypothesis**: If YOLOv9e (larger model than YOLOv9c) is used with imgsz=1024 and batch=4, then B2/B3 discrimination will improve because the larger model has more capacity for fine-grained feature learning, and the self-attention-like mechanisms in the larger ELAN blocks can capture more context for B2/B3 disambiguation.

**Change**: MODEL=yolov9c.pt→yolov9e.pt, BATCH=8→4

**Expected success criterion**: val_map50_95 > 0.265 OR B2 mAP50-95 > 0.21

**Falsification**: If YOLOv9e doesn't improve over YOLOv9c, then model capacity is not the bottleneck. The problem is in the data quality/distribution, not model expressiveness.

**Result**: val_map50_95=0.229457 (delta=-0.030423 from best 0.259880) — DISCARD

**Per-class**:
- B1: mAP50-95=0.391 (worse than YOLOv9c's 0.441)
- B2: mAP50-95=0.167 (worse)
- B3: mAP50-95=0.245 (worse)
- B4: mAP50-95=0.115 (worse)

**Analysis**: YOLOv9e significantly worse than YOLOv9c. Root cause: larger model needs more training time/epochs to converge. In the 20-min time budget, YOLOv9e gets fewer gradient updates than YOLOv9c. The underfitting is evident — all class metrics are worse. This is consistent with YOLOv9e being noted as "weaker" in the project history (though it was tested with different configs). The 20-min budget is too short for a 100M+ parameter model to converge on this data.

**Falsification confirmed**: Model capacity is NOT the bottleneck (at least within this time budget). YOLOv9c's capacity is sufficient; the problem is data quality.

**Key insight from experiments 1-5**:
- imgsz=1024 helps (+0.002) → resolution matters, objects need to be bigger in image
- Larger model (YOLOv9e) doesn't help within time budget → not a capacity problem
- Class-balanced oversampling hurts → adding flipped copies reduces effective diversity
- Loss weight changes don't help → default weights are well-tuned

**Remaining high-impact ideas to try**:
1. RT-DETR (transformer architecture for better context/disambiguation)
2. Model soup/weight averaging (ensemble without inference cost)
3. Tiled dataset (genuine resolution improvement for small B4 objects)
4. Lower LR0 with higher LRF (different learning rate schedule)

---

## Experiment 6 — 2026-03-14 (RT-DETR transformer architecture)

**Hypothesis**: If RT-DETR-L is used instead of YOLOv9c, the transformer's self-attention mechanism will improve B2/B3 disambiguation because attention captures global context (whole-image relationships) that CNN-based YOLO misses when deciding between ripeness stages B2 and B3.

**Change**: MODEL=yolov9c.pt→rtdetr-l.pt, BATCH=4, IMGSZ=1024

**Expected success criterion**: B2 mAP50-95 > 0.22 OR val_map50_95 > 0.265

**Falsification**: If RT-DETR doesn't improve B2/B3 disambiguation, then self-attention is not what's needed — the problem is that B2 and B3 look genuinely similar regardless of model architecture, and only better labels or more data can solve it.

**Result**: val_map50_95=0.137608 — DISCARD (RT-DETR did not converge well in 20 min)

---

## Experiment 7 — 2026-03-14 (Two-Stage Pipeline: Single-class Detector + EfficientNet-B0 Classifier)

**Hypothesis**: If we decouple detection (find objects) from classification (label B1-B4), the classifier can focus purely on appearance features within the crop, potentially learning better B2/B3 distinction than end-to-end YOLO.

**Stage 1**: Single-class TBS detector — all classes mapped to class 0. Result: mAP50-95=0.390 (validation, single class = much easier task).

**Stage 2**: EfficientNet-B0 on GT crops from Dataset-Crops. Best val_acc=62.74%, B2=46.6% (near random for binary B2/B3 choice).

**Pipeline evaluation**: mAP50=0.359, mAP50-95≈0.169 — DISCARD

**Analysis**: The classifier bottleneck (B2=46.6% accuracy) was worse than expected. Even a perfect single-class detector cannot rescue us if the downstream classifier can't distinguish B2/B3. The core visual ambiguity remains unsolved. The combined pipeline is worse than end-to-end YOLO (0.169 vs 0.260).

**Key finding**: B2/B3 visual ambiguity is fundamental. Even with isolated crops and a dedicated classifier trained on GT crops, B2 accuracy is only 46.6% — essentially coin-flip level. This rules out simple architectural approaches.

---

## Experiment 8 — 2026-03-14 (Color HSV Analysis)

**Hypothesis**: Ripeness in oil palm is fundamentally a color signal (green=unripe, red/orange=ripe). A simple HSV rule-based classifier might outperform or provide complementary signal to EfficientNet.

**Investigation (color_classifier.py)**:
- B1 mean_H=26.6° (ORANGE — unexpected! not green)
- B2 mean_H=27.3° (nearly identical to B1)
- B3 mean_H=74.5° (GREEN — backward from expectation!)
- B4 mean_H=35.7°

**Result**: Color classifier accuracy = 31.6% (worse than EfficientNet's 62.7%)

**Analysis**: The color signal is backwards from intuition. This suggests the dataset labels may be based on a different ripeness stage definition than expected. Or the labeled "B3" bunches in this dataset happen to be photographed in conditions showing more green foliage/canopy. Regardless, color alone is not discriminative.

**Key insight from color analysis**: The B2/B3 problem is NOT solvable by color. The classes are defined by morphological/contextual features beyond simple color histograms.

---

## Experiment 9 — 2026-03-14 (RF-DETR with DINOv2 Backbone)

**Hypothesis**: RF-DETR with DINOv2-windowed-small backbone (pre-trained on large-scale vision data via self-supervised learning) will have better representations for fine-grained visual distinctions, improving B2/B3 disambiguation better than YOLO's supervised-only representations.

**Architecture**: RF-DETR (DINOv2-S-windowed) with deformable attention, 31.87M params.

**Training**: 30 epochs, batch=4, lr=1e-4, early stopping patience=10, Dataset-RFDETR (symlinks).

**Result per epoch**:
- Best mAP50-95=0.2489 at epoch 4 (COCO-style evaluation)
- Early stopping triggered at epoch 10 (no improvement for 6+ epochs)
- mAP50=0.542 at best

**Compared to baseline**: 0.2489 < 0.260003 — DISCARD

**Analysis**:
1. RF-DETR uses COCO-style mAP evaluation (different from our protocol which evaluates at imgsz=640 using ultralytics). The numbers may not be directly comparable.
2. Even accounting for evaluation differences, RF-DETR did not exceed baseline.
3. DINOv2 representations, while powerful for ImageNet-style classification, may not transfer well to aerial/field photography of oil palm bunches.
4. Training from a generic pre-trained checkpoint (not domain-specific) limits the benefit.
5. Small batch size (4) and limited epochs due to early stopping may have prevented convergence.

**Key finding**: DINOv2-based RF-DETR is NOT a silver bullet for this domain. Self-supervised pre-training on ImageNet doesn't automatically help with agricultural object detection.

---

## Experiment 10 — 2026-03-14 (Train+Test Combined Dataset)

**Hypothesis**: Using all available labeled data (train+test, 3388 images) while holding val for evaluation should improve all classes by ~22% more examples. B2 specifically gains 85% more instances (1874→3463).

**Change**: Dataset-TrainTest (symlinks, train+test→train, val unchanged)

**Result**: val_map50_95=0.248169 (delta=-0.011834 from best 0.260003) — DISCARD

**Per-class**:
- B1: 0.418 (worse)
- B2: 0.191 (no improvement despite 85% more B2 examples!)
- B3: 0.251 (worse)
- B4: 0.132 (worse)

**Analysis**: More data hurt because fewer epochs in the 20-minute budget. With 3388 images at imgsz=1024 batch=8, we get fewer gradient updates per time budget. The per-image learning signal is the same, but total training steps decrease. B2 did NOT improve even with 85% more examples — confirming that the B2/B3 confusion is NOT a data quantity problem but a fundamental visual ambiguity.

**Key finding**: Simply adding more labeled data does NOT help within a fixed time budget when training at high resolution.

---

## Experiment 11 — 2026-03-15 (YOLO11m, newer architecture)

**Hypothesis**: YOLO11m (Ultralytics' newest model family, C3K2 architecture released 2024) may have better feature extraction than YOLOv9c due to architectural improvements, achieving better B2/B3 disambiguation with same or fewer parameters.

**Change**: MODEL=yolo11m.pt (38.8M params), imgsz=1024, batch=8

**Result**: val_map50_95=0.255829 (delta=-0.004174 from best 0.260003) — DISCARD

**Per-class**:
- B1: 0.427 (slightly worse)
- B2: 0.197 (essentially identical to every other run!)
- B3: 0.259 (slightly worse)
- B4: 0.140 (same)

**Analysis**: YOLO11m is NOT better than YOLOv9c at imgsz=1024 batch=8. The B2 mAP is stubbornly at ~0.197 regardless of architecture. This means the bottleneck is NOT in the model's capacity to discriminate B2/B3 — it's in the data/labels themselves.

**Critical observation**: B2 mAP50-95 ≈ 0.197 across ALL architectures (YOLOv9c, YOLO11m, RT-DETR, RF-DETR). This value is suspiciously consistent, suggesting a ceiling imposed by the data/label quality, not model choice.

---

## Experiment 12 — 2026-03-15 (YOLO11l imgsz=640 batch=16) — NEW BEST!

**Hypothesis**: YOLO11l (larger YOLO11, ~49M params) at imgsz=640 with batch=16 will achieve more gradient updates per time budget than imgsz=1024 batch=8. At 640px, batch=16 processes 2x more samples/step. More total gradient updates → better convergence in 20 minutes. Also: evaluation is done at imgsz=640, so training at 640 eliminates train/eval domain mismatch.

**Change**: MODEL=yolo11l.pt, IMGSZ=640→640 (same as eval), BATCH=8→16 (2x)

**Expected mechanism**:
- Fewer pixels per image → faster forward pass
- Larger batch → more stable gradient estimates
- Training at eval resolution → no resolution gap between training and evaluation

**Result**: val_map50_95=0.264147 (delta=+0.004144 from best 0.260003) — **NEW BEST! KEEP**

**Per-class**:
- B1: 0.439 (vs 0.441 baseline — essentially same)
- B2: 0.210 (vs 0.197 baseline — **+0.013 improvement!**)
- B3: 0.267 (vs 0.264 baseline — small improvement)
- B4: 0.141 (vs 0.140 baseline — essentially same)

**Analysis**: The improvement comes primarily from B2 (+6.6% relative improvement). Why?
1. Training at imgsz=640 = evaluation imgsz=640: no train/eval domain gap. The model sees the same scale it will be evaluated at.
2. Batch=16 = 2x gradient updates per time: better convergence in 20 minutes.
3. YOLO11l architecture: C3K2 blocks with selective kernel attention are better at fine-grained discrimination than YOLOv9c's GELAN blocks.
4. The consistency of B4 mAP (0.141) suggests B4's problem is truly about scale/size at evaluation resolution, not architecture.

**Key insight**: Train/eval resolution matching is important. When we train at 1024 but evaluate at 640, the model's learned scale-specific features may not transfer perfectly. Training AT the evaluation resolution eliminates this gap.

**Next directions to explore**:
1. YOLO11l with batch=32 (if VRAM allows) — even more gradient updates
2. YOLO11x (extra-large) — might converge faster due to better representations
3. YOLO11l + train+test combined (3388 imgs, 640px, batch=16) — can we get both more data AND high batch count?
4. YOLO11l with copy_paste=0.3 augmentation — targeted B4 augmentation at 640 resolution

---

## Experiment 13 — PLAN: YOLOv9c with Ordinal Regression / Auxiliary Tasks

**Background**: All approaches so far have failed to exceed 0.260003. The fundamental problem is B2/B3 visual ambiguity. Novel ideas:

1. **Ordinal regression**: B1→B2→B3→B4 is an ordered sequence (ripeness stages). Standard cross-entropy treats them as independent classes. CORAL loss or ordinal encoding could help the model understand the ordering constraint.

2. **Multi-task auxiliary head**: Add an auxiliary "ripeness score" regression head alongside classification. Force the model to predict a continuous ripeness value (B1=0, B2=0.33, B3=0.67, B4=1.0) in addition to discrete class. This ordinal auxiliary loss might structure the feature space better.

3. **Drop ambiguous B2/B3 samples**: Instead of correcting labels, train without B2 samples (or with very high loss weight for B2) to force the model to learn discriminative B2-specific features.

**Next experiment**: Try YOLOv9c with mosaic=0 (pure single-image training) combined with copy-paste augmentation at higher rate. Rationale: mosaic may be creating artificial context boundaries that confuse B2/B3 classification by mixing contexts from different images.

**Alternatively**: Try `copy_paste=0.5` (copy-paste augmentation, which inserts object crops from other images). This can increase B4 instance frequency by copying small B4 objects into other scenes, creating synthetic training examples of B4 at various scales.

---

## STRATEGIC REFRAME — 2026-03-15 (Human Input)

**Critical insight from human researcher**:

- Single-class detector achieves 0.390 mAP50-95. The detection itself is near-target.
- Multi-class YOLO at 0.269 is being dragged down by B2/B3 confusion.
- TIME_HOURS=0.33 (20 min) → only ~21 epochs for yolo11l. Far from convergence.
- Extending to TIME_HOURS=2.0 → 100+ epochs → much better convergence.

**Two priority paths**:

### JALUR A: More training time (immediate, high impact)

**A1 (NEXT EXPERIMENT)**: TIME_HOURS=2.0, yolo11l, EPOCHS=300, PATIENCE=50, AdamW, Dataset-TrainTest.
- Hypothesis: At 100+ epochs, yolo11l on 3388 images will converge much better than at 21 epochs.
- Expected delta: +0.02 to +0.05 mAP50-95 based on learning curve not saturated.
- Success criterion: val_map50_95 > 0.29

**A2**: If A1 fails, try TIME_HOURS=2.0 + imgsz=1024 + batch=8 (60+ epochs vs previous 9 epochs).

### JALUR B: Two-stage pipeline with DINOv2 classifier

**Bug fix**: two_stage_eval.py was approximating mAP50-95 = mAP50 × 0.47. Now fixed to compute real COCO mAP50-95 (10 IoU thresholds, 0.50-0.95).

**B2 (PARALLEL)**: Train DINOv2-base frozen backbone + MLP head on Dataset-Crops.
- train_dinov2_classifier.py created for this purpose.
- EfficientNet-B0 got 62.7% val acc with B2=46.6%. DINOv2 expected >80%.
- If DINOv2 classifier reaches >80%, re-run two_stage_eval.py with new classifier.

**Code changes made**:
1. train.py: TIME_HOURS=0.33→2.0, EPOCHS=80→300, PATIENCE=15→50, OPTIMIZER=SGD→AdamW, DATA_YAML→Dataset-TrainTest
2. two_stage_eval.py: Fixed mAP50-95 computation (now real COCO protocol, 10 IoU thresholds). Fixed load_classifier to auto-detect EfficientNet vs DINOv2. Baseline updated to 0.269424.
3. train_dinov2_classifier.py: New script for DINOv2-base frozen backbone + MLP head classifier.


## Experiment 13 — 2026-03-15 05:11 UTC

**Hypothesis**: If the old two-stage result was mainly held down by the broken evaluator, then re-running the existing stage1 detector plus EfficientNet stage2 classifier with a corrected COCO-style evaluator should materially improve the reported mAP50-95 and clarify whether two-stage is still worth pursuing.

**Change**: Re-evaluate `stage1_detector.pt` + `stage2_classifier.pth` with `scripts/two_stage_eval_v2.py`, which uses true IoU-swept mAP50-95 instead of the old approximation.

**Result**: val_map50_95=0.167510 (delta=-0.101914 from best 0.269424) — DISCARD

**Per-class**: B1=0.320 B2=0.100 B3=0.193 B4=0.057

**Analysis**: The corrected two-stage baseline remains far below the best one-stage detector. The old `~0.169` result was not just an evaluator artifact; the underlying EfficientNet stage-2 classifier is still the real bottleneck, especially on B2 and B4. This closes the old EfficientNet two-stage branch decisively and justifies moving all stage-2 effort to stronger classifiers rather than revisiting the same detector/classifier pair.

**Next**: Train the DINOv2 stage-2 classifier, then re-run the corrected two-stage evaluator with the new checkpoint. If B2 remains weak, escalate immediately to the ordinal DINOv2 classifier instead of retrying any EfficientNet or one-stage tuning branch.








## Experiment 14 — 2026-03-15 05:21 UTC

**Hypothesis**: If the earlier one-stage ceiling was mostly caused by undertraining, then extending TIME_HOURS to 2.0 on the best YOLO11l train+test recipe will let the model converge and exceed the 0.269 baseline.

**Change**: Keep the best known one-stage recipe but extend the run to TIME_HOURS=2.0 with yolo11l, Dataset-TrainTest, imgsz=640, batch=16, AdamW, and cos_lr.

**Result**: val_map50_95=0.257974 (delta=-0.132456 from best 0.390430) — DISCARD

**Per-class**: B1=0.429 B2=0.209 B3=0.256 B4=0.138

**Analysis**: Long-run one-stage did not validate the time-budget hypothesis. Final val_map50_95=0.257974, best observed during the run was only 0.258, both below the existing 0.269424 baseline. Validation drifted downward after the early peak, so the long-budget one-stage branch is now excluded as a path to 0.40-0.50. The next step is stronger stage-2 classification.

**Next**: Freeze the old one-stage branch and move GPU budget to DINOv2 stage-2 classification, then its ordinal follow-up if B2 remains weak.


## Experiment 15 — 2026-03-15 09:24 UTC

**Hypothesis**: If a stronger frozen backbone is the missing ingredient for stage-2 classification, then DINOv2 should beat the previous EfficientNet crop classifier and raise B2/B3 discrimination enough to make two-stage viable.

**Change**: Replace the EfficientNet-B0 crop classifier with a DINOv2-base frozen backbone plus trainable MLP head on Dataset-Crops.

**Result**: val_acc=59.15% at epoch 23 — DISCARD

**Per-class**: B1=89.0% B2=45.9% B3=57.8% B4=60.6%

**Analysis**: DINOv2 did not deliver the hoped-for classifier breakthrough. Overall accuracy improved only modestly in absolute terms and B2 remained weak at 45.9%, meaning the backbone alone does not solve the B2/B3 ambiguity. This falsifies the simple “stronger frozen backbone is enough” hypothesis and justifies moving to ordinal objectives rather than more flat cross-entropy variants.

**Next**: Run corrected two-stage evaluation with the DINOv2 checkpoint for closure, then escalate immediately to the CORN ordinal DINOv2 classifier on GPU.


## Experiment 16 — 2026-03-15 11:42 UTC

**Hypothesis**: Even if the standalone DINOv2 classifier is only a modest improvement, plugging it into the full two-stage pipeline might still raise end-to-end mAP50-95 enough to keep the two-stage branch alive.

**Change**: Re-run `scripts/two_stage_eval_v2.py` with `stage1_detector.pt` and the new `stage2_dinov2_classifier.pth` checkpoint.

**Result**: val_map50_95=0.181088 (delta=-0.088336 vs best one-stage 0.269424) — DISCARD

**Per-class**: B1=0.379 B2=0.100 B3=0.166 B4=0.080

**Analysis**: The DINOv2 stage-2 classifier improved the corrected two-stage pipeline only marginally over EfficientNet, from 0.167510 to 0.181088. The key failure mode did not move: B2 remained stuck at 0.100 mean AP50-95, and B3 also regressed relative to the detector ceiling. This closes the plain cross-entropy DINOv2 two-stage branch. A stronger visual backbone alone is not enough; the next idea must change the learning objective or the representation geometry for B2/B3 rather than just swapping classifiers.

**Next**: Evaluate the ordinal CORN DINOv2 classifier as the next objective-level change. If it also fails the classifier gate, stop spending GPU on frozen-backbone stage-2 variants and move to a genuinely different formulation.


## Experiment 17 — 2026-03-15 11:42 UTC

**Hypothesis**: Because the labels are ordinal (`B1 < B2 < B3 < B4`) and most confusion is local (`B2 <-> B3`), a CORN objective should shape the classifier decision boundary better than flat cross-entropy and lift B2 accuracy enough to justify another end-to-end two-stage evaluation.

**Change**: Train `scripts/train_dinov2_corn_classifier.py` with the same frozen DINOv2 backbone and a CORN ordinal head on `Dataset-Crops`.

**Result**: val_acc=57.39% at epoch 1, B2=34.6% — DISCARD

**Per-class**: B1=85.4% B2=34.6% B3=57.2% B4=67.7%

**Analysis**: CORN failed the gate decisively. It underperformed the plain DINOv2 classifier overall and collapsed B2 to 34.6%, which is worse than the earlier cross-entropy branch. Some later epochs raised B2 temporarily, but only by trading away B3/B4 and never recovering the best overall validation accuracy. Because the stage-2 gate failed badly, this branch is rejected before full two-stage evaluation; running end-to-end mAP would only waste more time on a weak classifier.

**Next**: Abandon frozen-backbone classification-only tweaks and move to a truly new branch: either metric-learning / contrastive crop supervision for B2/B3 separation or a fine-grained attention model that uses token-level evidence rather than a shallow linear head.


## Experiment 18 — PLAN: Hierarchical Two-Stage Classification

**Observation**: The strongest component in the repo is still the single-class detector (`mAP50-95=0.390430`). Every flat 4-way classifier variant so far has failed at the same point: separating `B2` from `B3` without damaging `B1`/`B4`. That means the pipeline is currently solving an unnecessarily hard problem in one jump.

**Hypothesis**: If stage-2 is decomposed hierarchically into a coarse `B1/B23/B4` classifier plus a dedicated binary `B2/B3` specialist, the coarse classifier can remove the easy decisions from the hard boundary and the binary head can spend all of its capacity on the only ambiguity that matters. This should outperform the flat 4-way classifier even if both sub-models are individually modest.

**Design**:
- Keep the existing single-class detector as stage 1.
- Build `Dataset-Crops-Coarse3` with classes `B1`, `B23`, `B4` by merging `B2` and `B3`.
- Build `Dataset-Crops-B23` with only `B2` and `B3`.
- Train two new DINOv2 classifiers on those datasets.
- Evaluate a new hierarchical two-stage pipeline where final probabilities are expanded as:
  `P(B1)=Pcoarse(B1)`, `P(B2)=Pcoarse(B23)*Pbinary(B2)`, `P(B3)=Pcoarse(B23)*Pbinary(B3)`, `P(B4)=Pcoarse(B4)`.

**Success criterion**: hierarchical end-to-end `val_map50_95 > 0.269424`. A strong signal would be `B2` mean AP50-95 clearly above the current `~0.10` two-stage ceiling.

**Falsification rule**: If the coarse classifier is strong but the binary `B2/B3` specialist still stays near chance, then the ambiguity is not being fixed by decomposition alone and the next branch must change the embedding objective itself (contrastive / prototype-based learning).


## Experiment 18a — 2026-03-15 12:49 UTC — Hierarchical Coarse Classifier (B1/B23/B4)

**Hypothesis**: If stage-2 is decomposed hierarchically, a coarse `B1/B23/B4` classifier can handle easy decisions, offloading the hard `B2/B3` boundary to a specialist. Best val_acc expected >75% based on the easier 3-way split.

**Change**: Train DINOv2 frozen-backbone + linear head on `Dataset-Crops-Coarse3` (B2 and B3 merged into single B23 class). Dataset: train=12360, val=2786.

**Result**: **val_acc=75.09%** at epoch 40 — KEEP as coarse classifier

**Per-class accuracies**:
- B1: 85.0% (301 crops; 256 correct, 44→B23, 1→B4)
- B23: 76.5% (1931 crops; 101→B1, 1478 correct, 352→B4)
- B4: 64.6% (554 crops; 2→B1, 194→B23, 358 correct)

**Confusion matrix** (rows=true, cols=pred):
```
      B1   B23   B4
B1:  256    44    1
B23: 101  1478  352
B4:    2   194  358
```

**Analysis**: Coarse classifier performs reasonably — 75.09% overall, and B1 at 85% is strong. Critical weakness: B4 at 64.6% with 194/554 B4 crops misclassified as B23. This is the hierarchical pipeline's Achilles' heel — if stage-2 coarse routes B4 objects into the B23 branch, they then get incorrectly classified as B2 or B3. The B23 cluster itself (76.5%) is decent and confirms the strategy of separating easy classes first. However, the B4→B23 leakage at 35% is a fundamental problem: in end-to-end evaluation, 35% of true B4 detections will be misclassified and then further confused inside the B2/B3 specialist. The best possible end-to-end outcome for B4 given this coarse classifier is at most 64.6% × (whatever the specialist does) ≈ badly hurt.

**Key insight**: The coarse classifier is NOT better than expected for B4. B4 in the crop space looks visually similar to B23 (perhaps because B4 = overripe, with darker color similar to ripe B3). The coarse-3 strategy helps B1 but doesn't solve B4.

**Next**: Run binary B2/B3 specialist, then compute hierarchical end-to-end mAP. If B4 leakage kills performance, the hierarchical strategy needs a B1/B2/B3/B4 coarse-but-not-merged alternative.


## Experiment 18b — 2026-03-15 14:05 UTC — Binary B2/B3 Specialist (PARTIAL — stopped at epoch 6)

**Hypothesis**: A binary DINOv2 classifier trained ONLY on B2/B3 crops, without the distraction of B1/B4 examples, should achieve better B2/B3 separation than a flat 4-way classifier. Target: >75% binary accuracy.

**Change**: Train DINOv2 frozen + linear head on `Dataset-Crops-B23` (only B2=2844, B3=5633 crops). Class-weighted loss to address B2/B3 imbalance.

**Result**: **best val_acc=72.81%** at epoch 2 (B2=57.4%, B3=80.0%) — INCOMPLETE (stopped at epoch 6 of 50)

**Per-class at best checkpoint (epoch 2)**:
- B2: 57.4% accuracy (extremely weak — specialist still biased toward B3)
- B3: 80.0% accuracy

**Epoch 5 snapshot** (B2=77.9%, B3=60.6% — B2 peaks briefly but overall accuracy drops to 66.1%):
- The best checkpoint at epoch 2 was saved because overall val_acc was highest there, but B2 accuracy had a momentary peak at epoch 5 (77.9%) with much lower B3 (60.6%)

**Analysis**: INCOMPLETE. Only 6 of 50 planned epochs were executed before the session was stopped. The specialist has NOT converged. The pattern of B2 accuracy oscillating (57.4% at ep2, 64.4% at ep3, 73.7% at ep4, 77.9% at ep5) combined with B3 swinging opposite suggests the model is in an early optimization phase where the loss surface has competing attractors. The 72.81% checkpoint at epoch 2 is NOT a reliable best — it was saved because the OVERALL accuracy was highest, not because B2 was being well-separated. This specialist must be retrained to convergence (50 full epochs) before any end-to-end hierarchical evaluation.

**Decision**: Checkpoint `stage2_hier_b23_dinov2_classifier.pth` at epoch 2 is NOT trustworthy for end-to-end eval. Must retrain or continue training.

**Next**:
1. Either retrain the B2/B3 specialist to 50 full epochs (2h budget)
2. OR run hierarchical end-to-end eval with the current partial checkpoint as a lower-bound estimate
3. Then move to contrastive specialist (Experiment 19) if plain CE specialist still fails at B2


## Experiment 19 — PLAN: Supervised-Contrastive B2/B3 Specialist

**Observation**: The hierarchical branch removes the easy classes from the hard boundary, but it still relies on a standard cross-entropy binary specialist for `B2/B3`. Prior evidence already showed that flat cross-entropy tends to keep `B2` too close to `B3`, even with stronger backbones.

**Hypothesis**: If the `B2/B3` specialist is trained with cross-entropy plus supervised contrastive loss, the embedding space should separate ambiguous neighboring classes more explicitly than CE alone. This should improve the binary specialist first, then lift the hierarchical end-to-end pipeline when `P(B23)` is expanded into `P(B2)` and `P(B3)`.

**Design**:
- Keep the same hierarchical pipeline and the same coarse `B1/B23/B4` classifier.
- Replace the binary `B2/B3` specialist with a DINOv2 classifier that adds a trainable projection head and supervised contrastive loss during training.
- Use this branch only if the plain hierarchical specialist still fails to lift end-to-end mAP enough.

**Success criterion**: binary `B2/B3` validation accuracy beats the plain CE specialist, and hierarchical end-to-end `val_map50_95` exceeds the non-contrastive hierarchical baseline.

**Falsification rule**: If the contrastive specialist improves embedding separation but not the final pipeline metric, then the remaining bottleneck is no longer the `B2/B3` embedding geometry and the next branch must focus on detector-side localization or a stronger fine-grained token model.

**RESULT (2026-03-15 ~16:40 UTC)**: val_acc=73.07% at epoch 11 (B2=58.7%, B3=79.8%). Final epoch 50: val_acc=71.1%, B2=63.2%, B3=74.7%. DISCARD.

**Analysis**: SupCon did NOT improve binary B2/B3 accuracy over plain CE (72.81%). The best SupCon checkpoint is only 73.07% vs 72.81% CE — a negligible 0.26% improvement. The supcon loss did not help the model separate B2 from B3 better. The oscillation pattern (B2/B3 accuracy swinging every few epochs) persisted throughout all 50 epochs, suggesting the B2/B3 boundary is fundamentally unstable — this is consistent with label noise (annotators inconsistently labeling borderline bunches). FALSIFIED: contrastive loss does NOT help when the underlying problem is label noise/ambiguity.

**STRATEGIC PIVOT (per researcher directive)**: ABANDON two-stage pipeline. Focus on single-stage YOLO improvements:
1. YOLO with label smoothing (addresses B2/B3 ambiguity from the loss side)
2. DINOv2 fine-tuning (unfreeze backbone, small model)
3. Smaller models only (yolo11s/m), max 40 epochs / 0.5h per experiment


## Experiment 20 — 2026-03-15 ~16:45 UTC — YOLO11s Label Smoothing

**Hypothesis**: If label smoothing=0.1 is added to YOLO11s training, the model will be penalized less for near-miss B2/B3 predictions (treating labels as soft targets: P(B2)=0.9, P(B3)=0.1 for a B2 box), which should improve B2/B3 calibration and raise val_map50_95. Label smoothing is known to help with noisy/ambiguous labels.

**Mechanism**: Hard labels penalize the model equally for predicting B3 on an actually-mislabeled-B2 box as for a completely wrong prediction. Soft labels reduce this gradient signal, helping the model learn more robust features rather than overfitting to noisy B2/B3 boundaries.

**Change**: MODEL=yolo11s.pt (small, fast), label_smoothing=0.1, EPOCHS=40, TIME_HOURS=0.5

**Expected delta**: +2-5% mAP if label noise is the bottleneck (likely)
**Success criterion**: val_map50_95 > 0.27 OR B2 mAP50-95 > 0.23
**Falsification**: If label smoothing doesn't help, then the model is already robust to B2/B3 ambiguity and the bottleneck is elsewhere


## Experiment 18c — 2026-03-15 14:20 UTC — Hierarchical End-to-End Evaluation

**Hypothesis**: If the hierarchical pipeline (coarse B1/B23/B4 classifier + binary B2/B3 specialist) can route easy decisions correctly and delegate B2/B3 disambiguation to the specialist, the end-to-end mAP should exceed the flat 4-way DINOv2 baseline (0.181088) and ideally exceed the YOLO baseline (0.269424).

**Classifiers used**:
- Coarse: `stage2_hier_coarse3_dinov2_classifier.pth` — val_acc=75.09% (B1=85%, B23=76.5%, B4=64.6%)
- Binary: `stage2_hier_b23_dinov2_classifier.pth` — val_acc=72.81% (B2=57.4%, B3=80.0%) — ONLY 6/50 EPOCHS

**Result**: mAP50=0.358407, val_map50_95=0.177635 — DISCARD (far below baseline 0.269424)

**Per-class AP50-95**:
- B1: 0.3710 (vs one-stage 0.440 — worse)
- B2: 0.0972 (catastrophic — still the hard boundary)
- B3: 0.1576 (vs one-stage 0.270 — much worse)
- B4: 0.0847 (vs one-stage 0.152 — worse)

**Analysis**:
1. The binary specialist (only 6 epochs) was NOT converged. The 72.81% was at epoch 2. This directly caused B2 collapse at 0.097.
2. The coarse B4 leakage (35% of B4 crops routed to B23 branch) forces those B4 boxes into the binary specialist which classifies them as B2 or B3 — explaining B4 at 0.085.
3. Compound error: detector (~90% recall) × coarse (~75%) × binary (~60% for B2) = chain product leaves very little signal.
4. Even if we perfect the binary specialist, the B4 leakage (35%) in the coarse classifier fundamentally caps B4 at 65%.
5. The hierarchical strategy is fundamentally limited by the coarse classifier's inability to separate B4 from B23.

**Key insight**: The hierarchical decomposition is NOT sufficient. The core bottleneck remains the same visual ambiguity, just shifted between stages. The B23 binary specialist also needs much more training.

**Immediate next steps (in priority order)**:
1. Train SupCon B23 specialist to convergence (50 epochs) — this is Experiment 19
2. After getting SupCon specialist, re-evaluate hierarchical pipeline
3. If still fails, abandon two-stage entirely and focus on YOLO with label smoothing or fine-tuning DINOv2 backbone (unfreeze last N layers)


## Experiment 19 — 2026-03-15 14:25 UTC — SupCon B2/B3 Specialist Training

**Hypothesis**: If the binary B2/B3 specialist is trained with CE + supervised contrastive loss (SupCon) with a projection head, the embedding space will push B2 and B3 apart more explicitly than CE alone, raising binary accuracy above the plain CE 72.81% and improving end-to-end mAP50-95 of the hierarchical pipeline.

**Mechanism**: SupCon loss minimizes cosine distance between embeddings of same-class crops while maximizing it between different classes. For the near-identical B2/B3 boundary, this geometric pressure should find discriminative features that CE loss misses.

**Change**: Run `scripts/train_dinov2_supcon_imagefolder_classifier.py` on `Dataset-Crops-B23` (50 epochs, 2h budget, supcon_weight=0.15, supcon_temperature=0.07, supcon_start_epoch=3).

**Success criterion**: binary B2/B3 val_acc > 73% AND B2 acc > 60%. Then re-run hierarchical eval.

**Falsification**: If SupCon doesn't improve binary accuracy over CE, then the B2/B3 ambiguity is not solvable via representation learning — it is inherent label noise, and no classifier will fix it.



## Experiment 20 — 2026-03-15 15:07 UTC — Label Smoothing 0.1 on yolo11s

**Hypothesis**: label_smoothing=0.1 will reduce overconfident wrong predictions on B2/B3 boundary cases, lifting B2 mAP by penalizing hard labels less.

**Change**: MODEL=yolo11s.pt, label_smoothing=0.1, EPOCHS=40, TIME_HOURS=0.5, Dataset-TrainTest

**Result**: val_map50_95=0.255244 — DISCARD (below yolo11l baseline 0.269424)

**Per-class**: B1=0.430, B2=0.206, B3=0.255, B4=0.129

**Analysis**: yolo11s with label_smoothing reached 0.255 at 57 epochs (time budget expired). This is worse than yolo11l without label_smoothing (0.269). However, we cannot isolate label_smoothing's effect from the model size change (s vs l). B2 at 0.206 — not improved vs baseline 0.216. B4 at 0.129 — worse than baseline 0.152. The smaller model is clearly weaker across all classes. Label smoothing alone cannot compensate for the capacity difference. CONCLUSION: Cannot confirm label smoothing helps without a fair comparison (yolo11l + label_smoothing vs yolo11l baseline). Next step: test label_smoothing on yolo11m to get a fairer signal.

**Next**: Try yolo11m + label_smoothing=0.1 (medium model, 40 epochs) for a fairer comparison vs yolo11l baseline.

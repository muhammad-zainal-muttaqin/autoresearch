# E0 Baseline Experimental Protocol

# Strategic Guide: RGB-Only Detection & 4-Class Maturity Classification

**Project**: Oil Palm Black Bunch Census (BBC)
**Phase**: Year 1, Post-DC1
**Duration**: ~4 weeks (guideline, not rigid schedule)
**Objective**: Establish RGB-only baseline performance for maturity classification

**Target Deployment**:

- **Device**: Xiaomi Pad 6 / Samsung Galaxy Tab S8 (Tablets, 8GB RAM, 256GB storage)
- **Latency Budget**: 5 seconds per image

---

## 1. METHODOLOGY: WHY SEQUENTIAL SEARCH?

### The Strategic Trade-Off

**Full Grid Search** (exhaustive but expensive):

```
3 LR × 2 batch × 3 aug × 10 arch × 2 seeds = 396 runs ≈ 1,200 GPU hours
```

**Sequential Search** (pragmatic approach):

```
Pipeline Decision → Architecture Sweep → Loss Function (Imbalance + Ordinal) → LR → Batch → Augmentation
Total: ~120-130 runs ≈ 360-390 GPU hours (68-71% reduction)
```

**What We Sacrifice**: Potential interaction effects between hyperparameters (e.g., "heavy augmentation works best with low LR"). These interactions are typically weak, especially in transfer learning scenarios where pre-trained weights provide stable initialization.

**What We Gain**: Budget efficiency, faster iteration, clearer attribution of performance gains to specific parameters.

**Decision Principle**: In budget-constrained research, sequential search isolates effects and maintains scientific rigor without exhaustive compute.

### Search Order Rationale

1. **Resolution + Data Sufficiency + Split** (Phase 0): Affects ALL subsequent experiments. Lock resolution, confirm data sufficiency, and establish tree-grouped stratified splits.
2. **Pipeline** (Phase 1A): One-stage vs two-stage is a fundamental design choice. Settle before sweeping architectures.
3. **Architecture** (Phase 1B): Sweep within the winning pipeline. Model capacity is fundamental — find the right family/size before fine-tuning.
4. **Loss Function: Imbalance + Ordinal** (Phase 2): Changes the optimization landscape fundamentally. Lock this before tuning LR.
5. **Learning Rate** (Phase 2): Most impactful hyperparameter for optimization dynamics.
6. **Batch Size** (Phase 2): Influences gradient estimation quality (interacts with LR, hence sequential).
7. **Augmentation** (Phase 2): Regularization strength, evaluated last with stable training dynamics.

This order follows standard deep learning practice: data validation → pipeline → capacity → loss function → optimization → regularization.

---

## 2. STRATEGIC OBJECTIVES & SUCCESS CRITERIA

### What Success Looks Like

We are NOT simply maximizing a single metric. E0 success means understanding baseline performance across multiple dimensions:

| Objective      | Metric           | Target   | Business Impact                                                                                         |
| -------------- | ---------------- | -------- | ------------------------------------------------------------------------------------------------------- |
| **Primary**    | mAP@0.5          | ≥85%     | Overall detection capability for census accuracy                                                        |
| **Co-Primary** | M2/M3 Confusion  | <30%     | Yield forecasting depends on distinguishing 2-month vs 3-month maturity                                 |
| **Tracked**    | M3/M4 Confusion  | Report   | Adjacent black bunch stages differing in size; less operationally critical than M2/M3 but informs E1-E6 |
| **Technical**  | M4 AP            | ≥70%     | Validates small object detection (~20×30px bunches at 640px)                                            |
| **Robustness** | Min Per-Class AP | All ≥70% | Prevents model ignoring minority classes                                                                |

### Decision Framework: Not Just "Is it Good?"

At the end of E0, we don't ask "did we get 90% mAP?" We ask "what's the smartest next step?"

| Scenario         | mAP@0.5 | M2/M3 Conf | Min Class | Decision         | Next Action                                |
| ---------------- | ------- | ---------- | --------- | ---------------- | ------------------------------------------ |
| **Excellent**    | ≥90%    | <20%       | ≥70%      | Deploy           | Skip E1-E6 feature engineering             |
| **Good**         | 85-90%  | 20-30%     | ≥70%      | Deploy           | Optional E1-E3 for refinement              |
| **Acceptable**   | 80-85%  | >30%       | ≥70%      | Deploy baseline  | E1-E3 mandatory before production          |
| **Needs Work**   | 75-80%  | Any        | <70% some | Don't deploy yet | Full E1-E6 feature engineering (4-6 weeks) |
| **Insufficient** | <75%    | Any        | Any       | Investigate      | Review data quality, potentially re-label  |

The framework acknowledges that 85% mAP might be deployment-ready if M2/M3 confusion is low, while 90% mAP with high M2/M3 confusion requires additional work.

---

## 3. DATASET PHILOSOPHY: WHY 4 CLASSES?

### Class Structure Decision

**E0 Baseline Classes** (4 total):

- **M1**: Mature, harvest in 1 month (red/orange, visually distinct)
- **M2**: Harvest in 2 months (black, large)
- **M3**: Harvest in 3 months (black, medium)
- **M4**: Harvest in 4 months (black, small)

---

## 4. PHASE 0: VALIDATION & CALIBRATION (~3 days guideline)

### Strategic Questions to Answer

1. **Class Balance**: Is the dataset severely imbalanced (>10:1 ratio)? If yes, add weighted loss or resampling strategies.
2. **M4 Object Size**: What is the minimum pixel dimension of M4 bunches? If <16px, standard 640px resolution will likely fail.
3. **Resolution Decision**: Does 1024px significantly improve MAP? If >5% AP gain, the 2× compute cost is justified.
4. **Data Sufficiency**: Is ~3900 images enough, or would more data collection yield bigger gains than architecture/hyperparameter tuning?

### Task A: Exploratory Data Analysis & Split Strategy

**Data Split** (lock before anything else):

- **Split unit is the tree, not the image.** Each tree has 4 images from different 90° angles. All 4 images of a tree go into the same split. With ~975 unique trees, a 70/15/15 tree-level split gives ~682/146/146 trees (~2730/585/585 images).
- **Stratify by dominant maturity class** at the tree level. Define dominant class as the most frequent class in that tree's annotations. This ensures each split has proportional M1/M2/M3/M4 representation.
- **Why group by tree?** Although views differ substantially at 90° angles (most FFBs visible in one view are not visible in others), same-tree images share background, trunk texture, canopy density, and lighting conditions. Grouping eliminates this potential confound at negligible cost — 975 trees is plenty for a well-powered split.
- All subsequent subsampling (Phase 0 Task C learning curve, Phase 1 runs) must respect tree-level grouping.

**What to Measure**:

- Class distribution across train/val splits (check for imbalance)
- Object size distributions (bounding box width/height in pixels)
- M4 minimum dimensions (critical bottleneck for small object detection)



### Task B: Resolution Sweep

**Experiment Design**:

- Test 1 YOLO model (e.g YOLOv8s) at 640px vs 1024px (baseline architecture, medium augmentation)
- Run 2 seeds (0, 42) for each resolution

**Decision Criteria**:

- If M4 AP improves >5% at 1024px → Lock 1024px for Phases 1-3 (worth 2× compute)
- If improvement 2-5% → Marginal case, consider tablet memory constraints (1024px = 4× memory)
- If improvement <2% → Stick with 640px (compute efficiency)

**Time Budget**: 12 GPU hours (4 runs × 3 hours each). Fits in 1 day with monitoring buffer.

### Task C: Data Amount Learning Curve

**Why Before Phase 1?**: If the learning curve is still climbing steeply at 100% of training data, collecting more images may outperform any amount of architecture or hyperparameter tuning. This is a prerequisite strategic decision, like resolution.

**Experiment Design**:

- Train at 25%, 50%, 75%, 100% of training data (same architecture as resolution sweep)
- Use **stratified sampling** by class for each fraction — ensure M4 (and other minority classes) are proportionally represented even in the 25% split. If M4 has <50 instances in the 25% split, that's itself a finding worth reporting.
- Use locked resolution from Task B, medium augmentation, 2 seeds (0, 42)
- Evaluate **per-class AP** at each data fraction (not just overall mAP). Plot M1/M2/M3/M4 curves separately.

**Decision Criteria**:

- If overall curve plateaus at 75-100% (gain <1%) → Data is sufficient, proceed with current dataset
- If curve still climbing at 100% (gain >2% from 75→100%) → More data would help. Flag to PI: data collection may be higher ROI than E1-E6 feature engineering
- If M4 AP specifically is still climbing steeply while other classes plateau → More M4 samples needed specifically (targeted data collection)
- Report the curves regardless—they contextualize all subsequent results

**Time Budget**: 16 GPU hours (8 runs × 2 hours each, smaller datasets train faster). ~1 day.

### Phase 0 Deliverables

Lock the following before Phase 1:

- **Data Split**: Tree-grouped, class-stratified train/val/test (70/15/15)
- **Input Resolution** (imgsz): 640 or 1024
- **Class Weighting Strategy**: None, weighted loss, or resampling
- **Data Sufficiency Assessment**: Per-class learning curve plot + recommendation (proceed vs collect more data)
- **Data Quality Assessment**: Are labels accurate? Any obvious errors in top 20 images?

**Critical Threshold**: If >10% of sampled images have labeling errors (wrong bounding boxes, misclassified maturity), STOP E0 and re-label the dataset. Training on bad labels wastes all subsequent effort.

---

## 5. PHASE 1: ARCHITECTURE & PIPELINE (~2 weeks guideline)

Phase 1 is split into two sub-phases. Phase 1A settles the pipeline question first (one-stage vs two-stage). Phase 1B then sweeps architectures within the winning pipeline. 

### Phase 1A: Pipeline Decision (~2-3 days)

**Why First?**: Prior experiments on a related dataset (2-class, public) showed two-stage outperforms one-stage for maturity classification. This is a fundamental pipeline decision that affects everything downstream — which architectures matter, how hyperparameters interact, and deployment complexity. Settle it before investing in a full sweep.

**One-Stage**: YOLO detects and classifies simultaneously (4-class output).

**Two-Stage**: Stage 1 detects all FFBs as a single class (binary: FFB vs background). Stage 2 takes cropped FFB regions and classifies maturity (M1/M2/M3/M4) using a dedicated classifier.

**Experiment Design**:

- Select 2-3 representative architectures spanning different families (e.g., YOLOv8m, YOLO26m, YOLOv10m — one per family, all medium-tier)
- Run each in both one-stage (4-class) and two-stage (binary YOLO + classifier) modes, 2 seeds
- For two-stage Stage 2, test 2-3 classifier options: EfficientNet-B0, ResNet-18, YOLO-cls
- Compare on dual-gate metrics: mAP@0.5 + M2/M3 confusion + M3/M4 confusion

**Decision Criteria**:

- If two-stage improves M2/M3 or M3/M4 confusion by >5% → Adopt two-stage for Phase 1B onward
- If improvement <2% → Stay one-stage (simpler deployment)
- If 2-5% → Judgment call based on deployment complexity tolerance

**The Tradeoff**: Two-stage adds deployment complexity (two models, crop pipeline). If two-stage wins on accuracy, use it.

### Phase 1B: Architecture Sweep (~1.5 weeks)

The sweep runs within the winning pipeline from Phase 1A.

**If one-stage won**: Sweep all YOLO architectures as 4-class detectors. Straightforward.

**If two-stage won**: The sweep has two dimensions:

- **Detector sweep** (Stage 1): Test YOLO architectures as binary FFB detectors. The detector determines what crops the classifier sees, so lock this first.
- **Classifier sweep** (Stage 2): Test 3-4 classifiers (EfficientNet-B0, ResNet-18/50, YOLO-cls, MobileNetV3) on top of the best detector. Smaller search space since crop classification is a simpler task.

### The 11 Detector Candidates

| Architecture | Params    | Rationale                                                       |
| ------------ | --------- | --------------------------------------------------------------- |
| YOLOv8n      | 3.2M      | Baseline nano, fastest                                          |
| YOLOv8s      | 11.2M     | Industry standard, transfer learning baseline                   |
| **YOLOv8m**  | **25.6M** | **Higher capacity, tablet-feasible**                            |
| YOLOv9-c     | 25.3M     | GELAN backbone (Feb 2024), architectural innovation             |
| YOLOv10n     | 2.3M      | NMS-free detection (May 2024), efficiency focus                 |
| YOLOv10s     | 7.2M      | NMS-free, small variant                                         |
| **YOLOv10m** | **15.8M** | **NMS-free medium, architectural efficiency**                   |
| **YOLO26n**  | **2.4M**  | **STAL small-target-aware, NMS-free v2, edge-first (Jan 2026)** |
| **YOLO26s**  | **9.5M**  | **STAL, 43% CPU speedup, edge-optimized**                       |
| **YOLO26m**  | **20.4M** | **Primary M4 feasibility test, STAL, tablet-ready**             |
| YOLOv11m     | 20.1M     | Sep 2024 medium, baseline comparison for YOLO26m                |

All models pre-trained on COCO dataset (80 classes, 118k images). Transfer learning from COCO to agricultural domains has proven effective in literature.

### What We're Learning

We're answering strategic questions:

1. **Capacity vs Overfitting**: Do medium models significantly outperform small (>2% mAP gap)? Or do they overfit on ~2800 training images?
2. **Architecture Innovations**: Do GELAN backbones (v9) or NMS-free heads (v10) help with agricultural scenes?
3. **Failure Modes**: Where do models fail? Occlusion? Lighting extremes? Scale variation? This informs E1-E6 priorities.

### Execution Principles

**Fair Comparison**: All architectures use identical hyperparameters (lr0=0.001, batch=16, medium augmentation, patience=15-25). Any differences in performance are architectural, not due to lucky hyperparameter choices.

**Reproducibility**: Two seeds (e.g. 0, 42) verify stability. If Seed 0 shows Model A > Model B by 3%, but Seed 42 shows the reverse, the difference is noise, not signal.

**Early Stopping**: Patience=15-25 prevents overfitting. Some architectures may converge in 30 epochs, others in 80. Let each model find its own convergence point.

**Per-Class Tracking**: Don't just report overall mAP. Track M1/M2/M3/M4 separately.

### End-of-Phase Triage

**After Phase 1A**: Is the pipeline decision clear-cut? If results are within noise (2-seed disagreement), run additional seeds before committing.

**After Phase 1B** — this is NOT just "rank by mAP and pick top 2." Ask:

- **Why did winners win?** Better feature extraction? Better handling of scale variation?
- **Where do ALL models fail?** If all 11 architectures struggle with M3/M4 confusion, it's a data problem, not architecture.
- **Architecture-specific strengths**: Did v9 GELAN excel at occlusion but fail at small objects? This guides Phase 2 and E1-E6.
- **Failure stratification**: Manually inspect worst 20 images per top-3 architecture. Categorize errors:
  - Small object missed (M4)
  - M2/M3 confusion (maturity misjudgment)
  - M3/M4 confusion (size/maturity overlap)
  - Occlusion (heavy leaves blocking bunches)
  - False positives (leaves/trunks detected as bunches)
  - Lighting extremes (backlit, deep shadow)
  - Label errors (suspected incorrect ground truth)

**Critical Threshold**: If >10% of errors are suspected label quality issues, STOP E0 immediately. Re-label the dataset. Training on bad labels is futile.

### Phase 1 Deliverables

- **Pipeline Decision** (Phase 1A): One-stage vs two-stage, with quantified comparison on dual-gate metrics. If two-stage, best Stage 2 classifier identified.
- **Top 2-3 Architectures** (Phase 1B): Selected based on performance + stability (2-seed agreement), within winning pipeline
- **Architecture Insights Report**: Why winners won, where ALL models failed, failure mode categorization
- **M4 Feasibility Assessment**: Is M4 detection viable at current resolution/data? If not, escalate to PI.
- **GO/NO-GO Decision**:
  - ✅ If best mAP ≥70% → Proceed to Phase 2
  - ❌ If best mAP <70% → STOP, review data quality before continuing

---

## 6. PHASE 2: HYPERPARAMETER OPTIMIZATION (~1 week guideline)

### Why Sequential Search? The Compute Economics

**Full Grid Search**: 3 loss × 3 LR × 2 batch × 3 aug = 54 configs per architecture. For 3 architectures × 2 seeds = 324 runs ≈ 972 GPU hours. Exceeds budget.

**Sequential Search**: Optimize one parameter at a time.

- Step 0a (Imbalance): 3 values × 3 arch × 2 seeds = 18 runs
- Step 0b (Ordinal): 2-3 values × 3 arch × 2 seeds = 12-18 runs
- Step 1 (LR): 3 values × 3 arch × 2 seeds = 18 runs
- Step 2 (Batch): 2 values × 3 arch × 2 seeds = 12 runs
- Step 3 (Aug): 3 levels × 3 arch × 2 seeds = 18 runs
- Total: ~78-84 runs ≈ 235-250 hours

**Trade-Off**: We might miss interaction effects (e.g., "heavy augmentation requires lower LR"). In practice, such interactions are weak when starting from pre-trained weights. Transfer learning provides stable initialization, reducing sensitivity to hyperparameter interactions.

### Search Space & Rationale

#### Step 0: Loss Function [Imbalance Handling + Ordinal Classification]

**Why Loss Function First?**: The loss function changes the optimization landscape fundamentally. Both class imbalance handling and ordinal awareness are loss-level decisions. Optimizing LR on a standard loss, then switching later, would invalidate the LR choice. Lock the loss, then tune everything else on top of it.

**Class Imbalance Options**:

- **No weighting**: Baseline, lets natural class distribution drive learning
- **Class-weighted loss**: Inverse frequency weighting, upweights minority classes (M4)
- **Focal loss** (γ=1.5): Downweights easy examples (M1), focuses on hard examples (M2/M3/M4 confusion)

**Ordinal Classification Options** (M1→M2→M3→M4 is a natural maturity ordering):

*If one-stage (Phase 1 decision)*:

- **Standard cross-entropy**: Baseline, treats all misclassifications equally
- **Ordinal-weighted cross-entropy**: Custom penalty matrix where adjacent-stage errors (M2→M3) are penalized less than distant errors (M1→M4)

*If two-stage (Phase 1 decision)*: Ordinal approaches apply to the Stage 2 classifier:

- **Standard cross-entropy**: Baseline
- **Ordinal-weighted cross-entropy**: Same penalty matrix as above
- **CORAL/CORN loss**: Purpose-built ordinal classification losses
- **Regression head**: Predict maturity as continuous value (1-4), threshold at deployment

**Why Ordinal Matters**: Predicting M2 when truth is M3 (1-step error) is less bad than predicting M1 when truth is M4 (3-step error). Standard classification doesn't know this. Ordinal-aware losses encode the maturity ordering directly. Two-stage pipelines make ordinal approaches easier to implement and more natural.

**Practical Approach**: Don't test the full Cartesian product. Test the best imbalance option against the best ordinal option:

1. Run imbalance comparison first (no weighting vs class-weighted vs focal), pick winner
2. Run ordinal comparison (standard CE vs ordinal-weighted vs CORAL if two-stage), pick winner
3. Optionally combine best imbalance + best ordinal if both show gains

**What We're Learning**: Is class imbalance or ordinal structure the bigger lever for M2/M3/M4 discrimination? This directly informs whether E1-E6 should focus on data balancing or classification methodology.

#### Step 1: Learning Rate [0.0005, 0.001, 0.002]

**Why LR First?**: Learning rate fundamentally affects optimization dynamics. Fix this before tuning regularization (batch, aug).

- **0.001**: Baseline from Phase 1, standard for YOLO fine-tuning
- **0.0005**: Conservative, may improve stability on difficult classes (M2/M3/M4)
- **0.002**: Aggressive, faster convergence but risks instability

**What We're Learning**: Do agricultural scenes (outdoor, variable lighting) require more conservative LR than COCO pre-training? Or can we converge faster with aggressive LR?

#### Step 2: Batch Size [8, 16]

**Why Batch After LR?**: Batch size affects gradient estimation quality, which interacts with LR. Tune with optimal LR locked.

- **16**: Baseline from Phase 1, fits on GPU
- **8**: Smaller batches = noisier gradients = better generalization (sometimes)

**What We're Learning**: Do smaller batches help with M2/M3 discrimination (regularization benefit)? Or does the noisier training harm convergence?

**Note**: Larger batches (32+) excluded due to GPU memory. 1024px resolution @ batch=32 exceeds Colab RAM.

#### Step 3: Augmentation [Light, Medium, Heavy]

**Why Aug Last?**: Augmentation is regularization. Evaluate after training dynamics (LR + batch) are optimized.

**Light**: Minimal augmentation (hsv ~50%, degrees 5°, mosaic 0.5)
**Medium**: Baseline from Phase 1 (hsv 70-100%, degrees 10°, mosaic 1.0)
**Heavy**: Aggressive (hsv 120%, degrees 15°, mosaic 1.0, mixup 0.15)

**Agricultural Context**: All settings simulate NORMAL field conditions (lighting variation, positioning, occlusion). Weather effects (rain, fog) are EXCLUDED—reserved for E4-E6 environmental robustness experiments.

**What We're Learning**: Do outdoor agricultural scenes benefit from heavy augmentation (high lighting/positioning variability)? Or does heavy augmentation hurt M4 detection (small objects need cleaner training)?

### End-of-Phase Triage

Don't just report "best config found." Ask:

1. **Did optimization matter?**: Compare Phase 2 best vs Phase 1 baseline. If improvement <1%, revert to Phase 1 config (simpler is better).
2. **Which parameter mattered most?**: Did loss function dominate (suggests imbalance/ordinal structure is real)? Did LR dominate (common)? Or did augmentation drive gains (suggests data variability)?
3. **Imbalance vs ordinal**: Which was the bigger lever for M2/M3 and M3/M4 confusion? This informs E1-E6 direction.
4. **Architecture-specific configs?**: Do all architectures prefer the same config (generalizable finding)? Or different configs (architecture-dependent tuning)?
5. **M2/M3 and M3/M4 improvement?**: Did any config reduce adjacent-class confusion? Track both confusion pairs.
6. **Per-class tracking**: Did M4 AP improve? Or did optimization boost easy classes (M1) while ignoring hard ones?

**Decision Principle**: If average improvement across top architectures is <1%, use Phase 1 baseline config. Simpler is better when gains are marginal.

### Phase 2 Deliverables

- **Best Config Per Architecture**: Loss function (imbalance + ordinal), LR, batch, augmentation locked for each top-2-3 architecture
- **Improvement Analysis**: Quantified gains vs Phase 1 baseline, per-class breakdown, M2/M3 and M3/M4 confusion changes
- **Parameter Sensitivity Report**: Which hyperparameter mattered most? Imbalance vs ordinal — which was the bigger lever?
- **Decision**: Use Phase 2 config or revert to Phase 1 if gains are marginal

---

## 7. PHASE 3: FINAL VALIDATION (~3 days guideline)

### Why Combine Train+Val for Final Training?

**Standard ML Practice**:

- **Development** (Phases 0-2): Use train/val split for model selection and hyperparameter tuning
- **Deployment**: Train final model on ALL labeled data for maximum capacity

This is NOT data leakage. Hyperparameters are FROZEN from Phase 2. We're not tuning anymore—just maximizing capacity with more training data.

**Why This Matters**: 2800 train → 3400 train+val = 21% more data. For deep learning, more data typically means better generalization (up to a point). Since we're no longer selecting models (that was Phase 1) or tuning hyperparameters (that was Phase 2), using all available labeled data is standard practice.

### What Changes from Phase 2

**Data**: 2800 train → 3400 train+val combined
**Epochs**: May need more (more data = slower convergence). Increase from 100 → 150 as buffer.
**Patience**: No early stopping (no validation set to monitor). Use fixed epochs instead.
**Hyperparameters**: FROZEN from Phase 2. No tuning in Phase 3.

### Why Evaluate on Original Val Split?

After training on train+val combined, evaluate on the ORIGINAL 600-image validation split.

**Why This Isn't Circular**: We're not measuring generalization anymore (model trained on this data). We're measuring:

1. **Final capacity**: How well does the model perform with maximum training data?
2. **Deployment readiness**: Can it achieve target metrics (≥85% mAP, <30% M2/M3 confusion)?
3. **Fair comparison**: Same eval set as Phase 1-2, so improvements are comparable.

**Test Set Still Reserved**: The 600-image test set is NEVER touched in E0. It's reserved for final system evaluation after E1-E6 + counting methods (Paper 2A/Paper 5 scope).

### Deployment Validation: Can This Model Actually Deploy?

Before error analysis, verify the model can export to production format.

**TFLite Export Test**:

- Export best model to TFLite format (Android/iOS deployment)
- Check file size (<100MB preferred for tablet storage)
- Verify export completes without errors (some YOLO operations may not be TFLite-compatible)

**Quantization Impact** (optional):

- Test INT8 quantization (4× size reduction, faster inference)
- Measure accuracy drop (typically <1-2% mAP acceptable)
- Note: Full INT8 validation requires TFLite interpreter on tablet

**Tablet Inference Test** (optional, hardware-dependent):

- If Xiaomi Pad 6 / Samsung Galaxy Tab S8 available, transfer model and test:
  - Inference latency (target: <500ms)
  - Memory usage (target: <2GB)
  - Verify predictions match desktop results

If tablet hardware unavailable, document as "post-E0 validation recommended."

### Confidence Threshold Optimization

**Why Here?**: The detection confidence threshold is a post-training calibration step—it doesn't change model weights, just the operating point on the precision-recall curve. Optimize it before error analysis so that all reported metrics and failure modes reflect the best operating point.

**Experiment Design**:

- Sweep confidence thresholds: [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5] on original val split
- Evaluate each threshold on dual-gate criteria: mAP@0.5 AND M2/M3 confusion rate
- Also track M4 recall specifically (lower thresholds may catch more M4 instances)

**Decision Criteria**:

- Select threshold that best balances the dual-gate (not just max mAP)
- A threshold that improves M4 recall by 5%+ at the cost of <1% overall mAP is likely worth it
- Document the full threshold-vs-metric curve for reference

**Cost**: Essentially free—inference passes with different thresholds, no retraining.

### End-of-Phase Triage

This analysis determines ALL next actions. Don't just report metrics—understand failure modes.

**Overall Metrics**:

- mAP@0.5, mAP@0.75 (stricter localization), per-class AP
- Calculate M2/M3 confusion rate from confusion matrix

**Structured Error Stratification** (mandatory):

- Identify worst 20 images (largest detection count errors)
- Manually categorize each error into:
  - Small object missed (M4)
  - M2/M3 confusion (maturity misjudgment)
  - M3/M4 confusion (size/maturity overlap)
  - Occlusion (heavy leaves blocking bunches)
  - False positives (leaves/trunks detected as fruit)
  - Lighting extremes (backlit, deep shadow)
  - Motion blur / out of focus
  - Label error (suspected incorrect ground truth)
- Calculate percentage distribution (critical for E1-E6 prioritization)

**Data Quality Final Check**:

- Count suspected label errors among worst 20 images
- If >10% are label quality issues → STOP, re-label dataset, restart E0
- This threshold is NON-NEGOTIABLE. Bad labels = wasted training.

**Root Cause → Next Action**:

- If errors dominated by small object issues (M4) → E4 (multi-scale features)
- If errors dominated by M2/M3 confusion → E1-E3 (position + size features)
- If errors dominated by occlusion → E5 (attention mechanisms)
- If errors dominated by lighting → E6 (weather augmentation, robust normalization)
- If errors evenly distributed → Full E1-E6 pipeline

### Phase 3 Deliverables

- **Deployment Validation Report**: TFLite export status, quantization results, tablet test (if available)
- **Confidence Threshold Report**: Optimal threshold, threshold-vs-metric curves for dual-gate criteria
- **Final Performance Metrics**: mAP@0.5, mAP@0.75, per-class AP, M2/M3 confusion rate (all at optimized threshold)
- **Confusion Matrix** (4-class): Visualized, with M2/M3 confusion highlighted
- **Error Stratification Table**: Categorized failure modes with percentages
- **E0 Final Decision Document**: Which scenario from decision framework, recommended next actions

---

## 8. DECISION FRAMEWORK: INTERPRETING RESULTS & NEXT ACTIONS

### The 5 Scenarios

**Scenario 1: Excellent (mAP ≥90%, M2/M3 <20%, all classes ≥70%)**

- **Interpretation**: RGB-only baseline exceeds expectations. High overall accuracy + low critical confusion.
- **Next Action**: Deploy baseline model. Skip E1-E6 feature engineering (not needed).
- **Proceed To**: Counting methods development (Paper 2A). E0 objective achieved.

**Scenario 2: Good (mAP 85-90%, M2/M3 20-30%, all classes ≥70%)**

- **Interpretation**: Strong baseline, but M2/M3 confusion slightly elevated.
- **Next Action**: Deploy baseline OR pursue E1-E3 (position + size features) for refinement. Optional, not mandatory.
- **Proceed To**: Counting methods in parallel with optional E1-E3 (~2 weeks if pursued).

**Scenario 3: Acceptable (mAP 80-85%, M2/M3 >30%, all classes ≥70%)**

- **Interpretation**: Baseline viability established, but M2/M3 confusion unacceptable for yield forecasting.
- **Next Action**: Deploy baseline for field testing, but E1-E3 is MANDATORY before production.
- **Proceed To**: E1-E3 feature engineering (~2-3 weeks), then counting methods.

**Scenario 4: Needs Work (mAP 75-80% OR some classes <70%)**

- **Interpretation**: Baseline detects bunches but struggles with maturity discrimination or minority classes.
- **Next Action**: Do NOT deploy yet. Pursue full E1-E6 feature engineering pipeline.
- **Proceed To**: E1-E6 (4-6 weeks), comprehensive feature engineering, then re-evaluate deployment.

**Scenario 5: Insufficient (mAP <75%)**

- **Interpretation**: Fundamental issues with data quality, class definition, or task feasibility.
- **Next Action**: STOP. Investigate root cause before proceeding.
- **Proceed To**: Data quality review, potential re-labeling, possibly revisit class definitions (merge M3/M4?).

### What Makes a "Smart" Decision?

Don't chase 95% mAP if 85% is deployment-ready. Ask:

- **Is M2/M3 confusion low enough for yield forecasting?** (Business requirement)
- **Are all classes detectable?** (Prevents bias toward easy classes)
- **Where is the ceiling?** (If errors are mostly label quality, more training won't help)
- **What's the ROI of E1-E6?** (If Scenario 1, E1-E6 is 4-6 weeks for marginal gain)

The decision framework balances scientific rigor (understand failure modes) with pragmatic action (don't over-engineer when baseline suffices).

---

## APPENDIX A: BASELINE CONFIGURATION REFERENCE

### Phase 0 & Phase 1 Baseline

| Category         | Parameter     | Value              | Rationale                                  |
| ---------------- | ------------- | ------------------ | ------------------------------------------ |
| **Data**         | imgsz         | *Phase 0 Decision* | 640 or 1024, locked after resolution sweep |
|                  | batch         | 16 (or 8 if OOM)   | GPU memory constraint                      |
| **Training**     | epochs        | 100                | Early stopping via patience                |
|                  | patience      | 15-25              | Prevent overfitting, allow convergence     |
|                  | optimizer     | AdamW              | Adaptive learning, weight decay            |
|                  | lr0           | 0.001              | Standard YOLO fine-tuning rate             |
|                  | lrf           | 0.01               | Final LR = 0.00001 (100× decay)            |
|                  | cos_lr        | True               | Cosine annealing schedule                  |
|                  | warmup_epochs | 5                  | Stabilize early training                   |
| **Augmentation** | hsv_h         | 0.015              | Hue jitter ±1.5% (lighting variation)      |
| (Medium)         | hsv_s         | 0.7                | Saturation ±70% (outdoor scenes)           |
|                  | hsv_v         | 0.4                | Brightness ±40% (shadow/sunlight)          |
|                  | degrees       | 10.0               | Rotation ±10° (camera angle variation)     |
|                  | translate     | 0.1                | Translation ±10% (positioning)             |
|                  | scale         | 0.5                | Scale 50-150% (distance variation)         |
|                  | fliplr        | 0.5                | Horizontal flip 50% (symmetry)             |
|                  | mosaic        | 1.0                | Mosaic augmentation (YOLO-specific)        |
|                  | mixup         | 0.0                | No mixup (Phase 1 baseline)                |

**Note**: Weather effects (rain, fog, haze) explicitly EXCLUDED. Reserved for E4-E6 environmental robustness. E0 simulates normal field conditions only.

### Phase 2 Search Ranges

| Step | Parameter              | Values Tested                              | Rationale                                                    |
| ---- | ---------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| 0a   | Imbalance Handling     | [No weighting, Class-weighted, Focal loss] | Lock imbalance strategy before ordinal                       |
| 0b   | Ordinal Classification | [Standard CE, Ordinal-weighted CE, CORAL*] | Encode maturity ordering in loss (*CORAL for two-stage only) |
| 1    | Learning Rate          | [0.0005, 0.001, 0.002]                     | Conservative → Baseline → Aggressive                         |
| 2    | Batch Size             | [8, 16, 32]                                | Gradient noise trade-off                                     |
| 3    | Augmentation           | [Light, Medium, Heavy]                     | Regularization strength                                      |

**Light Augmentation**: hsv ~50%, degrees 5°, mosaic 0.5
**Medium Augmentation**: Baseline from Phase 1
**Heavy Augmentation**: hsv 120%, degrees 15°, mosaic 1.0, mixup 0.15

---

## APPENDIX B: METRICS & THEIR BUSINESS MEANING

### Metric Definitions

| Metric              | Definition                        | Business Interpretation                                                                                            |
| ------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **mAP@0.5**         | Mean Average Precision at IoU=0.5 | Overall detection capability. ≥85% means model reliably finds bunches.                                             |
| **mAP@0.75**        | Mean AP at stricter IoU=0.75      | Localization precision. High mAP@0.75 means tight bounding boxes (important for size estimation in E1-E3).         |
| **M2/M3 Confusion** | Avg of (M2→M3 rate + M3→M2 rate)  | Yield forecasting error. 30% confusion = 30% of 2-month bunches mislabeled as 3-month (1-month forecasting error). |
| **M3/M4 Confusion** | Avg of (M3→M4 rate + M4→M3 rate)  | Adjacent stage confusion. Less operationally critical than M2/M3 but tracked to inform E1-E6 priorities.           |
| **Per-Class AP**    | AP for each of M1/M2/M3/M4        | Class-specific performance. Prevents high overall mAP masking weak minority classes.                               |
| **M4 AP**           | AP for M4 (smallest bunches)      | Small object detection capability. M4 AP <60% suggests resolution/data inadequacy.                                 |

### Success Thresholds Rationale

| Metric          | Threshold | Why This Number?                                                                                                                                  |
| --------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| mAP@0.5         | ≥85%      | Industry standard for deployment-ready object detection. Below 80%, false positives/negatives impact census accuracy.                             |
| M2/M3 Confusion | <30%      | 30% error in 2-month vs 3-month classification = ±1 month yield forecasting uncertainty. Acceptable for early estimates, too high for production. |
| Min Class AP    | ≥70%      | Ensures model doesn't ignore difficult classes. 70% = model detects most instances, failures are explicable (occlusion, extreme cases).           |

---

## APPENDIX C: ARCHITECTURE CANDIDATES & RATIONALE

### Why These Models?

| Architecture | Params    | GFLOPS   | Tablet Inference | Key Innovation                                           | Why Test It?                                          |
| ------------ | --------- | -------- | ---------------- | -------------------------------------------------------- | ----------------------------------------------------- |
| YOLOv8n      | 3.2M      | 8.7      | ~100ms           | Efficient baseline                                       | Speed benchmark (likely too small for M2/M3/M4)       |
| YOLOv8s      | 11.2M     | 28.6     | ~200ms           | Industry standard                                        | Transfer learning baseline, proven                    |
| **YOLOv8m**  | **25.6M** | **78.9** | **~350ms**       | **Higher capacity**                                      | **Viable on tablets, may excel at M2/M3/M4**          |
| YOLOv9-c     | 25.3M     | 102      | ~380ms           | GELAN backbone (Feb 2024)                                | Architectural innovation, gradient flow               |
| YOLOv10n     | 2.3M      | 6.7      | ~90ms            | NMS-free detection (v1)                                  | Efficiency (May 2024), post-processing speed          |
| YOLOv10s     | 7.2M      | 21.6     | ~180ms           | NMS-free small (v1)                                      | Speed/accuracy balance, architectural shift           |
| **YOLOv10m** | **15.8M** | **59.1** | **~300ms**       | **NMS-free medium (v1)**                                 | **Efficiency + capacity, architectural hypothesis**   |
| **YOLO26n**  | **2.4M**  | **5.4**  | **~80ms**        | **STAL (small-target-aware), NMS-free (v2), edge-first** | **Optimized for M4 detection, Jan 2026 SOTA nano**    |
| **YOLO26s**  | **9.5M**  | **20.7** | **~160ms**       | **STAL, 43% CPU speedup, edge-optimized**                | **Best small object focus, current SOTA small**       |
| **YOLO26m**  | **20.4M** | **68.2** | **~310ms**       | **STAL, NMS-free v2, tablet-ready**                      | **Primary M4 feasibility test, Jan 2026 SOTA medium** |
| YOLOv11m     | 20.1M     | 68.0     | ~320ms           | Sep 2024 medium                                          | Baseline comparison for YOLO26m evolution             |

**All models** pre-trained on COCO (80 classes, 118k images). Transfer learning from COCO to agricultural domains is well-established in literature.

**Inference times** estimated on tablet GPU (Snapdragon 8+ Gen 1 / Exynos 2200). Actual deployment may vary ±20%.

### 

---

## 

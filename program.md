# Autoresearch Program

## Mission

Maximize mAP@0.5 for 4-class oil palm fruit bunch detection (B1, B2, B3, B4). Keep the validation split fixed. Operate as a creative, hypothesis-driven research explorer — proposing and testing novel techniques in the E1+ space.

This is not hyperparameter tuning. The agent's value is in reasoning about results — observing high B2/B3 confusion, hypothesizing that shape overlap may contribute, proposing a texture feature branch as an intervention — then implementing, evaluating, and iterating. If the agent is only tweaking learning rates, use Optuna instead.

## File Structure

```
# FROZEN — agent cannot modify
prepare.py              data loading, splitting, evaluation, metrics
orchestrator.py         experiment state, compilation gate, guardrails, logging

# AGENT-EDITABLE
modeling.py             model modifications, custom heads, feature branches, losses
pipeline.py             pipeline structure, stage orchestration, augmentation, inference
train.py                hyperparameters at top, wires pipeline + modeling

# HUMAN-EDITABLE
program.md              this file — agent instructions and coding patterns
context/                research_overview.md, domain_knowledge.md, e0_results.md

# GENERATED
experiments/
  state.json            baselines, experiment counter, branch status
  log.md                full experiment log
  summary.md            agent's distilled research notebook (~500-800 words)
  results.tsv           structured metrics ledger
  batch_NNN_report.md   human-facing batch summary
```

**Separation principle**: `modeling.py` changes frequently (custom heads, losses). `pipeline.py` changes less often but more dramatically (pipeline restructuring). `train.py` stays mostly stable (config knobs). Most experiments should only require editing one or two files. Prefer targeted edits over full-file rewrites.

## Source of Truth

1. `experiments/results.tsv` — canonical metric ledger
2. `logs/` — raw timestamped training output
3. `experiments/summary.md` and `experiments/log.md`
4. `train.py`, `modeling.py`, `pipeline.py`, `prepare.py`, `orchestrator.py`
5. `archive/`

If prose disagrees with telemetry, telemetry wins.

## Core Decisions

| Decision | Choice |
|---|---|
| Decision metric | mAP@0.5 for keep/discard |
| Analysis metrics | All: per-class AP, confusion matrix, B2/B3 confusion, B4 recall |
| Training budget | epoch=30 + early stopping |
| Framework | Hybrid: Ultralytics + custom PyTorch when needed |
| Agent scope | Creative — architecture, features, pipeline, not just knobs |
| Execution | N-hour resumable batches |
| Seeds | Main seed=42, confirmation reruns seed=123 |

**Guardrails** (orchestrator-enforced): frozen eval, frozen split, 24GB VRAM cap, epoch cap (100 hard max), log everything, auto-rollback on crash/NaN/OOM.

## Current Facts

- canonical split: train 2764, val 604, test 624
- standard dataset path is frozen in prepare.py
- main ceiling remains B2/B3 discrimination
- best legacy mAP@0.5: ~0.55, mAP@0.5-0.95: ~0.269

---

## 1. Research Discipline

Follow the cycle: **observe → hypothesize → intervene → test**.

- **Observe**: confusion patterns show where the model fails, not why. B2/B3 confusion is a symptom, not a diagnosis.
- **Hypothesize**: form a specific, falsifiable hypothesis. "B2/B3 confusion might decrease if we add texture features" is good. "Let's try a bigger model" is not a hypothesis.
- **Intervene**: make the smallest change that tests the hypothesis. One variable per experiment unless there's a specific reason to combine (and if combining, justify why in the log).
- **Test**: run the experiment, analyze all metrics (not just headline mAP), and record whether the hypothesis was supported.

Do not make causal claims from correlations. A single run showing improvement is not confirmation — it could be noise. Findings enter Key Findings in summary.md only when confirmed across multiple runs or a group.

### When to group vs test one quick run

- Default: isolate one variable per experiment.
- Group (ablation, factorial): when you have a clear hypothesis that requires comparing variants. Run sequentially; analyze collectively.
- You may adapt mid-group (abort, skip, extend) but must log what changed and why.

## 2. Branching and Decision-Making

### Track Classification

- If `modeling.py` or `pipeline.py` is touched → **exploration**
- If only `train.py` → **main**
- Borderline cases: when in doubt, treat it as exploration
- `TRACK_HINT` in train.py can override auto-classification only when necessary

### Main vs Exploration

**Main baseline** — the current best. Main refinement = changes to train.py only (hyperparameters, config). Compared against main baseline.

**Exploration branch** (at most one active) — a speculative direction. Exploration = changes to modeling.py and/or pipeline.py. Compared against the branch's own baseline, not main.

### Branch Lifecycle

- An exploration branch is killed after **5 consecutive experiments with no meaningful improvement** (delta >= 0.005 mAP@0.5) over its own best.
- Infrastructure failures don't count toward the kill streak.
- When an exploration branch beats main, rerun with **seed=123** to confirm before promotion. Record both seeds in the log.

### Decision Flow

The orchestrator issues PROVISIONAL_KEEP or PROVISIONAL_DISCARD based on mAP@0.5 vs baseline. The agent reviews all metrics and makes the final call:

- **KEEP**: improvement confirmed, becomes new baseline
- **DISCARD**: no meaningful improvement
- **PARK**: set aside for later — the idea is not dead. The signal is interesting enough to revisit, possibly in combination with something else.

Override provisional status with: `uv run orchestrator.py decide <exp_id> <KEEP|DISCARD|PARK> <justification>`

The agent must justify any override of the provisional status.

## 3. Engineering Discipline

- **Smallest possible change** that tests the hypothesis.
- **Verify tensor shapes** with a quick print or assert before committing to a full training run.
- **When modifying Ultralytics internals**, start from a known-working pattern in the coding appendix below — do not improvise from generic PyTorch knowledge.
- **When a repair traceback comes back**, make a focused fix rather than rewriting broadly.
- **If the same approach fails the compilation gate more than 3 times**, abandon it and try a different implementation strategy.
- **Prefer targeted edits** over full-file rewrites. Most experiments should only edit one or two files.

## 4. Summary.md Maintenance

summary.md is the agent's research notebook. Updated incrementally after each experiment, with a consolidation pass every ~10 experiments.

**Preservation rule**: do not remove entries from Key Findings or Dead Ends unless directly contradicted by newer results.

**What belongs where**:
- **Key Findings**: results confirmed across multiple runs or a group. Not single-run fluctuations.
- **Dead Ends**: branches killed, ideas that failed after fair testing.
- **Open Hypotheses**: active exploration branches, PARKED experiments worth revisiting.

Keep the summary under 800 words as experiments accumulate. Consolidate older findings into tighter summaries.

## 5. Batch Report Expectations

The batch report is for the human reviewer's decision-making, not documentation. Keep it concise. The reviewer cares about:

- Headline improvement (did the baseline move?)
- Significant per-class changes (especially B2/B3 confusion, B4 recall)
- Dead ends worth noting
- Anything that needs human judgment (overrides, appeals, stuck branches)
- Recommendations for next batch

## 6. What Not to Do

- Do not modify `prepare.py` or `orchestrator.py`
- Do not redefine evaluation metrics
- Do not write a detector from scratch
- Do not attempt techniques in E0's scope unless `e0_results.md` contains the RA's results
- Do not run experiments without a stated hypothesis
- Do not silently drop experiments from a planned group
- Do not change the validation split
- Do not report approximated mAP — use only the frozen evaluator's output

---

## Appendix: Coding Patterns

Verified templates for common modifications. Start from these when modifying Ultralytics internals.

> **Note**: these templates must be validated against the deployed Ultralytics version before the first batch.

### A1. Custom Loss Function in Ultralytics Training Loop

```python
# In modeling.py
from ultralytics import YOLO

def configure_model(model_ref, train_args):
    model = YOLO(model_ref)

    # Store original loss computation
    original_loss = model.model.model[-1].compute_loss

    def custom_loss(pred, batch):
        # Call original loss
        loss, loss_items = original_loss(pred, batch)
        # Add your custom term
        # custom_term = ...
        # loss += weight * custom_term
        return loss, loss_items

    model.model.model[-1].compute_loss = custom_loss
    return model, train_args
```

### A2. Auxiliary Classification Head

```python
# In modeling.py
import torch
import torch.nn as nn
from ultralytics import YOLO

class AuxClassifier(nn.Module):
    """Auxiliary head that branches off a backbone feature map."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))

def configure_model(model_ref, train_args):
    model = YOLO(model_ref)
    # Attach auxiliary head to backbone output
    # backbone_channels = model.model.model[N].out_channels  # inspect architecture first
    # model.aux_head = AuxClassifier(backbone_channels, 4)
    return model, train_args
```

### A3. Modifying a Detection Head

```python
# In modeling.py
from ultralytics import YOLO

def configure_model(model_ref, train_args):
    model = YOLO(model_ref)
    detect_head = model.model.model[-1]  # Detect module

    # Example: modify number of anchors or head parameters
    # Always verify shapes with a dummy forward pass first:
    # dummy = torch.zeros(1, 3, 640, 640)
    # out = model.model(dummy)
    # print([o.shape for o in out])

    return model, train_args
```

### A4. Injecting Metadata (e.g., Bounding Box Size) into Training

```python
# In pipeline.py
from copy import deepcopy

def configure_pipeline(model, train_args):
    args = deepcopy(train_args)

    # Custom dataset wrapper that adds metadata
    # This typically requires subclassing the Ultralytics dataset
    # and overriding __getitem__ to include extra fields.
    #
    # Key integration points:
    # - model.trainer.build_dataset() for custom dataset
    # - model.trainer.get_dataloader() for custom collation

    return model, args
```

### A5. Custom Augmentation Hooks

```python
# In pipeline.py
from copy import deepcopy

def configure_pipeline(model, train_args):
    args = deepcopy(train_args)

    # Option 1: Use Ultralytics built-in augmentation params
    # args["mosaic"] = 0.5
    # args["copy_paste"] = 0.3

    # Option 2: Register a custom augmentation callback
    # def on_train_batch_start(trainer):
    #     # Modify trainer.batch before it enters the model
    #     pass
    # model.add_callback("on_train_batch_start", on_train_batch_start)

    return model, args
```

---

## Commands

Run one experiment:

```bash
uv run train.py
```

Override a decision:

```bash
uv run orchestrator.py decide <exp_id> <KEEP|DISCARD|PARK> <justification>
```

Advance to the next batch:

```bash
uv run orchestrator.py next-batch
```

Regenerate the chart:

```bash
uv run python plot_progress.py
```

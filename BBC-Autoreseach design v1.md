# BBC-Autoresearch: Design Document

---

## 1. Concept

Adapt [autoresearch](https://github.com/karpathy/autoresearch) for the BBC project. An AI agent runs in parallel with E0 as a creative, hypothesis-driven research explorer - proposing and testing novel techniques (architecture changes, feature engineering, pipeline restructuring, custom losses) in the E1+ space. It runs in N-hour batches; between batches, the human reviews results, updates domain context (including E0 findings), and steers exploration priorities. E0 findings inform the agent; agent discoveries become candidates for the RA to validate.

This is not hyperparameter tuning. The agent's value is in reasoning about results - observing high B2/B3 confusion, hypothesizing that shape overlap may contribute, proposing a texture feature branch as an intervention - then implementing, evaluating, and iterating. If the agent is only tweaking learning rates, use Optuna instead.

---

## 2. File Structure

```
# FROZEN - agent cannot modify
prepare.py              - data loading, tree-level splitting, evaluation, metrics
orchestrator.py         - experiment state tracking, compilation gate, guardrails,
                          logging, rollback on failure
dataset/                - images + annotations (read-only)

# AGENT-EDITABLE
modeling.py             - model modifications, custom heads, feature branches,
                          loss function definitions
pipeline.py             - pipeline structure (one-stage, two-stage), stage
                          orchestration, augmentation strategy, inference flow
train.py                - hyperparameters at the top, wires pipeline + modeling

# HUMAN-EDITABLE
program.md              - agent instructions, guardrails, coding patterns appendix
context/                - research_overview.md, domain_knowledge.md, e0_results.md

# GENERATED
experiments/
  state.json            - baselines, experiment counter, branch status
  log.md                - full experiment log (on disk, not in agent context)
  summary.md            - agent's distilled research notebook (~500-800 words,
                          always in agent context)
  results.tsv           - structured metrics: exp_id, track, map50, map50_95,
                          per-class AP (B1-B4), B2_B3_confusion, B4_recall,
                          B4_precision, time, vram, epoch, seed, status
  batch_NNN_report.md   - human-facing batch summary
```

**Separation principle**: `modeling.py` changes frequently (custom heads, losses). `pipeline.py` changes less often but more dramatically (pipeline restructuring). `train.py` stays mostly stable (config knobs). Most experiments should only require editing one or two files. The agent should prefer targeted edits over full-file rewrites.

**Frozen boundary**: `prepare.py` owns evaluation and data splitting. `orchestrator.py` owns experiment mechanics. The agent cannot touch these.

---

## 3. LLM vs Code Responsibilities

**The agent does the thinking**: analyze results across all metrics (not just headline mAP), form hypotheses, propose interventions, interpret failure modes, write analysis, decide what to explore next. The agent follows observe -> hypothesize -> intervene -> test: confusion patterns suggest hypotheses, hypotheses motivate interventions, experiments test whether interventions help.

**The orchestrator does the bookkeeping**: track experiment state, run the compilation gate, enforce guardrails, manage baselines, log metrics, rollback on failure. The agent reads `state.json` fresh each time and never needs to maintain state itself.

---

## 4. Context and Memory

The agent works from a layered memory, not the full experiment history.

**Always in context**: `program.md` (~1,000 words), `context/` files (~2,500 words total), `state.json`, `summary.md` (~500-800 words), last 2-3 experiment entries from `log.md`.

**On disk** (agent can grep): full `log.md`, `results.tsv`.

**summary.md** is the critical piece - the agent's research notebook. Updated incrementally after each experiment, with a consolidation pass every ~10 experiments. Preservation rule: do not remove entries from Key Findings or Dead Ends unless directly contradicted by newer results. Only include findings with sufficient confidence (confirmed across multiple runs or a group), not single-run fluctuations. Structure:

```markdown
## Current Best
Main: Exp 42, YOLOv9c + focal loss, mAP@0.5 = 0.62

## Key Findings
- Copy-paste 0.3 improved B4 recall by 8% (Exp 15-18)
- Ordinal loss reduced B2/B3 confusion 38%->29% but hurt B1 (Exp 23)

## Dead Ends
- Position features: no signal after 5 experiments (Group G03)

## Open Hypotheses
- Texture branch promising, needs more iterations
- Combining copy-paste + ordinal loss untested
```

**Context files should be actionable, not encyclopedic.** "B2 and B3 overlap 84% in bounding box size" is useful. Three paragraphs on plantation management is not.

**Code context**: as editable files grow, the agent sees the region being edited plus relevant call sites; untouched modules can be summarized to signatures.

---

## 5. Core Design Decisions

| Decision | Choice |
|---|---|
| Decision metric | mAP@0.5 for keep/discard |
| Analysis metrics | All: per-class AP, confusion matrix, B2/B3 confusion, B4 recall |
| Training budget | epoch=30 + early stopping |
| Framework | Hybrid: Ultralytics + custom PyTorch when needed |
| Agent scope | Creative - architecture, features, pipeline, not just knobs |
| Execution | N-hour resumable batches |
| Seeds | Main seed=42, confirmation reruns seed=123 |

**Guardrails** (orchestrator-enforced): frozen eval, frozen split, 24GB VRAM cap, epoch cap, log everything, auto-rollback on crash/NaN/OOM.

**Decision flow**: the orchestrator issues PROVISIONAL_KEEP or PROVISIONAL_DISCARD based on mAP@0.5 vs baseline. The agent reviews all metrics and makes the final call: KEEP, DISCARD, or PARK (set aside for later, idea not dead). The agent must justify any override of the provisional status.

---

## 6. Experiment Loop

### Compilation Gate

Before any experiment is counted, the orchestrator runs: syntax check, import validation, and a mandatory smoke test (one forward pass through the full pipeline). Failures return the traceback to the agent as a repair task. Repair cycles are not counted as experiments and do not appear in `results.tsv`.

Infrastructure failures during training (NaN, OOM, runtime crash) are categorized separately from scientific results. They trigger auto-rollback and a diagnostic message to the agent. They are logged but do not count as evidence against the hypothesis.

### Branching

The agent maintains two tracks:

**Main baseline** - the current best. Main refinement = changes to `train.py` only (hyperparameters, config). Compared against main baseline.

**Exploration branch** (at most one) - a speculative direction. Exploration = changes to `modeling.py` and/or `pipeline.py`. Compared against the branch's own baseline, not main.

Classification rule: if `modeling.py` or `pipeline.py` is touched, it's exploration. If only `train.py`, it's main. Borderline cases are flagged for human review.

An exploration branch is killed after **5 consecutive experiments with no meaningful improvement** (delta >= 0.005 mAP@0.5) over its own best. Infrastructure failures don't count toward this limit. When an exploration branch beats main, the agent reruns with **seed=123** (all regular experiments use **seed=42**) to confirm before promotion. Both seeds are recorded in the log.

### Single Experiments and Groups

The agent proposes experiments individually or as groups (ablations, factorial designs, architecture comparisons). Groups run sequentially. The agent can adapt mid-group (abort, skip, extend) but must log what changed and why. Results are analyzed collectively at the group level.

### Batch Execution

Before the first real batch, run a scripted dry batch of 3-5 pre-planned experiments to validate the orchestrator pipeline end-to-end. Treat the first 1-2 real batches as calibration (tuning operational parameters, not maximizing science).

Each steady-state batch runs as a mix of singles and groups. When the time budget expires, the agent produces a batch report. Between batches, the human reviews the report and updates `context/` and `program.md`.

---

## 7. Exploration Scope

**Seed ideas (E1-E4)**: E1 size features, E2 position features, E3 size + position combined, E4 texture features. Starting points, not a rigid sequence.

**Beyond E1-E4**: contrastive pre-training on FFB crops, attention mechanisms for occlusion, multi-view-inspired augmentation, learned augmentation policies, maturity-aware anchor design, curriculum learning, synthetic occlusion augmentation. Techniques already in E0 (focal loss, ordinal classification, copy-paste, two-stage, confidence thresholds) are out of scope until the RA completes the relevant phase and results are in `e0_results.md`.

---

## 8. Logging

**Experiment log entry** (in `log.md`):

```
## Experiment [NNN]: [Title]
Track: main / exploration:[name]
Hypothesis: [what and why]
Change: [what was modified, which module(s)]
Baseline: Exp [X], mAP@0.5 = X.XX

Results:
mAP@0.5: X.XX (delta) | B1=X.XX, B2=X.XX, B3=X.XX, B4=X.XX
B2/B3 confusion: X.X% | B4 recall: X.XX | Time: Xm | VRAM: X.XGB

Analysis: [what worked, what didn't, why - using all metrics]

Decision:
Provisional: PROVISIONAL_KEEP / PROVISIONAL_DISCARD
Final: KEEP / DISCARD / PARK
Justification: [required if overriding provisional]

Next: [what to try and why]
```

Groups use the same format with a runs table and collective analysis. Mid-group adaptations must be documented.

**Batch report** (`batch_NNN_report.md`):

```
## Batch Report - Batch [NNN] ([date], [duration])

Summary: [2-3 sentences - experiments run, baseline change, branches]

Significant Findings: [3-5 findings with experiment refs and deltas]

Dead Ends: [branches killed, ideas that failed]

Current State:
Main: Exp [X], mAP@0.5 = X.XX
Exploration: [name] ([N] exps in) / none active

Infrastructure: [experiments | repairs | OOM | NaN | crashes]

Recommended Next: [agent's suggestions for next batch]

Items for Human Review: [overrides, adaptations, flags, appeals]
```

---

## 9. program.md Requirements

`program.md` is the single most important document for operational success. It is both the agent's research brief and its engineering manual. Writing it is a separate, substantial task that requires the actual codebase running on the pod. Below is what it must cover.

**Research discipline**: the observe -> hypothesize -> intervene -> test cycle. The agent should not make causal claims from correlations (confusion patterns show where the model fails, not why). How to write useful analysis - what makes a good hypothesis, what counts as evidence for or against. When to commit to a group vs. test one quick run first. The default should be: isolate one variable per experiment unless there's a specific reason to combine changes. If combining, justify why in the log.

**Branching and decision-making**: how to decide whether an idea is main refinement or exploration. When in doubt, treat it as exploration (the orchestrator will flag if it disagrees). How to use PARK - it means "this didn't work yet but the signal is interesting enough to revisit, possibly in combination with something else." When to override a provisional decision and how much justification is expected.

**Engineering discipline**: prefer the smallest possible change that tests the hypothesis. Verify tensor shapes with a quick print or assert before committing to a full training run. When modifying Ultralytics internals, start from a known-working pattern in the coding appendix - do not improvise from generic PyTorch knowledge. When a repair traceback comes back, make a focused fix rather than rewriting broadly. If the same approach fails the compilation gate more than 3 times, abandon it and try a different implementation strategy.

**Coding patterns appendix**: verified, working templates for the most common modifications. At minimum: adding a custom loss function to the Ultralytics training loop, attaching an auxiliary classification head or feature branch, modifying a detection head without breaking the existing training path, injecting additional metadata (e.g., bounding box size) into the training pipeline, and extending or customizing augmentation hooks. Each template must be tested against the actual Ultralytics version deployed on the pod before the first batch. These templates are the agent's primary defense against repair-loop churn.

**Summary.md maintenance**: what belongs in Key Findings vs. Open Hypotheses (confirmed results vs. tentative signals). The preservation rule - do not remove findings unless directly contradicted. How to keep the summary under 800 words as experiments accumulate.

**Batch report expectations**: what the human reviewer cares about (headline improvement, significant per-class changes, dead ends worth noting, anything that needs human judgment). Keep it concise - the report is for decision-making, not documentation.

**What not to do**: do not modify `prepare.py` or `orchestrator.py`. Do not redefine evaluation metrics. Do not write a detector from scratch. Do not attempt techniques already in E0's scope unless `e0_results.md` contains the RA's results for that technique. Do not run experiments without a stated hypothesis. Do not silently drop experiments from a planned group.


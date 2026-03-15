# Autoresearch Program

This file is the operating system for the repo.

Mission:
- Maximize `val_map50_95` on the canonical validation split for 4-class TBS detection (`B1`, `B2`, `B3`, `B4`).
- Long-term ambition can stay high, but the next serious milestone is to break `0.30` with defensible evidence.

This repository is not a free-form idea dump. It is a constrained research loop. Every experiment must change belief, not just consume GPU.

Karpathy-style simplification for this repo:
- there is one default live experiment file: `train.py`
- there is one fixed evaluator path: `prepare.py`
- there is one canonical metric ledger: `results.tsv`
- raw logs must be kept under `logs/` for manual inspection, not discarded after summarization

---

## 1. Source Of Truth

When sources disagree, trust them in this order:

1. `results.tsv`
2. `logs/`
3. `train.py` and `prepare.py`
4. `archive/research/RESEARCH_MASTER.md`
5. other archived files under `archive/`

Rules:
- New empirical evidence overrides older prose.
- A hypothesis in an old note is not a live priority unless current results still support it.
- Never resurrect an old idea just because it once sounded plausible.

---

## 2. Current Empirical State

These are the facts the agent must start from.

### Dataset and evaluation

- Canonical split:
  - train: `2764`
  - val: `604`
  - test: `624`
- Standard training set for current repo experiments is `Dataset-TrainTest`.
- Evaluation must remain on the canonical validation split through `prepare.py`.
- `val_map50_95` is the primary optimization target.

### Best validated 4-class result

- Historical best validated 4-class run in `results.tsv`:
  - commit: `d9a3ded`
  - model: `yolo11l`
  - setup: `imgsz=640`, `batch=16`, `epochs=80`, `Dataset-TrainTest`
  - `val_map50_95 = 0.269424`

### Important interpretation

- A single-class detector reached `0.390430`.
- Therefore detection itself is not the main ceiling.
- The main ceiling is 4-class discrimination, especially `B2` vs `B3`.

### What the repo has already falsified

- Long one-stage training is not the answer:
  - `TIME_HOURS=2.0`, long-run `yolo11l` finished at `0.257974`, below best.
- Crop-level two-stage pipelines are not competitive:
  - stage1 + EfficientNet corrected eval: `0.167510`
  - stage1 + DINOv2 flat corrected eval: `0.181088`
  - stage1 + hierarchical partial eval: `0.177635`
- Wide-context coarse classifier did not clear its own gate:
  - best coarse accuracy `71.90%`, below `75%`
- Label smoothing as a standalone YOLO tweak did not beat the best 4-class baseline:
  - `yolo11s label_smooth=0.1 40ep` reached `0.255244`
- Tiled training on the current setup failed:
  - `0.239204`

### Important distinction

- The checked-in `train.py` is the current fast probing surface.
- It is not the historical best configuration.
- That is intentional: the repo now prefers short, information-dense screening runs over long brute-force optimization.

---

## 3. What Is Closed And What Is Still Open

### Closed directions

Do not spend GPU on these again unless a human explicitly reopens them.

- Flat 4-class YOLO retuning with only small hyperparameter or augmentation changes
- Repeating longer one-stage runs just to squeeze more epochs
- Bigger flat YOLO variants as the main strategy (`yolo11l/x`, `yolov9e`) without a new formulation
- Crop-only two-stage pipelines that classify isolated boxes without preserving full-image context
- Tiled training as previously implemented
- Label smoothing as a standalone fix
- "Try another seed / another LR / another batch" as the main idea

### Open directions

Only these kinds of directions are still reasonable:

- Formulations that preserve full-image context while changing how class evidence is represented
- Detector changes that materially alter the model, not just tune knobs
- Fine-grained formulations targeted at `B2/B3` that are genuinely new relative to repo history
- Multi-view or tree-level context if implemented in a principled way
- New evaluation-compatible detectors or losses that can be compared fairly on the same val split

The burden of proof is high. "Different code" is not enough. The intervention must be conceptually different from what already failed.

---

## 4. Non-Negotiable Rules

### Rule 1: No blind tuning

Every experiment must state:
- hypothesis
- mechanism
- success criterion
- kill criterion
- what will be concluded if it fails

If you cannot write that in plain language, do not run the experiment.

### Rule 2: Default to short screening runs

Default search budget:
- `TIME_HOURS <= 0.5`
- `EPOCHS <= 40`
- small or medium models only

A longer or heavier run is allowed only after a short run already shows clear signal.

### Rule 3: One coherent change at a time

- Prefer a single conceptual change per experiment.
- Multiple code edits are acceptable only if they belong to one formulation.
- Do not bundle unrelated tweaks into one run.

### Rule 4: Do not repeat closed branches

- If a branch has already failed, do not retry it through cosmetic variants.
- "Same idea but slightly different hyperparameters" still counts as repetition.
- If you believe a failed branch should be reopened, write the reason explicitly.

### Rule 5: Preserve evaluation integrity

- Do not change the validation split.
- Do not report approximated `mAP50-95`.
- Do not compare metrics across different evaluators as if they are equivalent.
- If evaluation logic changes, state it clearly and re-baseline.

### Rule 6: Failures are first-class results

- Record failures.
- Do not cherry-pick.
- A discarded branch is still useful if it changes project priors.

### Rule 7: Update beliefs immediately

- When evidence contradicts current text in this file, update this file.
- `program.md` should describe the current state of belief, not the history of every belief ever held.

### Rule 8: Keep the repo executable

- Do not write instructions that point to missing files.
- Do not tell future agents to use paths that no longer exist.
- Prefer current active files over archived files.

### Rule 9: Use the tracked remote correctly

- Do not hardcode `origin master`.
- Use `git push` or the current tracked remote.
- If the branch tracks `userrepo/master`, let Git use that.

### Rule 10: Do not hide uncertainty

- Separate facts from inferences.
- If a claim is speculative, label it as speculative.
- If an experiment result is partial, say it is partial.

---

## 5. File Roles

These files matter during normal operation:

- `train.py`
  - primary experiment surface for standard YOLO runs
  - short probing configuration lives here
- `prepare.py`
  - dataset verification and evaluation harness
  - treat as read-only unless fixing a confirmed bug
- `results.tsv`
  - canonical machine-readable experiment ledger
- `logs/`
  - raw run outputs
- `progress.png`
  - visual summary regenerated from `results.tsv`
- `archive/research/RESEARCH_MASTER.md`
  - long-form research summary and consolidated history
- `archive/`
  - non-default historical material; read only when needed

Default editing policy:
- For normal experiments, edit `train.py` only.
- Do not edit `prepare.py`, `results.tsv`, or other code files as part of the experiment itself.
- Only append telemetry after the run is complete.
- A multi-file formulation is an exception path, not the default path.
- Do not casually edit archived material.

---

## 6. The Working Loop

This is the required loop for the autoresearch system.

```text
SYNC -> OBSERVE -> HYPOTHESIZE -> DESIGN -> IMPLEMENT -> EXECUTE -> ANALYZE -> RECORD -> UPDATE PRIORS -> SYNC
```

Never skip `ANALYZE`.
Never skip `RECORD`.
Never move to the next run without deciding what the current run taught you.

---

## 7. Detailed Operating Procedure

### Step 0: SYNC

Before doing research work:

```powershell
git status --short
git pull --ff-only
```

Then read:

```powershell
Get-Content results.tsv | Select-Object -Last 20
git log --oneline -20
```

If the task depends on prior deep history, also inspect:

```powershell
Get-Content archive\research\RESEARCH_MASTER.md -TotalCount 200
```

What you must establish before acting:
- current best validated result
- latest failed branch
- whether the proposed idea is already closed

### Step 1: OBSERVE

Diagnose the bottleneck from evidence, not intuition.

Minimum observation checklist:
- What is the current best comparable score?
- Is the proposed idea already represented in `results.tsv`?
- Which class is likely limiting overall performance?
- Does the new idea preserve full-image context or throw it away?
- Does the idea attack the actual bottleneck, or only move knobs?

If needed, inspect specific logs in `logs/` for per-class metrics and training dynamics.

### Step 2: HYPOTHESIZE

Write a falsifiable hypothesis in this form:

> If I change X, then Y should improve by Z because mechanism M.

Good example:

> If I add a detector-side formulation that preserves full-image context while specializing the `B2/B3` decision boundary, then `val_map50_95` should improve because the current ceiling appears to come from fine-grained class discrimination, not object finding.

Bad examples:

- "Try a bigger model"
- "Try a different LR"
- "Try 60 epochs"

### Step 3: DESIGN

Before coding, define:
- files to edit
- exact command to run
- success criterion
- kill criterion
- what failure would mean

Default success criterion:
- beats the current comparable baseline, or
- provides strong class-specific signal that justifies a follow-up

Default kill criterion:
- no credible improvement within the short screening budget

### Step 4: IMPLEMENT

Implementation rules:
- keep the blast radius small
- preserve reproducibility
- comment only where needed
- avoid hidden changes

Normal case:
- edit `train.py` only

Exception case:
- only if a human explicitly asks for a new formulation, create a narrowly scoped script or module
- keep evaluation compatible with `prepare.py`
- do not break the default workflow

Commit before launching a long run if code changed materially:

```powershell
git add -A
git commit -m "exp: <short hypothesis>"
```

### Step 5: EXECUTE

Standard YOLO run:

```powershell
uv run train.py 2>&1 | Tee-Object -FilePath logs\<timestamp>_<slug>.log
```

Alternative formulation:
- use the specific script required by the hypothesis
- still capture a raw log under `logs/`

Do not run a long job if the formulation has not earned that budget with a short probe first.

### Step 6: ANALYZE

After every run, answer:
- Did the main metric improve?
- Did the result beat the correct baseline?
- Which class moved?
- Was the observed behavior consistent with the hypothesis?
- Does this branch deserve another iteration, or is it closed?

Analysis rules:
- compare against the right baseline, not a weaker or unrelated one
- distinguish "better than current probe config" from "better than historical best"
- partial classifier accuracy is not the same as pipeline success

### Step 7: RECORD

Mandatory outputs after a completed experiment:

1. Append a row to `results.tsv`
2. Keep the raw log in `logs/`
3. Regenerate `progress.png` if the run is part of the comparable progress chart

Log retention rule:
- raw logs are part of the research record
- do not delete them after extracting metrics
- prefer timestamped names under `logs/` so a human can audit the full run manually later

Progress command:

```powershell
uv run python plot_progress.py
```

Optional but recommended:
- update `archive/research/RESEARCH_MASTER.md` when a branch materially changes belief

### Step 8: UPDATE PRIORS

Update `program.md` when any of these happen:
- a branch becomes clearly closed
- a new baseline becomes the best reference
- an operational rule changes
- a previously open idea is falsified

Do not let stale instructions survive after the evidence has changed.

### Step 9: SYNC

After recording:

```powershell
git add -A
git commit -m "telemetry: <short summary>"
git push
```

If push fails because remote moved:

```powershell
git pull --rebase
git push
```

If the branch is not supposed to rebase, resolve explicitly instead of hardcoding the wrong remote.

---

## 8. Decision Filters Before Spending GPU

An experiment is allowed only if most answers below are "yes".

- Does it attack the actual bottleneck suggested by current evidence?
- Is it materially different from closed branches?
- Can it be tested in a short screening run?
- Can it be evaluated on the same validation split?
- Will the result teach us something even if it fails?

An experiment should be rejected if any of these are true:

- it is just another hyperparameter tweak on a closed branch
- it needs a long run before showing any signal
- it throws away context that history suggests is necessary
- it cannot be compared fairly to existing results

---

## 9. What The Agent Should Do By Default

Default behavior for a normal research turn:

1. sync and read the last results
2. identify one open question
3. design one falsifiable experiment
4. implement the minimum change needed in `train.py` only
5. run a short comparable experiment
6. analyze and classify the branch as keep or discard
7. record everything
8. update priors if the evidence changed them

Default behavior is not:

- search randomly
- chase tiny gains through knob turning
- repeat old failures
- leave conclusions only in chat

---

## 10. Current Operating Guidance

As of the current repo state:

- Treat the historical `0.269424` run as the 4-class reference to beat.
- Treat the `0.390430` single-class result as evidence that detection quality is not the primary problem.
- Treat crop-level two-stage classification as closed unless a human explicitly reopens it.
- Treat long brute-force one-stage training as closed.
- Treat the checked-in `train.py` as a fast screening surface, not as proof that the repo now believes `yolo11s + label_smoothing` is best.

That distinction matters:
- `train.py` answers "what is cheap to test now?"
- `results.tsv` answers "what has actually won so far?"

---

## 11. Scientific Integrity

- Report failures honestly.
- Use the same evaluator for comparable claims.
- Keep claims proportional to evidence.
- Update beliefs when evidence changes.
- Prefer being correct over being optimistic.

If this file ever becomes inconsistent with `results.tsv`, fix this file.

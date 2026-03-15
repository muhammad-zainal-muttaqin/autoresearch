# Autoresearch Program

Mission:
- maximize `val_map50_95`
- keep the validation split fixed
- make decisions from evidence, not guesswork

This repo has one default live experiment file: `train.py`.

The contract is simple:
- edit `train.py`
- run `uv run train.py`
- evaluate through `prepare.py`
- record the result in `results.tsv`
- keep the full raw log in `logs/`

## Source Of Truth

Trust sources in this order:

1. `results.tsv`
2. `logs/`
3. `train.py` and `prepare.py`
4. `archive/research/RESEARCH_MASTER.md`
5. other files in `archive/`

If prose disagrees with telemetry, telemetry wins.

## Current Facts

- canonical split: train `2764`, val `604`, test `624`
- standard training set for current comparable runs: `Dataset-TrainTest`
- best validated 4-class result in `results.tsv`: `0.269424`
- single-class detector reached `0.390430`
- main ceiling is class discrimination, especially `B2/B3`

Closed branches:
- long brute-force one-stage training
- crop-only two-stage pipelines
- tiled training as previously implemented
- label smoothing as a standalone fix
- small hyperparameter retuning of closed branches

## Hard Rules

1. Edit `train.py` only during normal experimentation.
2. Treat `prepare.py` as fixed unless there is a confirmed evaluator bug.
3. Every run must have a falsifiable hypothesis.
4. Default budget is short: `TIME_HOURS <= 0.5`, `EPOCHS <= 40`.
5. Do not repeat closed branches through cosmetic variants.
6. Do not change the validation split.
7. Do not report approximated `mAP50-95`.
8. Record failures honestly.
9. Keep raw logs under `logs/`; do not delete them after summarizing.
10. If `program.md` disagrees with `results.tsv`, fix `program.md`.

## Operating Loop

```text
SYNC -> OBSERVE -> HYPOTHESIZE -> EDIT -> RUN -> ANALYZE -> RECORD -> SYNC
```

Never skip `ANALYZE`.
Never move to the next experiment before deciding what the current run taught you.

## Operating Procedure

### 1. Sync

```powershell
git status --short
git pull --ff-only
Get-Content results.tsv | Select-Object -Last 20
git log --oneline -20
```

Read `archive/research/RESEARCH_MASTER.md` only if you need older context.

### 2. Observe

Before changing anything, answer:
- what is the current comparable score to beat?
- is this idea already in `results.tsv`?
- does this idea attack the actual bottleneck?
- does this idea preserve the fixed evaluation setup?

### 3. Hypothesize

Write a hypothesis in this form:

> If I change X, then Y should improve because Z.

If you cannot state the mechanism, do not run it.

### 4. Edit

Normal case:
- edit constants at the top of `train.py`

Exception case:
- only when a human explicitly asks for a new formulation, create new files
- keep evaluation compatible with `prepare.py`

### 5. Run

```powershell
uv run train.py 2>&1 | Tee-Object -FilePath logs\<timestamp>_<slug>.log
```

### 6. Analyze

After the run, answer:
- did `val_map50_95` improve?
- did the result beat the right baseline?
- which class moved?
- was the result consistent with the hypothesis?
- should this branch stay open or close?

### 7. Record

Mandatory:
- append a row to `results.tsv`
- keep the full log in `logs/`
- regenerate `progress.png` when the run belongs on the chart

```powershell
uv run python plot_progress.py
```

### 8. Sync

```powershell
git add -A
git commit -m "telemetry: <short summary>"
git push
```

## Default Behavior

Do:
- make one coherent change
- run a short comparable experiment
- update beliefs from evidence

Do not:
- search randomly
- chase tiny gains by knob turning
- reopen closed branches without a real reason
- hide failures

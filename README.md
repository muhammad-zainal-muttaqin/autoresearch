# autoresearch

![teaser](progress.png)

This repository runs Karpathy-style autonomous research for 4-class oil palm fruit bunch detection.

The live surface is intentionally narrow:
- `train.py` is the editable experiment spec
- `research.py` is the only second editable file for nontrivial model or pipeline changes
- `prepare.py` is the frozen evaluator
- `orchestrator.py` is the frozen runtime and state manager

The primary decision metric is `val_map50_95`.

## Live Surface

```text
train.py                editable experiment metadata + config
research.py             editable research hook for nontrivial changes
prepare.py              frozen dataset/evaluation harness
orchestrator.py         frozen runtime, gates, state, logging, batching
program.md              operating brief
context.md              human steering context
results.tsv             canonical ledger
experiments/            machine-managed state, summary, log, reports
logs/                   raw run logs
plot_progress.py        regenerate progress.png from results.tsv
progress.png            visual progress chart
archive/                non-default historical material
```

Datasets and generated training outputs under `runs/` are local-only artifacts and should not be committed.

## Quick Start

```bash
uv sync
uv run prepare.py
uv run train.py
```

Open the next batch explicitly with:

```bash
uv run orchestrator.py next-batch
```

## Operating Rules

- edit `train.py` for normal experiments
- edit `research.py` only when the experiment needs structural changes
- do not edit `prepare.py` or `orchestrator.py` during normal agent operation
- keep raw logs in `logs/`
- treat `results.tsv` as the canonical metric ledger
- treat `experiments/` as machine-managed memory, not the main experiment surface

## Dataset Expectations

The dataset must follow standard YOLO structure:

```text
Dataset-YOLO/
  data.yaml
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

## Analysis

Use `archive/analysis.ipynb` for ad hoc inspection. Regenerate `progress.png` with `uv run python plot_progress.py` after recording a comparable result.

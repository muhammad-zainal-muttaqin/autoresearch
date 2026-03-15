# autoresearch

![teaser](progress.png)

This repository applies the autoresearch idea to a practical computer vision task: autonomous tuning of a YOLO detector for oil palm fruit bunch images.

The project trains and evaluates a detector on 4 classes: `B1`, `B2`, `B3`, and `B4`.

The live surface is intentionally small:
- `train.py` is the one experiment surface
- `prepare.py` is the fixed dataset and evaluation harness
- `program.md` is the operating brief
- `results.tsv` is the canonical metric ledger
- `logs/` keeps full raw outputs for manual inspection

Everything else is telemetry or archive material.

The primary optimization target is `val_map50_95`.

Historical research notes and older assets live under `archive/`.

## Live Surface

```text
train.py                editable experiment surface
prepare.py              dataset constants, verification, evaluation
program.md              operating brief
results.tsv             canonical experiment ledger
plot_progress.py        regenerate progress.png from results.tsv
progress.png            visual progress chart
logs/                   raw experiment logs
archive/                non-default historical material
```

Datasets and generated training outputs under `runs/` are local-only artifacts and should not be committed.

## Requirements

- Python 3.10+
- `uv`
- a CUDA-capable NVIDIA GPU

## Quick Start

```bash
uv sync
uv run prepare.py
uv run train.py
```

Default paths are resolved from the repository root. If the dataset is stored outside the repository, point the code at it with `YOLO_DATASET_DIR`.

PowerShell example:

```powershell
$env:YOLO_DATASET_DIR = "D:\datasets\Dataset-YOLO"
uv run prepare.py
uv run train.py
```

## Operating Rules

- edit `train.py` only during normal experimentation
- treat `prepare.py` as read-only unless there is a confirmed evaluator bug
- append metrics to `results.tsv` after each completed run
- keep raw logs in `logs/`
- keep non-default material in `archive/`

## Dataset Expectations

The dataset must follow standard YOLO directory structure:

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

`prepare.py` validates that each split has both image and label directories before training starts.

## Analysis

Use `archive/analysis.ipynb` for ad hoc inspection. Regenerate `progress.png` with `uv run python plot_progress.py` after recording a comparable result.

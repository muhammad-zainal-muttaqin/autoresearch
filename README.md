# autoresearch

![teaser](progress.png)

This repository applies the "autoresearch" idea to a practical computer vision task: autonomous tuning of a YOLO detector for oil palm fruit bunch images. Instead of editing a large codebase, the workflow is intentionally narrow: `prepare.py` defines the fixed dataset and evaluation harness, while `train.py` is the main experiment surface.

The current project trains and evaluates a detector on 4 classes: `B1`, `B2`, `B3`, and `B4`.

## What It Does

- Verifies a local YOLO-format dataset before training.
- Runs a single-GPU Ultralytics YOLO experiment.
- Evaluates the best checkpoint on the validation split.
- Prints stable, machine-parseable metrics for humans or agents to compare runs.
- Supports an autonomous experiment loop driven by `program.md`.

The primary optimization target is `val_map50_95`. Higher is better.

## Project Layout

```text
train.py                editable experiment surface
prepare.py              dataset constants, verification, evaluation
program.md              agent instructions and experiment protocol
pyproject.toml          Python dependencies
Dataset-YOLO/           canonical local split (gitignored)
  data.yaml             dataset config
  images/{train,val,test}
  labels/{train,val,test}
```

Datasets, raw logs, and generated training outputs under `runs/` are local-only artifacts and should not be committed. The exceptions are `results.tsv` and `progress.png`, which act as durable experiment telemetry.

## Editing and Artifact Policy

- `train.py` is the only file the autoresearch agent should change during normal experimentation.
- `prepare.py` is read-only unless a confirmed runtime bug blocks execution.
- `results.tsv`, `baseline.log`, and `followup.log` are local artifacts and must stay uncommitted.
- `progress.png` must be regenerated from the current local `results.tsv` whenever a new result is recorded, and should be committed when publishing branch progress.

## Requirements

- Python 3.10+
- `uv`
- A CUDA-capable NVIDIA GPU

The project currently targets single-GPU training through `torch` and `ultralytics`.

## Quick Start

```bash
# Install dependencies
uv sync

# Verify dataset layout
uv run prepare.py

# Run one experiment
uv run train.py
```

Default paths are resolved from the repository root, not from the shell's current working directory. If the dataset is stored outside the repository, point the code at it with `YOLO_DATASET_DIR`; relative values are also interpreted from the repository root.

PowerShell example:

```powershell
$env:YOLO_DATASET_DIR = "D:\datasets\Dataset-YOLO"
uv run prepare.py
uv run train.py
```

The canonical dataset split used by this repository currently contains `2,764` train images, `604` validation images, and `624` test images.

## Files You Should and Should Not Edit

`train.py` is the file intended for iteration. It contains the model selection, time budget, optimizer settings, batch size, image size, augmentation settings, and loss weights.

`prepare.py` should be treated as read-only during normal experimentation. It defines:

- dataset location and class names
- dataset verification
- validation-time metric collection

`program.md` describes how an autonomous agent should set up branches, log results, and decide whether to keep or discard a change.

## Training and Metrics

By default, a run uses `TIME_HOURS = 0.5`, which means a 30-minute training budget. After training, `train.py` evaluates `runs/autoresearch/train/weights/best.pt` and prints:

- `val_map50`
- `val_map50_95`
- `precision`
- `recall`
- `peak_vram_mb`
- per-class metrics such as `map50_B1` and `map50_95_B4`

These printed names are intentionally stable so logs can be parsed automatically.

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

If you mirror the dataset from another source, treat it as equivalent to `Dataset-YOLO` only when the split membership matches the canonical local split, not just the class names or total image count.

## Autonomous Experimentation

The repository is designed to support repeated experiments where an agent:

1. reads `program.md`, `prepare.py`, and `train.py`
2. establishes a baseline run
3. edits only the experiment constants in `train.py`
4. reruns training
5. logs the resulting metrics to versioned `results.tsv` and refreshes `progress.png`

That makes the codebase small enough for fast iteration while keeping evaluation consistent across runs.

## Analysis

Use `analysis.ipynb` to inspect `results.tsv` after several experiments. The notebook is aligned with the current YOLO schema and tracks `val_map50_95` as the primary frontier metric, with higher values treated as better. It also regenerates `progress.png`, which is embedded at the top of this README. In the default workflow, both files are committed so progress survives pod restarts or eviction.

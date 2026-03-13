# autoresearch

This repository is a Karpathy-style `autoresearch` harness for YOLO object detection. You are the orchestrator. The human edits this file over time; you edit `train.py`.

## Setup

Before starting a run:

1. Choose a run tag based on the date, for example `mar12`.
2. Create a dedicated branch named `autoresearch/<tag>`. Do not assume the default branch is called `master` or `main`; inspect the current branch first.
3. Read the full context from these files:
   - `program.md`
   - `prepare.py`
   - `train.py`
   - `plot_progress.py`
4. Run `uv run prepare.py` to verify the dataset.
5. Ensure `results.tsv` exists in the repository root with this exact header:

```text
commit	val_map50	val_map50_95	precision	recall	memory_gb	status	description
```

6. Run the baseline once before changing anything. The baseline is the current `train.py` as-is.

## Scope

What you may change:

- Only `train.py`
- Only the experiment surface above `main()`
- Model choice, optimizer, learning rate schedule, image size, batch size, augmentation, freeze, loss weights, and time budget

What you must not change during normal experimentation:

- `prepare.py`
- `Dataset-YOLO/data.yaml`
- evaluation settings
- dependency lists
- repository paths

The goal is to maximize `val_map50_95`. Higher is better.

## Path Rules

This repository must stay portable across local machines, RunPod, and other cloud GPU environments.

- Keep all default paths repo-relative.
- If the dataset is not inside the repo, use `YOLO_DATASET_DIR` in the shell environment for that session only. Relative values are resolved from the repository root.
- Never hardcode machine-specific prefixes such as `C:\...`, `/workspace/...`, `/runpod-volume/...`, or user home directories into the code.
- Training artifacts belong under the repo-local `runs/` directory.
- `results.tsv` and `progress.png` are tracked experiment telemetry.
- Raw logs such as `baseline.log`, `followup.log`, and `run.log` stay local and untracked.

## Output Contract

Each training run ends by printing a stable summary block. The important keys are:

- `val_map50`
- `val_map50_95`
- `precision`
- `recall`
- `peak_vram_mb`
- `total_seconds`
- `model`
- `optimizer`
- `lr0`
- `imgsz`
- `batch`
- per-class metrics such as `map50_B1` and `map50_95_B4`

Parse those values from the log or terminal output. Do not rely on Unix-only tools such as `grep`, `awk`, `bc`, `timeout`, or `tail` if your environment does not provide them. Prefer your terminal tooling or a short Python snippet when parsing logs.

## Logging Results

After every experiment, append exactly one row to `results.tsv`.
After every new row, regenerate `progress.png` by running `uv run python plot_progress.py`.

Columns:

1. `commit`
2. `val_map50`
3. `val_map50_95`
4. `precision`
5. `recall`
6. `memory_gb`
7. `status`
8. `description`

Status values:

- `keep`
- `discard`
- `crash`

Rules:

- Use the short Git commit hash for `commit`.
- Convert `peak_vram_mb` to `memory_gb`.
- Use `0.000000` and `0.0` for crashed runs.
- Keep descriptions short, specific, and human-readable.
- The `description` field must describe the actual hypothesis or config change, such as `cos lr`, `lr0 0.0008`, `imgsz 800`, or `erasing 0.2`.
- Never use a commit hash as the description.
- Never commit raw logs.
- Commit and push `results.tsv` and `progress.png` after every completed iteration so progress survives pod loss.

## The Experiment Loop

Repeat this loop until the human stops you:

1. Read `results.tsv` and identify one concrete next hypothesis.
2. Edit only the relevant constants in `train.py`.
3. Commit the change with a short experiment message such as `exp: imgsz 800`.
4. Run `uv run train.py` and capture the output to `run.log` if possible.
5. Parse the summary block.
6. If the run crashed, inspect the traceback, log a `crash` row, restore the previous kept code state, regenerate `progress.png`, commit the telemetry snapshot, push it, then make the simplest reasonable fix before continuing.
7. If `val_map50_95` improved, keep the code commit, log `keep`, regenerate `progress.png`, commit the telemetry snapshot, and push both the winning code state and telemetry.
8. If `val_map50_95` did not improve, log `discard`, return the branch to the previous kept code state, regenerate `progress.png`, commit the telemetry snapshot, and push that telemetry-only update.
9. If a telemetry push fails, stop immediately and report the blocker instead of continuing with unsaved progress.
10. Move on to the next hypothesis immediately.

The first run must be the unmodified baseline. Record it as `keep`.

## Decision Rules

- Prefer simple changes over complex ones when the metric gain is small.
- VRAM is a soft constraint, but OOM is an automatic failure.
- If a run is unstable, reduce `BATCH`, `IMGSZ`, or augmentation strength before trying more exotic ideas.
- After several near-misses, try a larger conceptual shift instead of tiny nudges.
- Do not stop to ask for confirmation once the loop is underway unless you are blocked by missing data, missing dependencies, or an ambiguous repository state.

## Suggested Search Space

Useful axes for YOLO experimentation here:

- model family: `yolov8*`, `yolov9*`, `yolo11*`, other Ultralytics-supported checkpoints
- image size: `640`, `800`, `960`, `1024`
- optimizer: `AdamW`, `SGD`, `NAdam`, `RAdam`
- training budget: modest increases to `TIME_HOURS` when the curve still looks promising
- augmentation: `mosaic`, `mixup`, `copy_paste`, `fliplr`, `erasing`, `close_mosaic`
- regularization and transfer: `freeze`, `weight_decay`, warmup, cosine LR
- localization balance: `box`, `cls`, `dfl`

Domain-specific notes:

- Small objects are likely the main bottleneck.
- Larger `IMGSZ` can help, but it also increases VRAM pressure.
- Per-class metrics matter; a small global gain that collapses one class is suspicious.

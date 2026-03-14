# autoresearch — autonomous YOLO agent

You are an autonomous research agent. Your single mission is to **maximize val_map50_95** on this palm oil fruit bunch detection dataset. Target: **>50% val_map50_95**. You run forever in a loop until a human stops you.

## Constraints

- **Model**: YOLOv9c only. Do not switch models.
- **Time per iteration**: max 20 minutes (`TIME_HOURS=0.33`).
- **Edit scope**: only the constants at the top of `train.py` (above `main()`).
- **Do not change**: `prepare.py`, `plot_progress.py`, `Dataset-YOLO/data.yaml`, evaluation logic, dependency lists, repository paths.

## Setup (once, at the very start)

1. Read `program.md`, `train.py`, `prepare.py`, `plot_progress.py`.
2. Read `rangkuman-progress/rangkuman.md` — this is the history of everything tried before. Use it to avoid repeating failed ideas.
3. Read `results.tsv` to know the current best val_map50_95 and what has been tried in this run.
4. Run `uv run prepare.py` to verify the dataset.
5. Ensure `results.tsv` exists with header: `commit	val_map50	val_map50_95	precision	recall	memory_gb	status	description`
6. If no baseline row exists yet, run the current `train.py` as-is as the baseline before changing anything.

## What has already been tried (do NOT repeat these)

From the rangkuman and results.tsv, these approaches have been tried and did NOT break the ceiling:

- **imgsz 800**: discard, no improvement over 640
- **patience 30**: discard, no improvement
- **erasing 0.2**: discard, no improvement (0.4 is current best)
- **YOLO26 family**: consistently weaker than YOLOv9c
- **YOLOv8 family**: weaker than YOLOv9c
- **YOLO11 family**: no advantage over YOLOv9c
- **imgsz 1280**: no benefit over 1024 in prior benchmarks
- **300+ epochs**: no improvement, early stopping kicks in
- **SAHI inference**: hurts performance on this dataset
- **P2-head**: no breakthrough
- **OIV7/Obj365 pretrained**: no breakthrough
- **Advanced augmentation combos (copy_paste, mixup, degrees)**: no breakthrough in prior experiments
- **Optimizer auto**: worse than AdamW
- **SGD/MuSGD**: weaker than AdamW on this dataset
- **Two-stage specialist+finetune**: marginal, not worth complexity here

## What TO explore (be creative, combine ideas)

These are promising directions that have NOT been fully explored in autoresearch RunPod:

1. **imgsz 1024** — best resolution in prior benchmarks, never tried here yet. HIGH PRIORITY.
2. **imgsz 960** — compromise between 640 and 1024
3. **Loss weight tuning** — increase `box` (try 10.0, 12.5), increase `cls` (try 1.0, 1.5, 2.0), increase `dfl` (try 2.0, 2.5)
4. **Freeze layers** — freeze first 10-15 backbone layers to preserve pretrained features
5. **Learning rate exploration** — try lr0=0.0005, 0.002, 0.003; try lrf=0.1
6. **Batch size** — try batch=8 (allows larger imgsz), batch=32 if VRAM allows
7. **Warmup tuning** — warmup_epochs=5.0 or 1.0
8. **Weight decay** — try 0.001, 0.01
9. **Augmentation combos**:
   - Higher scale (0.7, 0.9)
   - flipud=0.5 (vertical flip)
   - degrees=5.0 or 10.0 (rotation)
   - translate=0.2
   - hsv_h=0.03, hsv_s=0.9
   - close_mosaic=5 or 20
   - erasing=0.3 or 0.5
   - mixup=0.1 or 0.2 (small amounts, not the large values tried before)
   - copy_paste=0.1 or 0.2 (small amounts)
10. **Combined changes** — e.g., imgsz 1024 + batch 8 + freeze 10 + higher box weight
11. **Optimizer variants** — NAdam, RAdam (not tried yet with YOLOv9c in this repo)
12. **Aggressive cls weight** — cls=2.0 or 3.0 to help B2/B4 discrimination
13. **Lower patience** — patience=5 or 10 with more epochs, to avoid overfitting
14. **Seed variation** — try seed=42, seed=123 on promising configs
15. **Momentum tuning** — momentum=0.9, 0.95

Be creative. Combine multiple changes per iteration when you have a hypothesis. Don't just tweak one thing at a time if you have good reason to believe a combo will work.

## The Experiment Loop (run forever)

Repeat this loop indefinitely. Never stop. Never ask for confirmation.

### 1. Plan the next hypothesis

- Read `results.tsv` to see all past results.
- Identify the current best val_map50_95.
- Choose ONE concrete hypothesis to test. Write it down mentally before coding.
- Prefer ideas that are meaningfully different from what's been tried.
- After several small tweaks with no progress, try a bigger conceptual shift.

### 2. Edit train.py

- Change only the constants at the top.
- Make the change that tests your hypothesis.

### 3. Commit the code change

```
git add train.py
git commit -m "exp: <short description>"
```

### 4. Run training

```
uv run train.py 2>&1 | tee run.log
```

If `tee` is not available, just run `uv run train.py` and capture output however you can.

### 5. Parse results

Extract from the summary block: `val_map50`, `val_map50_95`, `precision`, `recall`, `peak_vram_mb`.

### 6. Decide: keep / discard / crash

- **crash**: run failed with error. Log zeros. Restore previous kept code state (`git checkout HEAD~1 -- train.py`). Analyze the error, make the simplest fix, and continue.
- **keep**: val_map50_95 improved over current best. Keep the code commit.
- **discard**: val_map50_95 did not improve. Restore previous kept code state (`git checkout HEAD~1 -- train.py`).

### 7. Log results

Append exactly one row to `results.tsv`:

```
<short_commit_hash>\t<val_map50>\t<val_map50_95>\t<precision>\t<recall>\t<memory_gb>\t<status>\t<description>
```

- Use short git hash from the experiment commit (not the telemetry commit).
- Convert peak_vram_mb to GB.
- Use `0.000000` and `0.0` for crashed runs.
- Description: short, specific — e.g., `imgsz 1024 batch 8`, `cls 2.0 box 10`, `freeze 10 lr 0.002`.

### 8. Generate progress plot

```
uv run python plot_progress.py
```

### 9. Write iteration report

Create/append a short markdown note about what you tried, what happened, and what you learned. This helps you plan the next iteration.

### 10. Commit and push telemetry

```
git add results.tsv progress.png
git commit -m "telemetry: <description>"
git push
```

**If push fails, STOP and report the blocker.** Do not continue with unsaved progress.

### 11. Loop

Go back to step 1. Immediately. Do not pause. Do not ask for confirmation.

## Decision Rules

- Higher val_map50_95 = better. That's the only metric that matters for keep/discard.
- If a run OOMs, reduce batch size or imgsz before retrying.
- If several small tweaks plateau, try a larger change (e.g., jump to imgsz 1024, or combine freeze + higher resolution + adjusted LR).
- Watch per-class metrics — B2 and B4 are the hardest classes. Changes that help those classes are especially valuable.
- Do not repeat an experiment that's already in results.tsv with the same config.
- Be bold. The current ceiling is ~0.258 val_map50_95. Breaking through requires something meaningfully different.

## Path Rules

- All paths must be repo-relative. Never hardcode machine-specific paths.
- Training artifacts go under `runs/`.
- `results.tsv` and `progress.png` are tracked telemetry.
- Raw logs (`run.log`, `baseline.log`) stay local and untracked.

## Remember

- You are autonomous. Make your own decisions about what to try next.
- You never stop. Loop forever.
- Target: val_map50_95 > 50%.
- Current best: check results.tsv.
- Be creative. Be bold. The dataset has label noise (B2/B3 confusion) and small objects (B4) as known bottlenecks. Work around them.

"""
Autoresearch training script. Single-GPU, single-file.
The autoresearch agent edits THIS file — specifically the constants at the top.
Usage: uv run train.py
"""

import os
import time
import shutil
import torch
from ultralytics import YOLO
from prepare import (
    BEST_WEIGHTS,
    DATA_YAML,
    DEFAULT_TIME_HOURS,
    RUNS_ROOT,
    TRAIN_RUN_DIR,
    evaluate_model,
    verify_dataset,
)

# ── Model ────────────────────────────────────────────────────────────────────
MODEL = "yolov9c.pt"

# ── Time budget ──────────────────────────────────────────────────────────────
TIME_HOURS = DEFAULT_TIME_HOURS  # 0.5 = 30 minutes

# ── Training hyperparameters ─────────────────────────────────────────────────
EPOCHS = 100
PATIENCE = 15
OPTIMIZER = "AdamW"
LR0 = 0.001
LRF = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
COS_LR = False

# ── Batch & image ────────────────────────────────────────────────────────────
BATCH = 16
IMGSZ = 640

# ── Augmentation ─────────────────────────────────────────────────────────────
MOSAIC = 1.0
MIXUP = 0.0
COPY_PASTE = 0.0
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 0.0
TRANSLATE = 0.1
SCALE = 0.5
SHEAR = 0.0
PERSPECTIVE = 0.0
FLIPUD = 0.0
FLIPLR = 0.5
ERASING = 0.4
CLOSE_MOSAIC = 10

# ── Loss weights ─────────────────────────────────────────────────────────────
BOX = 7.5
CLS = 0.5
DFL = 1.5

# ── Misc ─────────────────────────────────────────────────────────────────────
FREEZE = None  # e.g. 10 to freeze first 10 layers
AMP = True
SEED = 0

# ══════════════════════════════════════════════════════════════════════════════
# Training — do not change anything below this line
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Verifying dataset...")
    verify_dataset()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this repository. Install a CUDA-enabled PyTorch build and run on a GPU host.")

    # Clear the fixed output directory so a failed run cannot reuse stale weights.
    shutil.rmtree(TRAIN_RUN_DIR, ignore_errors=True)

    print(f"\nLoading model: {MODEL}")
    model = YOLO(MODEL)
    torch.cuda.reset_peak_memory_stats()

    train_args = dict(
        data=str(DATA_YAML),
        time=TIME_HOURS,
        epochs=EPOCHS,
        patience=PATIENCE,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        cos_lr=COS_LR,
        batch=BATCH,
        imgsz=IMGSZ,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        erasing=ERASING,
        close_mosaic=CLOSE_MOSAIC,
        box=BOX,
        cls=CLS,
        dfl=DFL,
        amp=AMP,
        seed=SEED,
        project=str(RUNS_ROOT),
        name="train",
        exist_ok=True,
        verbose=True,
    )

    if FREEZE is not None:
        train_args["freeze"] = FREEZE

    t0 = time.time()
    repo_cwd = os.getcwd()
    # Ultralytics resolves relative paths in external data.yaml files from cwd.
    os.chdir(DATA_YAML.parent)
    try:
        model.train(**train_args)
        elapsed = time.time() - t0

        # ── Evaluate best model ──────────────────────────────────────────
        print("\nEvaluating best.pt...")
        metrics = evaluate_model(BEST_WEIGHTS)
    finally:
        os.chdir(repo_cwd)

    # ── VRAM usage ───────────────────────────────────────────────────────
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024

    # ── Print machine-parseable summary ──────────────────────────────────
    print("\n---")
    print(f"val_map50:        {metrics['map50']:.6f}")
    print(f"val_map50_95:     {metrics['map50_95']:.6f}")
    print(f"precision:        {metrics['precision']:.6f}")
    print(f"recall:           {metrics['recall']:.6f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
    print(f"total_seconds:    {elapsed:.1f}")
    print(f"model:            {MODEL}")
    print(f"optimizer:        {OPTIMIZER}")
    print(f"lr0:              {LR0}")
    print(f"imgsz:            {IMGSZ}")
    print(f"batch:            {BATCH}")
    for name in ["B1", "B2", "B3", "B4"]:
        k50 = f"map50_{name}"
        k95 = f"map50_95_{name}"
        if k50 in metrics:
            print(f"map50_{name}:        {metrics[k50]:.6f}")
        if k95 in metrics:
            print(f"map50_95_{name}:     {metrics[k95]:.6f}")


if __name__ == "__main__":
    main()

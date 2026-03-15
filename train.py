"""
Editable experiment surface for autoresearch.

Normal mode:
- edit this file
- optionally edit research.py for nontrivial model/pipeline work
- run `uv run train.py`
"""

from prepare import DATA_YAML, DEFAULT_TIME_HOURS, RUNS_ROOT

EXPERIMENT_TITLE = "baseline probe"
HYPOTHESIS = "If I run the small-model probe baseline, I will get a comparable signal for the next decision."
SUCCESS_CRITERION = "Match or beat the current comparable main baseline on val_map50_95."
TRACK_HINT = "auto"  # one of: auto, main, exploration
EXPLORATION_NAME = ""

MODEL = "yolo11s.pt"

TIME_HOURS = DEFAULT_TIME_HOURS
EPOCHS = 40
PATIENCE = 15
SEED = 42

OPTIMIZER = "AdamW"
LR0 = 0.001
LRF = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
COS_LR = True

BATCH = 16
IMGSZ = 640

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

BOX = 7.5
CLS = 0.5
DFL = 1.5
LABEL_SMOOTHING = 0.0

FREEZE = None
AMP = True


def build_experiment_spec() -> dict:
    """Return the frozen orchestrator spec for this experiment."""
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
        label_smoothing=LABEL_SMOOTHING,
        amp=AMP,
        seed=SEED,
        project=str(RUNS_ROOT),
        name="train",
        exist_ok=True,
        verbose=True,
    )
    if FREEZE is not None:
        train_args["freeze"] = FREEZE

    return {
        "title": EXPERIMENT_TITLE,
        "hypothesis": HYPOTHESIS,
        "success_criterion": SUCCESS_CRITERION,
        "track_hint": TRACK_HINT,
        "exploration_name": EXPLORATION_NAME,
        "seed": SEED,
        "model_ref": MODEL,
        "train_args": train_args,
        "imgsz": IMGSZ,
        "research_module": "research",
    }


def main() -> None:
    from orchestrator import run_from_train_spec

    run_from_train_spec(build_experiment_spec())


if __name__ == "__main__":
    main()

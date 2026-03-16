"""
Dataset verification and evaluation utilities (READ-ONLY).

This file should NOT be edited by the autoresearch agent during normal operation.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _resolve_path(value: str | None, default: Path, *, base: Path) -> Path:
    """Resolve an env-provided path without hardcoding any machine-local prefix."""
    if value is None or not value.strip():
        return default.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


DATASET_DIR = _resolve_path(
    os.environ.get("YOLO_DATASET_DIR"),
    REPO_ROOT / "Dataset-YOLO",
    base=REPO_ROOT,
)
DATA_YAML = DATASET_DIR / "data.yaml"
RUNS_ROOT = REPO_ROOT / "runs" / "autoresearch"
TRAIN_RUN_DIR = RUNS_ROOT / "train"
BEST_WEIGHTS = TRAIN_RUN_DIR / "weights" / "best.pt"
RESULTS_TSV = REPO_ROOT / "experiments" / "results.tsv"
NUM_CLASSES = 4
CLASS_NAMES = ["B1", "B2", "B3", "B4"]
DEFAULT_TIME_HOURS = 0.5


def verify_dataset() -> None:
    """Check that images and labels exist for each split. Raises on failure."""
    assert DATA_YAML.exists(), f"data.yaml not found at {DATA_YAML}"
    for split in ("train", "val", "test"):
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        assert img_dir.exists(), f"Missing {img_dir}"
        assert lbl_dir.exists(), f"Missing {lbl_dir}"
        n_img = len(list(img_dir.iterdir()))
        n_lbl = len(list(lbl_dir.iterdir()))
        print(f"  {split:5s}: {n_img} images, {n_lbl} labels")
    print("Dataset OK")


def _compute_b23_confusion(confusion_matrix) -> float | None:
    matrix = getattr(confusion_matrix, "matrix", None)
    if matrix is None or matrix.shape[0] < 4 or matrix.shape[1] < 4:
        return None

    # ConfusionMatrix uses rows=predicted, cols=actual, plus background class.
    b2_actual_total = float(matrix[:, 1].sum())
    b3_actual_total = float(matrix[:, 2].sum())
    denom = b2_actual_total + b3_actual_total
    if denom <= 0:
        return None

    confused = float(matrix[2, 1] + matrix[1, 2])
    return confused / denom


def evaluate_model(model_path: str | Path) -> dict:
    """Run validation on a trained model and return metrics dict."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    validator_cls = model._smart_load("validator")
    validator = validator_cls(
        args={
            **model.overrides,
            "rect": True,
            "data": str(DATA_YAML),
            "imgsz": 640,
            "conf": 0.001,
            "iou": 0.6,
            "split": "val",
            "mode": "val",
            "verbose": False,
            "plots": False,
        },
        _callbacks=model.callbacks,
    )
    validator(model=model.model)
    metrics_obj = validator.metrics

    metrics = {
        "map50": float(metrics_obj.box.map50),
        "map50_95": float(metrics_obj.box.map),
        "precision": float(metrics_obj.box.mp),
        "recall": float(metrics_obj.box.mr),
        "b2_b3_confusion": _compute_b23_confusion(getattr(validator, "confusion_matrix", None)),
    }

    for i, name in enumerate(CLASS_NAMES):
        if i >= len(metrics_obj.box.ap50) or i >= len(metrics_obj.box.ap):
            continue
        p_i, r_i, ap50_i, ap_i = metrics_obj.class_result(i)
        metrics[f"precision_{name}"] = float(p_i)
        metrics[f"recall_{name}"] = float(r_i)
        metrics[f"map50_{name}"] = float(ap50_i)
        metrics[f"map50_95_{name}"] = float(ap_i)

    metrics["b4_precision"] = metrics.get("precision_B4")
    metrics["b4_recall"] = metrics.get("recall_B4")
    return metrics


if __name__ == "__main__":
    print(f"Dataset dir: {DATASET_DIR.resolve()}")
    verify_dataset()

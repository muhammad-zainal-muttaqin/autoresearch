"""
Dataset verification and evaluation utilities (READ-ONLY).

This file should NOT be edited by the autoresearch agent.
Usage: uv run prepare.py
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


# ── Constants ────────────────────────────────────────────────────────────────
DATASET_DIR = _resolve_path(
    os.environ.get("YOLO_DATASET_DIR"),
    REPO_ROOT / "Dataset-YOLO",
    base=REPO_ROOT,
)
DATA_YAML = DATASET_DIR / "data.yaml"
RUNS_ROOT = REPO_ROOT / "runs" / "autoresearch"
TRAIN_RUN_DIR = RUNS_ROOT / "train"
BEST_WEIGHTS = TRAIN_RUN_DIR / "weights" / "best.pt"
RESULTS_TSV = REPO_ROOT / "results.tsv"
NUM_CLASSES = 4
CLASS_NAMES = ["B1", "B2", "B3", "B4"]
DEFAULT_TIME_HOURS = 0.5  # 30 minutes


def verify_dataset():
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


def evaluate_model(model_path: str | Path) -> dict:
    """Run validation on a trained model and return metrics dict."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.val(
        data=str(DATA_YAML),
        imgsz=640,
        conf=0.001,
        iou=0.6,
        split="val",
        verbose=False,
    )
    metrics = {
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }
    # Per-class breakdown
    for i, name in enumerate(CLASS_NAMES):
        if i < len(results.box.ap50):
            metrics[f"map50_{name}"] = float(results.box.ap50[i])
        if i < len(results.box.ap):
            metrics[f"map50_95_{name}"] = float(results.box.ap[i])
    return metrics


if __name__ == "__main__":
    print(f"Dataset dir: {DATASET_DIR.resolve()}")
    verify_dataset()

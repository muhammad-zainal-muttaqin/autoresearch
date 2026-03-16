"""
E0 Baseline Experimental Protocol — Automated Runner

Runs the full E0 protocol for BBC oil palm fruit bunch detection:
  Phase 0A: EDA (class distribution, bbox sizes, M4 dimensions)
  Phase 0B: Resolution sweep (640 vs 1024)
  Phase 0C: Data learning curve (25/50/75/100%)
  Phase 1B: Architecture sweep (small models only)
  Phase 2:  Hyperparameter optimization (LR, batch, augmentation)
  Phase 3:  Final validation (confusion matrix, threshold sweep, TFLite export)

Usage:
    uv run e0_protocol.py              # run from scratch
    uv run e0_protocol.py --resume     # resume from last completed phase
    uv run e0_protocol.py --status     # show progress
    uv run e0_protocol.py --phase 1b   # run specific phase
"""

from __future__ import annotations

import csv
import gc
import json
import os
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "Dataset-YOLO"
DATA_YAML = DATASET_DIR / "data.yaml"

E0_DIR = REPO_ROOT / "e0_results"
RUNS_DIR = E0_DIR / "runs"
PLOTS_DIR = E0_DIR / "plots"
REPORTS_DIR = E0_DIR / "reports"
SUBSETS_DIR = E0_DIR / "subsets"
STATE_PATH = E0_DIR / "state.json"
RESULTS_CSV = E0_DIR / "results.csv"

CLASS_NAMES = ["B1", "B2", "B3", "B4"]
NUM_CLASSES = 4
MAX_EPOCHS = 40
PATIENCE = 15
SEEDS = [0, 42]
BASELINE_MODEL = "yolo11s.pt"

SMALL_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov10n.pt",
    "yolov10s.pt",
    "yolo11n.pt",
    "yolo11s.pt",
]

RESOLUTIONS = [640, 1024]
DATA_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
LR_VALUES = [0.0005, 0.001, 0.002]
BATCH_VALUES = [8, 16]
AUG_PRESETS = {
    "light": dict(hsv_h=0.008, hsv_s=0.35, hsv_v=0.2, degrees=5.0, mosaic=0.5, mixup=0.0, scale=0.3, fliplr=0.5),
    "medium": dict(hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10.0, mosaic=1.0, mixup=0.0, scale=0.5, fliplr=0.5),
    "heavy": dict(hsv_h=0.02, hsv_s=0.9, hsv_v=0.5, degrees=15.0, mosaic=1.0, mixup=0.15, scale=0.7, fliplr=0.5),
}
CONF_THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

RESULTS_COLUMNS = [
    "run_id", "phase", "model", "imgsz", "seed", "batch", "lr0", "aug",
    "data_fraction", "epochs_completed",
    "map50", "map50_95", "map75", "precision", "recall",
    "map50_B1", "map50_B2", "map50_B3", "map50_B4",
    "map50_95_B1", "map50_95_B2", "map50_95_B3", "map50_95_B4",
    "b2_b3_confusion", "b3_b4_confusion",
    "b4_precision", "b4_recall",
    "time_minutes", "vram_gb", "status",
]


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"completed_phases": [], "locked": {}, "top_archs": [], "best_configs": {}}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _mark_phase_complete(phase: str, state: dict) -> None:
    if phase not in state["completed_phases"]:
        state["completed_phases"].append(phase)
    _save_state(state)


def _is_phase_done(phase: str, state: dict) -> bool:
    return phase in state["completed_phases"]


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------
def _ensure_results_csv() -> None:
    if not RESULTS_CSV.exists():
        with RESULTS_CSV.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, delimiter=",").writeheader()


def _append_result(row: dict) -> None:
    _ensure_results_csv()
    normalized = {col: row.get(col, "") for col in RESULTS_COLUMNS}
    with RESULTS_CSV.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, delimiter=",").writerow(normalized)


def _run_exists(run_id: str) -> bool:
    if not RESULTS_CSV.exists():
        return False
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter=","):
            if row.get("run_id") == run_id and row.get("status") == "ok":
                return True
    return False


def _read_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        return pd.DataFrame(columns=RESULTS_COLUMNS)
    return pd.read_csv(RESULTS_CSV)


# ---------------------------------------------------------------------------
# Enhanced evaluation
# ---------------------------------------------------------------------------
def evaluate_extended(model_path: str | Path, data_yaml: str | Path, imgsz: int = 640,
                      conf: float = 0.001) -> dict:
    """Evaluate model with extended metrics: mAP@0.75, M3/M4 confusion, full confusion matrix."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    validator_cls = model._smart_load("validator")
    validator = validator_cls(
        args={
            **model.overrides,
            "rect": True,
            "data": str(data_yaml),
            "imgsz": imgsz,
            "conf": conf,
            "iou": 0.6,
            "split": "val",
            "mode": "val",
            "verbose": False,
            "plots": False,
        },
        _callbacks=model.callbacks,
    )
    validator(model=model.model)
    mo = validator.metrics

    metrics: dict[str, Any] = {
        "map50": float(mo.box.map50),
        "map50_95": float(mo.box.map),
        "precision": float(mo.box.mp),
        "recall": float(mo.box.mr),
    }

    # mAP@0.75 — index 5 in the 10-step IoU range [0.50..0.95]
    try:
        all_ap = mo.box.all_ap  # shape (num_classes, 10)
        if all_ap is not None and all_ap.shape[1] > 5:
            metrics["map75"] = float(all_ap[:, 5].mean())
        else:
            metrics["map75"] = None
    except Exception:
        metrics["map75"] = None

    # Per-class metrics
    for i, name in enumerate(CLASS_NAMES):
        try:
            p_i, r_i, ap50_i, ap_i = mo.class_result(i)
            metrics[f"precision_{name}"] = float(p_i)
            metrics[f"recall_{name}"] = float(r_i)
            metrics[f"map50_{name}"] = float(ap50_i)
            metrics[f"map50_95_{name}"] = float(ap_i)
        except Exception:
            pass

    metrics["b4_precision"] = metrics.get("precision_B4")
    metrics["b4_recall"] = metrics.get("recall_B4")

    # Confusion matrix based metrics
    cm = getattr(validator, "confusion_matrix", None)
    matrix = getattr(cm, "matrix", None) if cm else None
    if matrix is not None and matrix.shape[0] >= 4 and matrix.shape[1] >= 4:
        # B2/B3 confusion (rows=predicted, cols=actual)
        b2_total = float(matrix[:, 1].sum())
        b3_total = float(matrix[:, 2].sum())
        denom_23 = b2_total + b3_total
        if denom_23 > 0:
            metrics["b2_b3_confusion"] = float(matrix[2, 1] + matrix[1, 2]) / denom_23
        # B3/B4 confusion
        b4_total = float(matrix[:, 3].sum())
        denom_34 = b3_total + b4_total
        if denom_34 > 0:
            metrics["b3_b4_confusion"] = float(matrix[3, 2] + matrix[2, 3]) / denom_34
        # Save raw matrix for visualization
        metrics["_confusion_matrix"] = matrix.copy()

    return metrics


def _cleanup_gpu():
    """Free GPU memory between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------
def run_training(
    model_name: str,
    imgsz: int,
    run_id: str,
    phase: str,
    data_yaml: str | Path = DATA_YAML,
    seed: int = 42,
    batch: int = 16,
    lr0: float = 0.001,
    epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    aug_preset: str = "medium",
    data_fraction: float = 1.0,
    extra_args: dict | None = None,
) -> dict:
    """Run one YOLO training experiment and return extended metrics."""
    from ultralytics import YOLO

    if _run_exists(run_id):
        print(f"  [SKIP] {run_id} already completed")
        # Return cached result
        df = _read_results()
        row = df[df["run_id"] == run_id].iloc[0].to_dict()
        return row

    print(f"\n{'='*60}")
    print(f"  RUN: {run_id}")
    print(f"  Model: {model_name} | imgsz: {imgsz} | seed: {seed}")
    print(f"  batch: {batch} | lr0: {lr0} | aug: {aug_preset} | frac: {data_fraction}")
    print(f"{'='*60}")

    _cleanup_gpu()

    aug = AUG_PRESETS.get(aug_preset, AUG_PRESETS["medium"])
    run_dir = RUNS_DIR / run_id

    train_args = dict(
        data=str(data_yaml),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        lrf=0.01,
        optimizer="AdamW",
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        seed=seed,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        verbose=False,
        workers=4,
        **aug,
    )
    if extra_args:
        train_args.update(extra_args)

    result_row = {
        "run_id": run_id,
        "phase": phase,
        "model": model_name,
        "imgsz": imgsz,
        "seed": seed,
        "batch": batch,
        "lr0": lr0,
        "aug": aug_preset,
        "data_fraction": data_fraction,
    }

    try:
        model = YOLO(model_name)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        cwd = os.getcwd()
        os.chdir(Path(str(data_yaml)).parent)
        try:
            model.train(**train_args)
        finally:
            os.chdir(cwd)
        elapsed = time.time() - t0

        # Find best weights
        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            best_pt = run_dir / "weights" / "last.pt"

        metrics = evaluate_extended(best_pt, data_yaml, imgsz=imgsz)

        peak_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

        result_row.update({
            "map50": f"{metrics.get('map50', 0):.6f}",
            "map50_95": f"{metrics.get('map50_95', 0):.6f}",
            "map75": f"{metrics.get('map75', 0):.6f}" if metrics.get("map75") else "",
            "precision": f"{metrics.get('precision', 0):.6f}",
            "recall": f"{metrics.get('recall', 0):.6f}",
            "b2_b3_confusion": f"{metrics.get('b2_b3_confusion', 0):.6f}" if metrics.get("b2_b3_confusion") is not None else "",
            "b3_b4_confusion": f"{metrics.get('b3_b4_confusion', 0):.6f}" if metrics.get("b3_b4_confusion") is not None else "",
            "b4_precision": f"{metrics.get('b4_precision', 0):.6f}" if metrics.get("b4_precision") is not None else "",
            "b4_recall": f"{metrics.get('b4_recall', 0):.6f}" if metrics.get("b4_recall") is not None else "",
            "time_minutes": f"{elapsed / 60:.1f}",
            "vram_gb": f"{peak_vram:.1f}",
            "status": "ok",
        })
        for name in CLASS_NAMES:
            result_row[f"map50_{name}"] = f"{metrics.get(f'map50_{name}', 0):.6f}"
            result_row[f"map50_95_{name}"] = f"{metrics.get(f'map50_95_{name}', 0):.6f}"

        # Save confusion matrix if available
        if "_confusion_matrix" in metrics:
            cm_path = PLOTS_DIR / f"cm_{run_id}.npy"
            np.save(str(cm_path), metrics["_confusion_matrix"])

        print(f"  mAP@0.5={metrics.get('map50',0):.4f} | mAP@0.5-0.95={metrics.get('map50_95',0):.4f} | "
              f"B2/B3={metrics.get('b2_b3_confusion','n/a')} | {elapsed/60:.1f}m")

    except Exception:
        print(f"  [FAIL] {run_id}")
        traceback.print_exc()
        result_row["status"] = "fail"

    _append_result(result_row)
    _cleanup_gpu()
    return result_row


# ---------------------------------------------------------------------------
# Phase 0A: EDA
# ---------------------------------------------------------------------------
def phase0a_eda() -> dict:
    """Exploratory data analysis: class distribution, bbox sizes, M4 dimensions."""
    print("\n" + "=" * 60)
    print("PHASE 0A: Exploratory Data Analysis")
    print("=" * 60)

    results: dict[str, Any] = {}

    # Class distribution per split
    split_counts: dict[str, Counter] = {}
    for split in ["train", "val", "test"]:
        lbl_dir = DATASET_DIR / "labels" / split
        counts = Counter()
        if lbl_dir.exists():
            for lbl_file in lbl_dir.glob("*.txt"):
                for line in lbl_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        cls = int(line.split()[0])
                        counts[cls] += 1
        split_counts[split] = counts
        n_images = len(list((DATASET_DIR / "images" / split).glob("*"))) if (DATASET_DIR / "images" / split).exists() else 0
        print(f"  {split}: {n_images} images, {sum(counts.values())} annotations")
        for c in range(NUM_CLASSES):
            print(f"    {CLASS_NAMES[c]}: {counts[c]}")

    results["split_counts"] = {s: dict(c) for s, c in split_counts.items()}

    # Bbox size analysis
    bbox_data = []  # (class, norm_w, norm_h)
    for split in ["train", "val", "test"]:
        lbl_dir = DATASET_DIR / "labels" / split
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            for line in lbl_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, _, _, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    bbox_data.append((cls, w, h))

    bbox_arr = np.array(bbox_data) if bbox_data else np.zeros((0, 3))
    results["bbox_data"] = bbox_arr

    if len(bbox_arr) > 0:
        # At 640px
        for c in range(NUM_CLASSES):
            mask = bbox_arr[:, 0] == c
            if mask.sum() > 0:
                pw = bbox_arr[mask, 1] * 640
                ph = bbox_arr[mask, 2] * 640
                print(f"  {CLASS_NAMES[c]}: median {np.median(pw):.0f}x{np.median(ph):.0f}px, "
                      f"min {pw.min():.0f}x{ph.min():.0f}px, n={mask.sum()}")

        # M4 specific
        m4_mask = bbox_arr[:, 0] == 3
        if m4_mask.sum() > 0:
            m4_w = bbox_arr[m4_mask, 1] * 640
            m4_h = bbox_arr[m4_mask, 2] * 640
            m4_min_dim = min(m4_w.min(), m4_h.min())
            results["m4_min_pixel"] = float(m4_min_dim)
            print(f"\n  M4 (B4) minimum dimension at 640px: {m4_min_dim:.1f}px")
            if m4_min_dim < 16:
                print(f"  WARNING: M4 min dimension < 16px. Standard 640px may struggle.")

    # Generate plots
    _plot_class_distribution(split_counts)
    _plot_bbox_sizes(bbox_arr)

    # Generate report
    _generate_phase0a_report(results)
    return results


def _plot_class_distribution(split_counts: dict[str, Counter]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(NUM_CLASSES)
    width = 0.25
    for i, split in enumerate(["train", "val", "test"]):
        counts = [split_counts[split].get(c, 0) for c in range(NUM_CLASSES)]
        ax.bar(x + i * width, counts, width, label=split)
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Annotation Count")
    ax.set_title("Class Distribution per Split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_class_distribution.png", dpi=150)
    plt.close(fig)


def _plot_bbox_sizes(bbox_arr: np.ndarray) -> None:
    if len(bbox_arr) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#3b82f6", "#f59e0b", "#ef4444", "#10b981"]

    for c in range(NUM_CLASSES):
        mask = bbox_arr[:, 0] == c
        if mask.sum() == 0:
            continue
        pw = bbox_arr[mask, 1] * 640
        ph = bbox_arr[mask, 2] * 640
        axes[0].scatter(pw, ph, s=3, alpha=0.3, c=colors[c], label=CLASS_NAMES[c])
        axes[1].hist(np.minimum(pw, ph), bins=40, alpha=0.5, color=colors[c], label=CLASS_NAMES[c])

    axes[0].set_xlabel("Width (px at 640)")
    axes[0].set_ylabel("Height (px at 640)")
    axes[0].set_title("Bounding Box Sizes")
    axes[0].legend(markerscale=5)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Min Dimension (px at 640)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Min Dimension Distribution")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].axvline(16, color="red", linestyle="--", alpha=0.5, label="16px threshold")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_bbox_sizes.png", dpi=150)
    plt.close(fig)


def _generate_phase0a_report(results: dict) -> None:
    lines = ["# Phase 0A: EDA Report\n"]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("## Class Distribution\n")
    for split, counts in results.get("split_counts", {}).items():
        total = sum(counts.values())
        lines.append(f"**{split}**: {total} annotations")
        for c in range(NUM_CLASSES):
            lines.append(f"  - {CLASS_NAMES[c]}: {counts.get(str(c), counts.get(c, 0))}")
        lines.append("")
    lines.append("![Class Distribution](../plots/eda_class_distribution.png)\n")
    lines.append("## Bounding Box Sizes\n")
    lines.append("![Bbox Sizes](../plots/eda_bbox_sizes.png)\n")
    if "m4_min_pixel" in results:
        lines.append(f"**M4 minimum dimension at 640px**: {results['m4_min_pixel']:.1f}px\n")
    (REPORTS_DIR / "phase0a_eda_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 0B: Resolution Sweep
# ---------------------------------------------------------------------------
def phase0b_resolution_sweep(state: dict) -> int:
    """Test 640 vs 1024. Returns locked resolution."""
    print("\n" + "=" * 60)
    print("PHASE 0B: Resolution Sweep")
    print("=" * 60)

    for imgsz in RESOLUTIONS:
        for seed in SEEDS:
            run_id = f"p0b_res{imgsz}_s{seed}"
            run_training(BASELINE_MODEL, imgsz, run_id, "0B", seed=seed)

    # Analyze
    df = _read_results()
    p0b = df[df["phase"] == "0B"].copy()
    p0b["map50"] = pd.to_numeric(p0b["map50"], errors="coerce")
    p0b["map50_B4"] = pd.to_numeric(p0b["map50_B4"], errors="coerce")

    avg_by_res = p0b.groupby("imgsz")[["map50", "map50_B4"]].mean()
    print("\n  Resolution comparison:")
    print(avg_by_res.to_string())

    m4_640 = avg_by_res.loc[640, "map50_B4"] if 640 in avg_by_res.index else 0
    m4_1024 = avg_by_res.loc[1024, "map50_B4"] if 1024 in avg_by_res.index else 0
    m4_delta = m4_1024 - m4_640

    if m4_delta > 0.05:
        locked = 1024
        reason = f"M4 AP improved {m4_delta:.1%} at 1024px (>5% threshold)"
    elif m4_delta > 0.02:
        locked = 640
        reason = f"M4 AP improved {m4_delta:.1%} at 1024px (marginal, staying 640 for efficiency)"
    else:
        locked = 640
        reason = f"M4 AP delta {m4_delta:.1%} (<2%, staying 640)"

    print(f"\n  Decision: Lock resolution = {locked}px. {reason}")
    state["locked"]["imgsz"] = locked

    # Plot
    _plot_resolution_comparison(p0b)
    _generate_resolution_report(avg_by_res, locked, reason)
    return locked


def _plot_resolution_comparison(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["map50", "map50_B1", "map50_B2", "map50_B3", "map50_B4"]
    labels = ["Overall", "B1", "B2", "B3", "B4"]
    x = np.arange(len(labels))
    width = 0.35

    for i, imgsz in enumerate([640, 1024]):
        sub = df[df["imgsz"] == imgsz]
        vals = [pd.to_numeric(sub[m], errors="coerce").mean() for m in metrics]
        ax.bar(x + i * width, vals, width, label=f"{imgsz}px")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Resolution Comparison: 640 vs 1024")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p0b_resolution_comparison.png", dpi=150)
    plt.close(fig)


def _generate_resolution_report(avg_by_res, locked, reason):
    lines = [
        "# Phase 0B: Resolution Sweep Report\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"Model: {BASELINE_MODEL}, Seeds: {SEEDS}, Epochs: {MAX_EPOCHS}\n",
        "## Results\n",
        avg_by_res.to_markdown() + "\n",
        f"\n## Decision\n",
        f"**Locked resolution: {locked}px**\n",
        f"Reason: {reason}\n",
        "![Resolution Comparison](../plots/p0b_resolution_comparison.png)\n",
    ]
    (REPORTS_DIR / "phase0b_resolution_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 0C: Learning Curve
# ---------------------------------------------------------------------------
def _create_data_subset(fraction: float, seed: int = 42) -> Path:
    """Create a subset of training data at the given fraction, respecting sequence grouping."""
    if fraction >= 1.0:
        return DATA_YAML

    subset_key = f"frac_{int(fraction * 100)}_s{seed}"
    subset_dir = SUBSETS_DIR / subset_key
    subset_yaml = subset_dir / "data.yaml"
    if subset_yaml.exists():
        return subset_yaml

    # Discover sequences in training set
    train_img_dir = DATASET_DIR / "images" / "train"
    train_lbl_dir = DATASET_DIR / "labels" / "train"
    sequences: dict[str, list[str]] = defaultdict(list)

    for img in sorted(train_img_dir.glob("*.jpg")):
        parts = img.stem.split("_")
        seq_key = "_".join(parts[:-1])
        sequences[seq_key].append(img.stem)

    # Compute dominant class per sequence for stratification
    seq_keys = list(sequences.keys())
    seq_strata = []
    for sk in seq_keys:
        counts = Counter()
        for stem in sequences[sk]:
            lbl = train_lbl_dir / f"{stem}.txt"
            if lbl.exists():
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        counts[int(line.split()[0])] += 1
        seq_strata.append(counts.most_common(1)[0][0] if counts else 0)

    # Subsample sequences
    n_keep = max(1, int(len(seq_keys) * fraction))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(seq_keys))[:n_keep]
    kept_seqs = {seq_keys[i] for i in indices}

    # Create subset directories with symlinks (or copies on Windows)
    subset_img_dir = subset_dir / "images" / "train"
    subset_lbl_dir = subset_dir / "labels" / "train"
    subset_img_dir.mkdir(parents=True, exist_ok=True)
    subset_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Symlink val and test from original
    for split in ["val", "test"]:
        for subdir in ["images", "labels"]:
            src = DATASET_DIR / subdir / split
            dst = subset_dir / subdir / split
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.symlink(str(src), str(dst), target_is_directory=True)
                except OSError:
                    shutil.copytree(str(src), str(dst))

    # Copy selected training files
    count = 0
    for seq_key in kept_seqs:
        for stem in sequences[seq_key]:
            src_img = train_img_dir / f"{stem}.jpg"
            src_lbl = train_lbl_dir / f"{stem}.txt"
            dst_img = subset_img_dir / f"{stem}.jpg"
            dst_lbl = subset_lbl_dir / f"{stem}.txt"
            if src_img.exists() and not dst_img.exists():
                try:
                    os.symlink(str(src_img), str(dst_img))
                except OSError:
                    shutil.copy2(str(src_img), str(dst_img))
            if src_lbl.exists() and not dst_lbl.exists():
                try:
                    os.symlink(str(src_lbl), str(dst_lbl))
                except OSError:
                    shutil.copy2(str(src_lbl), str(dst_lbl))
            count += 1

    # Write data.yaml
    yaml_content = f"path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\nnc: {NUM_CLASSES}\nnames:\n"
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"
    subset_yaml.write_text(yaml_content, encoding="utf-8")

    print(f"  Created subset {subset_key}: {count} images ({len(kept_seqs)}/{len(seq_keys)} sequences)")
    return subset_yaml


def phase0c_learning_curve(state: dict) -> None:
    """Train at 25/50/75/100% data fractions, 2 seeds each."""
    print("\n" + "=" * 60)
    print("PHASE 0C: Data Learning Curve")
    print("=" * 60)

    imgsz = state["locked"].get("imgsz", 640)

    for frac in DATA_FRACTIONS:
        data_path = _create_data_subset(frac)
        for seed in SEEDS:
            run_id = f"p0c_frac{int(frac*100)}_s{seed}"
            run_training(BASELINE_MODEL, imgsz, run_id, "0C",
                         data_yaml=data_path, seed=seed, data_fraction=frac)

    # Plot
    df = _read_results()
    p0c = df[df["phase"] == "0C"].copy()
    _plot_learning_curves(p0c)
    _generate_learning_curve_report(p0c)


def _plot_learning_curves(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#3b82f6", "#f59e0b", "#ef4444", "#10b981"]
    df["data_fraction"] = pd.to_numeric(df["data_fraction"], errors="coerce")

    # Overall mAP@0.5
    avg = df.groupby("data_fraction")["map50"].apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
    axes[0].plot(avg.index * 100, avg.values, "o-", color="#3b82f6", linewidth=2, markersize=8)
    axes[0].set_xlabel("Training Data (%)")
    axes[0].set_ylabel("mAP@0.5")
    axes[0].set_title("Overall mAP@0.5 vs Data Fraction")
    axes[0].grid(alpha=0.3)

    # Per-class
    for c, name in enumerate(CLASS_NAMES):
        col = f"map50_{name}"
        if col in df.columns:
            avg_c = df.groupby("data_fraction")[col].apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
            axes[1].plot(avg_c.index * 100, avg_c.values, "o-", color=colors[c], label=name, linewidth=2, markersize=6)

    axes[1].set_xlabel("Training Data (%)")
    axes[1].set_ylabel("mAP@0.5")
    axes[1].set_title("Per-Class mAP@0.5 vs Data Fraction")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p0c_learning_curves.png", dpi=150)
    plt.close(fig)


def _generate_learning_curve_report(df):
    df = df.copy()
    df["data_fraction"] = pd.to_numeric(df["data_fraction"], errors="coerce")
    df["map50"] = pd.to_numeric(df["map50"], errors="coerce")
    avg = df.groupby("data_fraction")["map50"].mean()

    gain_75_100 = avg.get(1.0, 0) - avg.get(0.75, 0)
    if gain_75_100 > 0.02:
        verdict = f"Curve still climbing ({gain_75_100:.1%} gain from 75%->100%). More data would help."
    elif gain_75_100 > 0.01:
        verdict = f"Curve flattening ({gain_75_100:.1%} gain). Data likely sufficient."
    else:
        verdict = f"Curve plateaued ({gain_75_100:.1%} gain). Data is sufficient."

    lines = [
        "# Phase 0C: Learning Curve Report\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "## Results\n",
        avg.to_markdown() + "\n",
        f"\n## Assessment\n",
        f"{verdict}\n",
        "![Learning Curves](../plots/p0c_learning_curves.png)\n",
    ]
    (REPORTS_DIR / "phase0c_learning_curve_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 1B: Architecture Sweep
# ---------------------------------------------------------------------------
def phase1b_architecture_sweep(state: dict) -> list[str]:
    """Sweep small architectures, return top 2."""
    print("\n" + "=" * 60)
    print("PHASE 1B: Architecture Sweep (small models only)")
    print("=" * 60)
    print("  Skipping Phase 1A (two-stage pipeline) — one-stage only for simplicity.\n")

    imgsz = state["locked"].get("imgsz", 640)

    for model_name in SMALL_MODELS:
        for seed in SEEDS:
            run_id = f"p1b_{model_name.replace('.pt','')}_s{seed}"
            run_training(model_name, imgsz, run_id, "1B", seed=seed)

    # Analyze
    df = _read_results()
    p1b = df[df["phase"] == "1B"].copy()
    p1b["map50"] = pd.to_numeric(p1b["map50"], errors="coerce")

    arch_avg = p1b.groupby("model")["map50"].mean().sort_values(ascending=False)
    print("\n  Architecture ranking (avg mAP@0.5 across seeds):")
    for model, score in arch_avg.items():
        print(f"    {model:20s}  {score:.4f}")

    top_2 = list(arch_avg.head(2).index)
    print(f"\n  Top 2: {top_2}")

    # GO/NO-GO gate
    best_map = arch_avg.iloc[0] if len(arch_avg) > 0 else 0
    if best_map < 0.70:
        print(f"\n  WARNING: Best mAP@0.5 = {best_map:.4f} < 70%. Protocol says STOP.")
        print("  Continuing anyway for completeness.")

    state["top_archs"] = top_2

    _plot_architecture_sweep(p1b)
    _generate_architecture_report(p1b, top_2, best_map)
    return top_2


def _plot_architecture_sweep(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall mAP@0.5
    avg = df.groupby("model")["map50"].apply(lambda x: pd.to_numeric(x, errors="coerce").mean()).sort_values()
    axes[0].barh(range(len(avg)), avg.values, color="#3b82f6")
    axes[0].set_yticks(range(len(avg)))
    axes[0].set_yticklabels(avg.index, fontsize=9)
    axes[0].set_xlabel("mAP@0.5")
    axes[0].set_title("Architecture Comparison (avg 2 seeds)")
    axes[0].grid(axis="x", alpha=0.3)

    # Per-class for top models
    colors = ["#3b82f6", "#f59e0b", "#ef4444", "#10b981"]
    top_models = avg.tail(4).index.tolist()
    x = np.arange(NUM_CLASSES)
    width = 0.2
    for i, model in enumerate(top_models):
        sub = df[df["model"] == model]
        vals = [pd.to_numeric(sub[f"map50_{name}"], errors="coerce").mean() for name in CLASS_NAMES]
        axes[1].bar(x + i * width, vals, width, label=model.replace(".pt", ""), color=colors[i % len(colors)])

    axes[1].set_xticks(x + width * 1.5)
    axes[1].set_xticklabels(CLASS_NAMES)
    axes[1].set_ylabel("mAP@0.5")
    axes[1].set_title("Per-Class Performance (top architectures)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p1b_architecture_sweep.png", dpi=150)
    plt.close(fig)


def _generate_architecture_report(df, top_2, best_map):
    avg = df.groupby("model")[["map50", "map50_95"]].apply(
        lambda x: pd.to_numeric(x.stack(), errors="coerce").unstack().mean()
    ).sort_values("map50", ascending=False)

    go = "GO" if best_map >= 0.70 else "NO-GO"
    lines = [
        "# Phase 1B: Architecture Sweep Report\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "Phase 1A (two-stage pipeline) skipped — using one-stage only.\n",
        "## Results\n",
        avg.to_markdown() + "\n",
        f"\n## Top 2 Architectures\n",
        f"1. {top_2[0]}\n" if len(top_2) > 0 else "",
        f"2. {top_2[1]}\n" if len(top_2) > 1 else "",
        f"\n## GO/NO-GO Gate\n",
        f"Best mAP@0.5 = {best_map:.4f} → **{go}** (threshold: 70%)\n",
        "![Architecture Sweep](../plots/p1b_architecture_sweep.png)\n",
    ]
    (REPORTS_DIR / "phase1b_architecture_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 2: Hyperparameter Optimization
# ---------------------------------------------------------------------------
def phase2_optimization(state: dict) -> dict:
    """Sequential optimization: LR -> Batch -> Augmentation on top 2 architectures."""
    print("\n" + "=" * 60)
    print("PHASE 2: Hyperparameter Optimization")
    print("=" * 60)

    imgsz = state["locked"].get("imgsz", 640)
    top_archs = state.get("top_archs", SMALL_MODELS[:2])
    best_configs: dict[str, dict] = {}

    for arch in top_archs:
        print(f"\n--- Optimizing {arch} ---")
        best_lr = 0.001
        best_batch = 16
        best_aug = "medium"

        # Step 1: LR sweep (single seed for speed)
        print("  Step 1: Learning rate sweep")
        lr_scores = {}
        for lr in LR_VALUES:
            run_id = f"p2_{arch.replace('.pt','')}_lr{lr}"
            r = run_training(arch, imgsz, run_id, "2", lr0=lr, seed=42)
            lr_scores[lr] = float(r.get("map50", 0))
        best_lr = max(lr_scores, key=lr_scores.get)
        print(f"  Best LR: {best_lr} (scores: {lr_scores})")

        # Step 2: Batch sweep
        print("  Step 2: Batch size sweep")
        batch_scores = {}
        for batch in BATCH_VALUES:
            run_id = f"p2_{arch.replace('.pt','')}_b{batch}"
            r = run_training(arch, imgsz, run_id, "2", lr0=best_lr, batch=batch, seed=42)
            batch_scores[batch] = float(r.get("map50", 0))
        best_batch = max(batch_scores, key=batch_scores.get)
        print(f"  Best batch: {best_batch} (scores: {batch_scores})")

        # Step 3: Augmentation sweep
        print("  Step 3: Augmentation sweep")
        aug_scores = {}
        for aug_name in AUG_PRESETS:
            run_id = f"p2_{arch.replace('.pt','')}_aug{aug_name}"
            r = run_training(arch, imgsz, run_id, "2", lr0=best_lr, batch=best_batch,
                             aug_preset=aug_name, seed=42)
            aug_scores[aug_name] = float(r.get("map50", 0))
        best_aug = max(aug_scores, key=aug_scores.get)
        print(f"  Best aug: {best_aug} (scores: {aug_scores})")

        # Confirmation run with both seeds
        print("  Confirmation runs (2 seeds)")
        for seed in SEEDS:
            run_id = f"p2_{arch.replace('.pt','')}_best_s{seed}"
            run_training(arch, imgsz, run_id, "2", lr0=best_lr, batch=best_batch,
                         aug_preset=best_aug, seed=seed)

        best_configs[arch] = {"lr0": best_lr, "batch": best_batch, "aug": best_aug}

    state["best_configs"] = best_configs
    print(f"\n  Best configs: {best_configs}")

    _plot_hyperparam_results()
    _generate_hyperparam_report(best_configs)
    return best_configs


def _plot_hyperparam_results() -> None:
    df = _read_results()
    p2 = df[df["phase"] == "2"].copy()
    if p2.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    p2["map50"] = pd.to_numeric(p2["map50"], errors="coerce")

    # LR
    lr_runs = p2[p2["run_id"].str.contains("_lr")]
    if not lr_runs.empty:
        for model in lr_runs["model"].unique():
            sub = lr_runs[lr_runs["model"] == model]
            axes[0].plot(sub["lr0"].astype(float), sub["map50"], "o-", label=model.replace(".pt", ""))
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("mAP@0.5")
    axes[0].set_title("LR Sweep")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Batch
    batch_runs = p2[p2["run_id"].str.contains("_b")]
    batch_runs = batch_runs[~batch_runs["run_id"].str.contains("_best")]
    if not batch_runs.empty:
        for model in batch_runs["model"].unique():
            sub = batch_runs[batch_runs["model"] == model]
            axes[1].plot(sub["batch"].astype(int), sub["map50"], "o-", label=model.replace(".pt", ""))
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("mAP@0.5")
    axes[1].set_title("Batch Sweep")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # Aug
    aug_runs = p2[p2["run_id"].str.contains("_aug")]
    if not aug_runs.empty:
        for model in aug_runs["model"].unique():
            sub = aug_runs[aug_runs["model"] == model]
            axes[2].bar(sub["aug"], sub["map50"], alpha=0.7, label=model.replace(".pt", ""))
    axes[2].set_xlabel("Augmentation")
    axes[2].set_ylabel("mAP@0.5")
    axes[2].set_title("Augmentation Sweep")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p2_hyperparam_optimization.png", dpi=150)
    plt.close(fig)


def _generate_hyperparam_report(best_configs):
    lines = [
        "# Phase 2: Hyperparameter Optimization Report\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "## Best Configurations\n",
    ]
    for arch, cfg in best_configs.items():
        lines.append(f"**{arch}**: lr0={cfg['lr0']}, batch={cfg['batch']}, aug={cfg['aug']}\n")
    lines.append("\n![Hyperparameter Results](../plots/p2_hyperparam_optimization.png)\n")
    (REPORTS_DIR / "phase2_hyperparam_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 3: Final Validation
# ---------------------------------------------------------------------------
def phase3_final_validation(state: dict) -> None:
    """Confusion matrix, confidence threshold sweep, TFLite export."""
    print("\n" + "=" * 60)
    print("PHASE 3: Final Validation")
    print("=" * 60)

    imgsz = state["locked"].get("imgsz", 640)
    best_configs = state.get("best_configs", {})

    if not best_configs:
        # Fallback: use top architecture with defaults
        top = state.get("top_archs", [BASELINE_MODEL])
        best_configs = {top[0]: {"lr0": 0.001, "batch": 16, "aug": "medium"}}

    # Pick the single best model (first in top_archs with best config)
    best_arch = list(best_configs.keys())[0]
    cfg = best_configs[best_arch]

    # Run final training with both seeds
    for seed in SEEDS:
        run_id = f"p3_final_{best_arch.replace('.pt','')}_s{seed}"
        run_training(best_arch, imgsz, run_id, "3",
                     lr0=cfg["lr0"], batch=cfg["batch"], aug_preset=cfg["aug"], seed=seed)

    # Find best weights from final runs
    best_pt = None
    best_score = -1
    df = _read_results()
    p3 = df[df["phase"] == "3"]
    for _, row in p3.iterrows():
        score = float(row.get("map50", 0))
        if score > best_score:
            best_score = score
            run_dir = RUNS_DIR / row["run_id"]
            candidate = run_dir / "weights" / "best.pt"
            if candidate.exists():
                best_pt = candidate

    if best_pt is None:
        print("  No best weights found. Skipping Phase 3 analysis.")
        return

    print(f"\n  Best model: {best_pt} (mAP@0.5 = {best_score:.4f})")

    # 1. Full evaluation with confusion matrix
    print("  Running full evaluation...")
    full_metrics = evaluate_extended(best_pt, DATA_YAML, imgsz=imgsz)
    cm = full_metrics.get("_confusion_matrix")
    if cm is not None:
        _plot_confusion_matrix(cm)

    # 2. Confidence threshold sweep
    print("  Running confidence threshold sweep...")
    threshold_results = []
    for conf in CONF_THRESHOLDS:
        m = evaluate_extended(best_pt, DATA_YAML, imgsz=imgsz, conf=conf)
        threshold_results.append({
            "conf": conf,
            "map50": m.get("map50", 0),
            "precision": m.get("precision", 0),
            "recall": m.get("recall", 0),
            "b2_b3_confusion": m.get("b2_b3_confusion"),
            "b3_b4_confusion": m.get("b3_b4_confusion"),
            "b4_recall": m.get("b4_recall"),
        })
    _plot_confidence_sweep(threshold_results)

    # 3. TFLite export
    print("  Attempting TFLite export...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(best_pt))
        export_path = model.export(format="tflite", imgsz=imgsz)
        export_size = Path(export_path).stat().st_size / 1024 / 1024 if export_path else 0
        print(f"  TFLite export: {export_path} ({export_size:.1f} MB)")
        tflite_status = f"Success: {export_size:.1f} MB"
    except Exception as e:
        print(f"  TFLite export failed: {e}")
        tflite_status = f"Failed: {e}"

    # 4. Decision framework
    map50 = full_metrics.get("map50", 0)
    b23_conf = full_metrics.get("b2_b3_confusion")
    min_class_ap = min(
        full_metrics.get(f"map50_{name}", 0) or 0 for name in CLASS_NAMES
    )

    if map50 >= 0.90 and (b23_conf is not None and b23_conf < 0.20) and min_class_ap >= 0.70:
        scenario = "EXCELLENT"
        action = "Deploy. Skip E1-E6."
    elif map50 >= 0.85 and (b23_conf is None or b23_conf < 0.30) and min_class_ap >= 0.70:
        scenario = "GOOD"
        action = "Deploy. Optional E1-E3."
    elif map50 >= 0.80 and min_class_ap >= 0.70:
        scenario = "ACCEPTABLE"
        action = "Deploy baseline. E1-E3 mandatory."
    elif map50 >= 0.75:
        scenario = "NEEDS WORK"
        action = "Don't deploy. Full E1-E6."
    else:
        scenario = "INSUFFICIENT"
        action = "STOP. Investigate data quality."

    print(f"\n  E0 Decision: {scenario}")
    print(f"  Action: {action}")

    _generate_final_report(full_metrics, threshold_results, tflite_status, scenario, action, best_arch, cfg)


def _plot_confusion_matrix(matrix: np.ndarray) -> None:
    # Normalize by column (actual class)
    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    norm = matrix / col_sums

    # Use only the first NUM_CLASSES rows/cols (exclude background)
    n = min(NUM_CLASSES, matrix.shape[0] - 1)
    cm = norm[:n, :n]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=12)

    # Highlight B2/B3 confusion
    from matplotlib.patches import Rectangle
    rect = Rectangle((0.5, 0.5), 2, 2, linewidth=3, edgecolor="red", facecolor="none", linestyle="--")
    ax.add_patch(rect)
    ax.text(2.6, 0.3, "B2/B3\nconfusion", color="red", fontsize=9, ha="left")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES[:n])
    ax.set_yticklabels(CLASS_NAMES[:n])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Normalized Confusion Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p3_confusion_matrix.png", dpi=150)
    plt.close(fig)


def _plot_confidence_sweep(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    confs = [r["conf"] for r in results]
    map50s = [r["map50"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]

    axes[0].plot(confs, map50s, "o-", color="#3b82f6", label="mAP@0.5", linewidth=2)
    axes[0].plot(confs, precisions, "s--", color="#f59e0b", label="Precision", linewidth=1.5)
    axes[0].plot(confs, recalls, "^--", color="#10b981", label="Recall", linewidth=1.5)
    axes[0].set_xlabel("Confidence Threshold")
    axes[0].set_ylabel("Metric Value")
    axes[0].set_title("Detection Metrics vs Confidence")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    b23 = [r.get("b2_b3_confusion") for r in results]
    b34 = [r.get("b3_b4_confusion") for r in results]
    b4r = [r.get("b4_recall") for r in results]

    if any(v is not None for v in b23):
        axes[1].plot(confs, [v or 0 for v in b23], "o-", color="#ef4444", label="B2/B3 Confusion", linewidth=2)
    if any(v is not None for v in b34):
        axes[1].plot(confs, [v or 0 for v in b34], "s-", color="#f59e0b", label="B3/B4 Confusion", linewidth=2)
    if any(v is not None for v in b4r):
        axes[1].plot(confs, [v or 0 for v in b4r], "^-", color="#10b981", label="B4 Recall", linewidth=2)
    axes[1].set_xlabel("Confidence Threshold")
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Confusion & B4 Recall vs Confidence")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p3_confidence_sweep.png", dpi=150)
    plt.close(fig)


def _generate_final_report(metrics, threshold_results, tflite, scenario, action, arch, cfg):
    b23 = metrics.get("b2_b3_confusion")
    b34 = metrics.get("b3_b4_confusion")
    lines = [
        "# Phase 3: Final Validation Report\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"## Best Model: {arch}\n",
        f"Config: lr0={cfg['lr0']}, batch={cfg['batch']}, aug={cfg['aug']}\n",
        "## Metrics\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| mAP@0.5 | {metrics.get('map50', 0):.4f} |",
        f"| mAP@0.5-0.95 | {metrics.get('map50_95', 0):.4f} |",
        f"| mAP@0.75 | {metrics.get('map75', 'n/a')} |",
        f"| Precision | {metrics.get('precision', 0):.4f} |",
        f"| Recall | {metrics.get('recall', 0):.4f} |",
        f"| B2/B3 Confusion | {b23:.4f if b23 else 'n/a'} |",
        f"| B3/B4 Confusion | {b34:.4f if b34 else 'n/a'} |",
        "",
        "### Per-Class AP@0.5\n",
    ]
    for name in CLASS_NAMES:
        v = metrics.get(f"map50_{name}", 0)
        lines.append(f"- {name}: {v:.4f}")

    lines += [
        "",
        "## Confusion Matrix\n",
        "![Confusion Matrix](../plots/p3_confusion_matrix.png)\n",
        "## Confidence Threshold Sweep\n",
        "![Confidence Sweep](../plots/p3_confidence_sweep.png)\n",
        f"## TFLite Export\n",
        f"{tflite}\n",
        f"## E0 Decision\n",
        f"**Scenario: {scenario}**\n",
        f"**Action: {action}**\n",
    ]
    (REPORTS_DIR / "phase3_final_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# E0 Summary Report
# ---------------------------------------------------------------------------
def _generate_e0_summary(state: dict) -> None:
    df = _read_results()
    total_runs = len(df[df["status"] == "ok"])
    total_time = pd.to_numeric(df["time_minutes"], errors="coerce").sum()

    lines = [
        "# E0 Baseline Protocol — Summary Report\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"Total runs: {total_runs} | Total GPU time: {total_time:.0f} minutes ({total_time/60:.1f} hours)\n",
        "## Phases Completed\n",
    ]
    for phase in state.get("completed_phases", []):
        lines.append(f"- {phase}")

    lines.append(f"\n## Locked Decisions\n")
    for key, val in state.get("locked", {}).items():
        lines.append(f"- {key}: {val}")

    lines.append(f"\n## Top Architectures\n")
    for arch in state.get("top_archs", []):
        lines.append(f"- {arch}")

    lines.append(f"\n## Best Configs\n")
    for arch, cfg in state.get("best_configs", {}).items():
        lines.append(f"- {arch}: {cfg}")

    lines.append("\n## Reports\n")
    if (REPORTS_DIR).exists():
        for f in sorted(REPORTS_DIR.glob("*.md")):
            lines.append(f"- [{f.name}]({f.name})")

    lines.append("\n## All Plots\n")
    if PLOTS_DIR.exists():
        for f in sorted(PLOTS_DIR.glob("*.png")):
            lines.append(f"- ![{f.stem}](../plots/{f.name})")

    (REPORTS_DIR / "e0_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="E0 Baseline Protocol Runner")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed phase")
    parser.add_argument("--status", action="store_true", help="Show progress")
    parser.add_argument("--phase", type=str, help="Run specific phase (0a, 0b, 0c, 1b, 2, 3)")
    args = parser.parse_args()

    # Setup directories
    for d in [E0_DIR, RUNS_DIR, PLOTS_DIR, REPORTS_DIR, SUBSETS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    _ensure_results_csv()

    state = _load_state()

    if args.status:
        print("E0 Protocol Status:")
        print(f"  Completed: {state.get('completed_phases', [])}")
        print(f"  Locked: {state.get('locked', {})}")
        print(f"  Top archs: {state.get('top_archs', [])}")
        print(f"  Best configs: {state.get('best_configs', {})}")
        df = _read_results()
        print(f"  Total runs: {len(df[df.get('status', pd.Series()) == 'ok']) if 'status' in df.columns else 0}")
        return

    phases_to_run = ["0A", "0B", "0C", "1B", "2", "3"]
    if args.phase:
        phases_to_run = [args.phase.upper()]

    print("=" * 60)
    print("E0 BASELINE EXPERIMENTAL PROTOCOL")
    print(f"Small models only | {MAX_EPOCHS} epochs | Seeds: {SEEDS}")
    print("=" * 60)

    for phase in phases_to_run:
        if args.resume and _is_phase_done(phase, state):
            print(f"\n[SKIP] Phase {phase} already completed")
            continue

        if phase == "0A":
            phase0a_eda()
            _mark_phase_complete("0A", state)

        elif phase == "0B":
            phase0b_resolution_sweep(state)
            _mark_phase_complete("0B", state)
            _save_state(state)

        elif phase == "0C":
            phase0c_learning_curve(state)
            _mark_phase_complete("0C", state)

        elif phase == "1B":
            phase1b_architecture_sweep(state)
            _mark_phase_complete("1B", state)
            _save_state(state)

        elif phase == "2":
            phase2_optimization(state)
            _mark_phase_complete("2", state)
            _save_state(state)

        elif phase == "3":
            phase3_final_validation(state)
            _mark_phase_complete("3", state)

    _generate_e0_summary(state)
    print("\n" + "=" * 60)
    print("E0 Protocol complete!")
    print(f"Results: {RESULTS_CSV}")
    print(f"Reports: {REPORTS_DIR}")
    print(f"Plots:   {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
backfill_metrics.py — Re-evaluate completed training runs whose metrics were not recorded.

Reads args.yaml from each run dir, calls evaluate_extended(), and updates e0_results/results.csv.
Safe to re-run: skips runs already marked status=ok in results.csv.
"""
from __future__ import annotations

import csv
import sys
import traceback
from pathlib import Path

import yaml

# Add repo to path so we can import e0_protocol helpers
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from e0_protocol import (
    RESULTS_CSV, RESULTS_COLUMNS, E0_DIR, RUNS_DIR, DATA_YAML,
    evaluate_extended, _append_result, _cleanup_gpu,
)


def _run_exists_ok(run_id: str) -> bool:
    if not RESULTS_CSV.exists():
        return False
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("run_id") == run_id and row.get("status") == "ok":
                return True
    return False


def _remove_failed_row(run_id: str) -> None:
    """Remove any existing rows for this run_id so we can write a clean ok row."""
    if not RESULTS_CSV.exists():
        return
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if r.get("run_id") != run_id]
    if len(kept) < len(rows):
        with RESULTS_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
            writer.writeheader()
            writer.writerows(kept)
        print(f"  Removed old failed row for {run_id}")


def _phase_from_run_id(run_id: str) -> str:
    prefix = run_id.split("_")[0]
    return {"p0a": "0A", "p0b": "0B", "p0c": "0C", "p1b": "1B", "p2": "2", "p3": "3"}.get(prefix, "?")


def backfill_run(run_dir: Path) -> None:
    run_id = run_dir.name
    phase = _phase_from_run_id(run_id)

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"
    if not best_pt.exists():
        print(f"  [SKIP] {run_id} — no weights found")
        return

    if _run_exists_ok(run_id):
        print(f"  [SKIP] {run_id} — already ok in results.csv")
        return

    # Read training args from the run's args.yaml
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.exists():
        print(f"  [SKIP] {run_id} — no args.yaml")
        return

    with args_yaml.open() as f:
        args = yaml.safe_load(f)

    imgsz = int(args.get("imgsz", 640))
    seed = int(args.get("seed", 0))
    batch = int(args.get("batch", 16))
    lr0 = float(args.get("lr0", 0.001))
    model_name = Path(args.get("model", "yolo11s.pt")).name
    if not model_name.endswith(".pt"):
        model_name += ".pt"

    # Determine aug preset (best effort from known patterns)
    hsv_s = args.get("hsv_s", 0.7)
    if hsv_s <= 0.4:
        aug_preset = "light"
    elif hsv_s <= 0.75:
        aug_preset = "medium"
    else:
        aug_preset = "heavy"

    print(f"  Evaluating {run_id} (imgsz={imgsz}, model={model_name}, seed={seed}) ...")

    result_row = {
        "run_id": run_id,
        "phase": phase,
        "model": model_name,
        "imgsz": imgsz,
        "seed": seed,
        "batch": batch,
        "lr0": lr0,
        "aug": aug_preset,
        "data_fraction": float(args.get("fraction", 1.0)),
    }

    try:
        metrics = evaluate_extended(best_pt, DATA_YAML, imgsz=imgsz)

        result_row.update({
            "map50": f"{metrics.get('map50', 0):.6f}",
            "map50_95": f"{metrics.get('map50_95', 0):.6f}",
            "map75": f"{metrics.get('map75', 0):.6f}" if metrics.get("map75") is not None else "",
            "precision": f"{metrics.get('precision', 0):.6f}",
            "recall": f"{metrics.get('recall', 0):.6f}",
            "b2_b3_confusion": f"{metrics.get('b2_b3_confusion', 0):.6f}" if metrics.get("b2_b3_confusion") is not None else "",
            "b3_b4_confusion": f"{metrics.get('b3_b4_confusion', 0):.6f}" if metrics.get("b3_b4_confusion") is not None else "",
            "b4_precision": f"{metrics.get('b4_precision', 0):.6f}" if metrics.get("b4_precision") is not None else "",
            "b4_recall": f"{metrics.get('b4_recall', 0):.6f}" if metrics.get("b4_recall") is not None else "",
            "time_minutes": "",
            "vram_gb": "",
            "status": "ok",
        })
        CLASS_NAMES = ["B1", "B2", "B3", "B4"]
        for name in CLASS_NAMES:
            result_row[f"map50_{name}"] = f"{metrics.get(f'map50_{name}', 0):.6f}"
            result_row[f"map50_95_{name}"] = f"{metrics.get(f'map50_95_{name}', 0):.6f}"

        print(f"    mAP@0.5={metrics.get('map50', 0):.4f} | precision={metrics.get('precision', 0):.4f} | recall={metrics.get('recall', 0):.4f}")

    except Exception:
        print(f"  [FAIL] {run_id}")
        traceback.print_exc()
        result_row["status"] = "fail_backfill"

    _remove_failed_row(run_id)
    _append_result(result_row)
    _cleanup_gpu()


def main():
    if not RUNS_DIR.exists():
        print("No runs directory found.")
        return

    run_dirs = sorted(RUNS_DIR.iterdir())
    print(f"Found {len(run_dirs)} run directories to process.\n")

    for run_dir in run_dirs:
        if run_dir.is_dir():
            backfill_run(run_dir)

    print("\nBackfill complete. Current results.csv:")
    import pandas as pd
    df = pd.read_csv(RESULTS_CSV)
    print(df[["run_id", "phase", "map50", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()

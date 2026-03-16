"""
Experiment orchestrator for BBC Autoresearch (Design v1).

FROZEN — the agent cannot modify this file.

Responsibilities:
- Experiment state tracking (state.json)
- Compilation gate (syntax + import validation)
- Smoke gate (dataset verification + one forward pass)
- Track classification (main vs exploration)
- Guardrails (VRAM cap, epoch cap, frozen eval/split)
- Metrics capture and logging
- Results ledger (experiments/results.tsv)
- Experiment log (experiments/log.md)
- Research notebook (experiments/summary.md)
- Batch reports (experiments/batch_NNN_report.md)
- Auto-rollback on crash/NaN/OOM
- Decision override (decide command)
"""

from __future__ import annotations

import contextlib
import csv
import importlib
import io
import json
import os
import py_compile
import shutil
import subprocess
import sys
import textwrap
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from ultralytics import YOLO

from prepare import (
    BEST_WEIGHTS,
    REPO_ROOT,
    RESULTS_TSV,
    RUNS_ROOT,
    TRAIN_RUN_DIR,
    evaluate_model,
    verify_dataset,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
STATE_PATH = EXPERIMENTS_DIR / "state.json"
SUMMARY_PATH = EXPERIMENTS_DIR / "summary.md"
LOG_PATH = EXPERIMENTS_DIR / "log.md"
LOGS_DIR = REPO_ROOT / "logs"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MEANINGFUL_IMPROVEMENT = 0.005  # mAP@0.5 delta for exploration progress
EXPLORATION_KILL_STREAK = 5  # consecutive misses before branch is killed
MAX_EPOCHS = 100  # hard epoch cap
VRAM_CAP_GB = 24.0  # warning threshold

# Agent-editable files used for track classification
MODELING_FILE = "modeling.py"
PIPELINE_FILE = "pipeline.py"
TRAIN_FILE = "train.py"
EDITABLE_FILES = [TRAIN_FILE, MODELING_FILE, PIPELINE_FILE]

RESULTS_COLUMNS = [
    "exp_id",
    "batch_id",
    "track",
    "seed",
    "commit",
    "title",
    "hypothesis",
    "success_criterion",
    "val_map50",
    "val_map50_95",
    "precision",
    "recall",
    "map50_B1",
    "map50_B2",
    "map50_B3",
    "map50_B4",
    "map50_95_B1",
    "map50_95_B2",
    "map50_95_B3",
    "map50_95_B4",
    "b2_b3_confusion",
    "b4_precision",
    "b4_recall",
    "time_minutes",
    "vram_gb",
    "provisional_status",
    "final_status",
    "status",
    "description",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
@dataclass
class TeeStream:
    file_obj: io.TextIOBase
    stream: io.TextIOBase

    def write(self, data: str) -> int:
        self.file_obj.write(data)
        self.file_obj.flush()
        return self.stream.write(data)

    def flush(self) -> None:
        self.file_obj.flush()
        self.stream.flush()


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "experiment"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_ratio(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def _run_git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _head_commit() -> str:
    return _run_git("rev-parse", "--short", "HEAD") or "uncommitted"


# ---------------------------------------------------------------------------
# Track classification
# ---------------------------------------------------------------------------
def _changed_live_files() -> set[str]:
    """Detect which agent-editable files have uncommitted changes."""
    names: set[str] = set()
    file_args = [f"-- {f}" for f in EDITABLE_FILES]
    try:
        diff_output = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "--"] + EDITABLE_FILES,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        names.update(line.strip() for line in diff_output.splitlines() if line.strip())

        status_output = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--"] + EDITABLE_FILES,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for line in status_output.splitlines():
            if len(line) >= 4:
                path = line[3:].strip()
                if path:
                    names.add(path)
    except Exception:
        pass
    return names


def _classify_track(track_hint: str, changed_files: set[str]) -> str:
    """Classify experiment track based on which files changed.

    Rule from design doc:
    - If modeling.py or pipeline.py is touched -> exploration
    - If only train.py -> main
    - TRACK_HINT can override
    """
    if track_hint in {"main", "exploration"}:
        return track_hint
    if MODELING_FILE in changed_files or PIPELINE_FILE in changed_files:
        return "exploration"
    return "main"


# ---------------------------------------------------------------------------
# Results TSV
# ---------------------------------------------------------------------------
def _read_results_rows() -> list[dict[str, str]]:
    if not RESULTS_TSV.exists():
        return []
    with RESULTS_TSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def ensure_results_schema() -> None:
    """Ensure results.tsv exists with the correct column schema."""
    rows = _read_results_rows()
    if not RESULTS_TSV.exists():
        RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS_TSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, delimiter="\t")
            writer.writeheader()
        return

    existing_columns = []
    with RESULTS_TSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            existing_columns = next(reader)
        except StopIteration:
            existing_columns = []

    if existing_columns == RESULTS_COLUMNS:
        return

    # Migrate: map old column names to new ones
    COLUMN_ALIASES = {
        "memory_gb": "vram_gb",
    }
    normalized_rows = []
    for row in rows:
        normalized: dict[str, str] = {}
        for column in RESULTS_COLUMNS:
            # Check direct match first, then aliases
            value = row.get(column, "")
            if not value:
                for old_name, new_name in COLUMN_ALIASES.items():
                    if new_name == column and old_name in row:
                        value = row[old_name]
                        break
            if not value and column == "description":
                value = row.get("title", "") or ""
            if not value and column == "final_status":
                value = row.get("status", "") or ""
            normalized[column] = value
        normalized_rows.append(normalized)

    with RESULTS_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(normalized_rows)


def _append_results_row(row: dict[str, Any]) -> None:
    ensure_results_schema()
    normalized = {column: row.get(column, "") for column in RESULTS_COLUMNS}
    with RESULTS_TSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, delimiter="\t")
        writer.writerow(normalized)


def _update_results_row(exp_id: int, updates: dict[str, str]) -> bool:
    """Update a specific row in results.tsv by exp_id. Returns True if found."""
    rows = _read_results_rows()
    found = False
    for row in rows:
        if str(row.get("exp_id", "")) == str(exp_id):
            row.update(updates)
            found = True
            break
    if found:
        with RESULTS_TSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    return found


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
def _bootstrap_main_best(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    """Find the best kept main-track experiment by mAP@0.5."""
    best_row = None
    best_score = float("-inf")
    for row in rows:
        status = (row.get("final_status") or row.get("status") or "").strip().lower()
        description = (row.get("description") or row.get("title") or "").strip().lower()
        if status != "keep":
            continue
        if "single-class" in description or "two-stage" in description or "classifier only" in description:
            continue
        score = _safe_float(row.get("val_map50"))
        if score > best_score:
            best_score = score
            best_row = row
    if best_row is None:
        return None
    return {
        "source": "legacy",
        "exp_id": best_row.get("exp_id") or "",
        "commit": best_row.get("commit", ""),
        "score": best_score,
        "title": best_row.get("title") or best_row.get("description") or "legacy best",
    }


def ensure_experiment_files() -> dict[str, Any]:
    """Initialize all experiment infrastructure and return current state."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_results_schema()

    if not STATE_PATH.exists():
        rows = _read_results_rows()
        state = {
            "next_exp_id": len(rows) + 1,
            "current_batch_id": 1,
            "main_best": _bootstrap_main_best(rows),
            "active_exploration": None,
            "exploration_best": None,
            "exploration_miss_streak": 0,
            "closed_explorations": [],
            "last_completed_exp": None,
        }
        save_state(state)
    else:
        state = load_state()

    if not LOG_PATH.exists():
        LOG_PATH.write_text("# Experiment Log\n\n", encoding="utf-8")
    if not SUMMARY_PATH.exists():
        SUMMARY_PATH.write_text("# Summary\n\n", encoding="utf-8")

    batch_path = EXPERIMENTS_DIR / f"batch_{state['current_batch_id']:03d}_report.md"
    if not batch_path.exists():
        batch_path.write_text(
            f"## Batch Report - Batch {state['current_batch_id']:03d}\n\nNo completed experiments yet.\n",
            encoding="utf-8",
        )

    regenerate_summary(state)
    regenerate_batch_report(state)
    return state


def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Log entries
# ---------------------------------------------------------------------------
def _append_log_entry(title: str, body: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"## {title}\n\n{body.strip()}\n\n---\n\n")


# ---------------------------------------------------------------------------
# Summary (agent's research notebook, per design doc Section 4)
# ---------------------------------------------------------------------------
def regenerate_summary(state: dict[str, Any]) -> None:
    """Regenerate experiments/summary.md from current state and results."""
    main_best = state.get("main_best") or {}
    exploration_best = state.get("exploration_best") or {}
    active_exploration = state.get("active_exploration")
    closed = state.get("closed_explorations") or []
    rows = _read_results_rows()

    # Current Best
    if main_best:
        current_best = (
            f"Main: Exp {main_best.get('exp_id') or main_best.get('commit', 'n/a')}, "
            f"{main_best.get('title', 'untitled')}, "
            f"mAP@0.5 = {main_best.get('score', 'n/a')}"
        )
    else:
        current_best = "Main: none"

    # Key Findings (recent kept experiments)
    key_findings = []
    for row in reversed(rows):
        effective_status = (row.get("final_status") or row.get("status") or "").strip().lower()
        if effective_status == "keep":
            title = row.get("title") or row.get("description") or "untitled"
            exp_id = row.get("exp_id", "?")
            map50 = row.get("val_map50", "")
            key_findings.append(f"- Exp {exp_id}: {title} -> mAP@0.5 = {map50}")
        if len(key_findings) >= 5:
            break
    if not key_findings:
        key_findings = ["- No confirmed kept improvements yet."]

    # Dead Ends
    dead_ends = [f"- {name}" for name in closed[-5:]] or ["- None."]

    # Open Hypotheses
    open_hypotheses = []
    if active_exploration:
        miss_streak = state.get("exploration_miss_streak", 0)
        open_hypotheses.append(
            f"- Active exploration: {active_exploration} "
            f"({miss_streak}/{EXPLORATION_KILL_STREAK} misses)"
        )
    parked = [
        row for row in rows
        if (row.get("final_status") or row.get("status") or "").strip().lower() == "park"
    ]
    for row in parked[-3:]:
        open_hypotheses.append(
            f"- PARKED: Exp {row.get('exp_id', '?')} - {row.get('title', 'untitled')}"
        )
    if not open_hypotheses:
        open_hypotheses = ["- None."]

    content = "\n".join([
        "## Current Best",
        current_best,
        "",
        "## Key Findings",
        *key_findings,
        "",
        "## Dead Ends",
        *dead_ends,
        "",
        "## Open Hypotheses",
        *open_hypotheses,
        "",
    ])
    SUMMARY_PATH.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Batch report (per design doc Section 8)
# ---------------------------------------------------------------------------
def regenerate_batch_report(state: dict[str, Any]) -> None:
    """Regenerate the current batch report."""
    batch_id = state["current_batch_id"]
    rows = [
        row for row in _read_results_rows()
        if str(row.get("batch_id") or "") == str(batch_id)
    ]
    report_path = EXPERIMENTS_DIR / f"batch_{batch_id:03d}_report.md"

    # Status counts
    status_counts: dict[str, int] = {}
    for row in rows:
        s = (row.get("final_status") or row.get("status") or "unknown").strip().lower()
        status_counts[s] = status_counts.get(s, 0) + 1

    n_total = len(rows)
    n_keep = status_counts.get("keep", 0)
    n_discard = status_counts.get("discard", 0)
    n_infra = status_counts.get("infra_fail", 0) + status_counts.get("crash", 0)
    n_park = status_counts.get("park", 0)

    # Significant findings
    keep_rows = [
        row for row in rows
        if (row.get("final_status") or row.get("status") or "").strip().lower() == "keep"
    ]
    findings = []
    for row in keep_rows[-5:]:
        title = row.get("title") or row.get("description") or "untitled"
        map50 = row.get("val_map50", "")
        findings.append(f"- Exp {row.get('exp_id', '?')}: {title} -> mAP@0.5 = {map50}")
    if not findings:
        findings = ["- No kept improvements in this batch yet."]

    # Dead ends
    closed = state.get("closed_explorations") or []
    dead_ends = [f"- {name}" for name in closed[-5:]] or ["- None in this batch."]

    # Current state
    main_best = state.get("main_best") or {}
    main_line = "Main: none"
    if main_best:
        main_line = (
            f"Main: Exp {main_best.get('exp_id') or main_best.get('commit', 'n/a')}, "
            f"mAP@0.5 = {main_best.get('score', 'n/a')}"
        )

    active = state.get("active_exploration")
    exploration_line = "Exploration: none active"
    if active:
        miss_streak = state.get("exploration_miss_streak", 0)
        exploration_line = f"Exploration: {active} ({miss_streak} exps since last improvement)"

    content = "\n".join([
        f"## Batch Report - Batch {batch_id:03d}",
        "",
        f"Summary: {n_total} experiments run. "
        f"{n_keep} kept, {n_discard} discarded, {n_park} parked, {n_infra} infrastructure failures.",
        "",
        "### Significant Findings",
        *findings,
        "",
        "### Dead Ends",
        *dead_ends,
        "",
        "### Current State",
        main_line,
        exploration_line,
        "",
        f"### Infrastructure",
        f"experiments={n_total} | kept={n_keep} | discarded={n_discard} | "
        f"parked={n_park} | infra_fail={n_infra}",
        "",
        "### Recommended Next",
        "_(agent fills this in)_",
        "",
        "### Items for Human Review",
        "_(agent flags overrides, adaptations, appeals here)_",
        "",
    ])
    report_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Batch management
# ---------------------------------------------------------------------------
def next_batch() -> None:
    state = ensure_experiment_files()
    state["current_batch_id"] += 1
    save_state(state)
    regenerate_summary(state)
    regenerate_batch_report(state)
    print(f"Moved to batch {state['current_batch_id']:03d}")


# ---------------------------------------------------------------------------
# Validation and gates
# ---------------------------------------------------------------------------
def _validate_spec(spec: dict[str, Any]) -> None:
    required = ["title", "hypothesis", "success_criterion", "track_hint", "seed", "model_ref", "train_args", "imgsz"]
    missing = [key for key in required if not spec.get(key) and spec.get(key) != 0]
    if missing:
        raise ValueError(f"Missing experiment spec fields: {missing}")
    if spec["track_hint"] not in {"auto", "main", "exploration"}:
        raise ValueError("TRACK_HINT must be one of: auto, main, exploration")

    # Epoch cap guardrail
    epochs = spec["train_args"].get("epochs", 30)
    if epochs > MAX_EPOCHS:
        raise ValueError(f"Epoch cap exceeded: {epochs} > {MAX_EPOCHS}. Hard limit is {MAX_EPOCHS}.")


def _compile_gate() -> None:
    """Syntax check all agent-editable files."""
    for filename in EDITABLE_FILES:
        filepath = REPO_ROOT / filename
        if filepath.exists():
            py_compile.compile(str(filepath), doraise=True)


def _import_module(name: str) -> ModuleType:
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def _resolve_experiment(spec: dict[str, Any]) -> tuple[YOLO, dict[str, Any]]:
    """Apply modeling and pipeline hooks, return resolved model and args."""
    train_args = deepcopy(spec["train_args"])
    model_ref = spec["model_ref"]

    # Step 1: modeling.configure_model
    modeling_module = _import_module("modeling")
    configured = modeling_module.configure_model(model_ref, train_args)
    if not isinstance(configured, tuple) or len(configured) != 2:
        raise TypeError("modeling.configure_model() must return (model_ref_or_yolo, train_args)")
    resolved_model, resolved_args = configured
    if not isinstance(resolved_args, dict):
        raise TypeError("modeling.configure_model() must return a dict for train_args")

    # Instantiate YOLO if needed
    if isinstance(resolved_model, YOLO):
        model = resolved_model
    else:
        model = YOLO(resolved_model)

    # Step 2: pipeline.configure_pipeline
    pipeline_module = _import_module("pipeline")
    configured = pipeline_module.configure_pipeline(model, resolved_args)
    if not isinstance(configured, tuple) or len(configured) != 2:
        raise TypeError("pipeline.configure_pipeline() must return (model, train_args)")
    model, resolved_args = configured
    if not isinstance(resolved_args, dict):
        raise TypeError("pipeline.configure_pipeline() must return a dict for train_args")

    return model, resolved_args


def _smoke_gate(spec: dict[str, Any]) -> None:
    """Dataset verification + one forward pass through the full pipeline."""
    verify_dataset()

    model, _ = _resolve_experiment(spec)
    imgsz = int(spec["imgsz"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device)
    model.model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        model.model(dummy)
    model.model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Experiment ID reservation
# ---------------------------------------------------------------------------
def _reserve_exp_id(state: dict[str, Any]) -> int:
    exp_id = int(state["next_exp_id"])
    state["next_exp_id"] = exp_id + 1
    save_state(state)
    return exp_id


# ---------------------------------------------------------------------------
# Baseline and decision logic (mAP@0.5 per design doc)
# ---------------------------------------------------------------------------
def _baseline_for_track(state: dict[str, Any], track: str, branch_name: str) -> float:
    if track == "exploration" and state.get("exploration_best") and state["exploration_best"].get("branch") == branch_name:
        return _safe_float(state["exploration_best"].get("score"))
    if track == "main" and state.get("main_best"):
        return _safe_float(state["main_best"].get("score"))
    return float("-inf")


def _decide_provisional_status(track: str, baseline_score: float, score: float) -> str:
    """Provisional decision based on mAP@0.5 vs baseline."""
    if baseline_score == float("-inf"):
        return "keep"
    if track == "exploration":
        return "keep" if score >= baseline_score + MEANINGFUL_IMPROVEMENT else "discard"
    return "keep" if score > baseline_score else "discard"


def _update_state_after_success(
    state: dict[str, Any], row: dict[str, Any], branch_name: str
) -> None:
    """Update state.json after a successful experiment, using mAP@0.5 as score."""
    track = row["track"]
    score = _safe_float(row["val_map50"])  # Decision metric = mAP@0.5
    exp_id = row["exp_id"]
    title = row["title"]
    effective_status = (row.get("final_status") or row.get("status") or "").strip().lower()

    # Only update baselines for kept experiments
    if effective_status not in ("keep",):
        state["last_completed_exp"] = exp_id
        if track == "exploration":
            state["exploration_miss_streak"] = int(state.get("exploration_miss_streak") or 0) + 1
            if state["exploration_miss_streak"] >= EXPLORATION_KILL_STREAK:
                closed = state.setdefault("closed_explorations", [])
                if branch_name not in closed:
                    closed.append(branch_name)
                state["active_exploration"] = None
                state["exploration_miss_streak"] = 0
                print(f"\n*** Exploration branch '{branch_name}' killed after {EXPLORATION_KILL_STREAK} consecutive misses ***")
        save_state(state)
        return

    if track == "main":
        current_best = _safe_float((state.get("main_best") or {}).get("score"))
        if state.get("main_best") is None or score > current_best:
            state["main_best"] = {
                "source": "experiments",
                "exp_id": exp_id,
                "commit": row["commit"],
                "score": score,
                "title": title,
            }
    else:
        best = state.get("exploration_best")
        if best is None or best.get("branch") != branch_name:
            # New exploration branch
            state["exploration_best"] = {
                "branch": branch_name,
                "exp_id": exp_id,
                "commit": row["commit"],
                "score": score,
                "title": title,
            }
            state["active_exploration"] = branch_name
            state["exploration_miss_streak"] = 0
        else:
            if score >= _safe_float(best.get("score")) + MEANINGFUL_IMPROVEMENT:
                state["exploration_best"] = {
                    "branch": branch_name,
                    "exp_id": exp_id,
                    "commit": row["commit"],
                    "score": score,
                    "title": title,
                }
                state["exploration_miss_streak"] = 0
            else:
                state["exploration_miss_streak"] = int(state.get("exploration_miss_streak") or 0) + 1
                if state["exploration_miss_streak"] >= EXPLORATION_KILL_STREAK:
                    closed = state.setdefault("closed_explorations", [])
                    if branch_name not in closed:
                        closed.append(branch_name)
                    state["active_exploration"] = None
                    state["exploration_miss_streak"] = 0
                    print(f"\n*** Exploration branch '{branch_name}' killed after {EXPLORATION_KILL_STREAK} consecutive misses ***")

    state["last_completed_exp"] = exp_id
    save_state(state)


# ---------------------------------------------------------------------------
# Results row construction
# ---------------------------------------------------------------------------
def _build_results_row(
    spec: dict[str, Any],
    exp_id: int,
    batch_id: int,
    track: str,
    metrics: dict[str, Any],
    peak_vram_mb: float,
    elapsed_seconds: float,
    provisional_status: str,
) -> dict[str, Any]:
    row = {
        "exp_id": exp_id,
        "batch_id": batch_id,
        "track": track,
        "seed": spec["seed"],
        "commit": _head_commit(),
        "title": spec["title"],
        "hypothesis": spec["hypothesis"],
        "success_criterion": spec["success_criterion"],
        "val_map50": f"{_safe_float(metrics.get('map50')):.6f}",
        "val_map50_95": f"{_safe_float(metrics.get('map50_95')):.6f}",
        "precision": f"{_safe_float(metrics.get('precision')):.6f}",
        "recall": f"{_safe_float(metrics.get('recall')):.6f}",
        "b2_b3_confusion": _safe_ratio(metrics.get("b2_b3_confusion")),
        "b4_precision": _safe_ratio(metrics.get("b4_precision")),
        "b4_recall": _safe_ratio(metrics.get("b4_recall")),
        "time_minutes": f"{elapsed_seconds / 60:.1f}",
        "vram_gb": f"{peak_vram_mb / 1024:.1f}",
        "provisional_status": provisional_status,
        "final_status": provisional_status,  # default: same as provisional
        "status": provisional_status,  # effective status
        "description": spec["title"],
    }
    for name in ["B1", "B2", "B3", "B4"]:
        row[f"map50_{name}"] = _safe_ratio(metrics.get(f"map50_{name}"))
        row[f"map50_95_{name}"] = _safe_ratio(metrics.get(f"map50_95_{name}"))
    return row


# ---------------------------------------------------------------------------
# Logging (per design doc Section 8)
# ---------------------------------------------------------------------------
def _log_success(
    spec: dict[str, Any],
    row: dict[str, Any],
    baseline_score: float,
    branch_name: str,
    changed_files: set[str],
    elapsed_seconds: float,
) -> None:
    """Append a structured experiment entry to log.md."""
    delta = _safe_float(row["val_map50"]) - baseline_score if baseline_score != float("-inf") else 0.0
    delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
    baseline_str = f"mAP@0.5 = {baseline_score:.4f}" if baseline_score != float("-inf") else "none"
    changed_str = ", ".join(sorted(changed_files)) if changed_files else "train.py"

    body = "\n".join([
        f"Track: {row['track']}" + (f":{branch_name}" if row['track'] == 'exploration' else ""),
        f"Hypothesis: {spec['hypothesis']}",
        f"Change: {changed_str}",
        f"Baseline: {baseline_str}",
        "",
        "Results:",
        (
            f"mAP@0.5: {row['val_map50']} ({delta_str}) | "
            f"B1={row.get('map50_B1', 'n/a')}, B2={row.get('map50_B2', 'n/a')}, "
            f"B3={row.get('map50_B3', 'n/a')}, B4={row.get('map50_B4', 'n/a')}"
        ),
        (
            f"mAP@0.5-0.95: {row['val_map50_95']} | "
            f"B2/B3 confusion: {row.get('b2_b3_confusion') or 'n/a'} | "
            f"B4 recall: {row.get('b4_recall') or 'n/a'} | "
            f"Time: {row.get('time_minutes', '?')}m | VRAM: {row.get('vram_gb', '?')}GB"
        ),
        "",
        "Decision:",
        f"Provisional: {row['provisional_status'].upper()}",
        f"Final: {row['final_status'].upper()}",
        "",
        "Analysis: _(agent fills in)_",
        "",
        "Next: _(agent fills in)_",
    ])
    _append_log_entry(f"Experiment {row['exp_id']}: {spec['title']}", body)


def _log_infra_failure(title: str, exp_id: int | None, track: str, error_text: str) -> None:
    label = f"Experiment {exp_id}: {title}" if exp_id else f"Gate Failure: {title}"
    body = "\n".join([
        f"Track: {track}",
        "Status: infra_fail",
        "",
        "Traceback:",
        "```text",
        error_text.strip()[-2000:],  # cap traceback length
        "```",
    ])
    _append_log_entry(label, body)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
def _print_metrics(
    metrics: dict[str, Any],
    model_name: str,
    optimizer: str,
    lr0: float,
    imgsz: int,
    batch: int,
    peak_vram_mb: float,
    elapsed: float,
) -> None:
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  mAP@0.5:          {_safe_float(metrics.get('map50')):.6f}  <-- decision metric")
    print(f"  mAP@0.5-0.95:     {_safe_float(metrics.get('map50_95')):.6f}")
    print(f"  precision:         {_safe_float(metrics.get('precision')):.6f}")
    print(f"  recall:            {_safe_float(metrics.get('recall')):.6f}")
    print(f"  peak VRAM:         {peak_vram_mb:.1f} MB ({peak_vram_mb/1024:.1f} GB)")
    print(f"  elapsed:           {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  model:             {model_name}")
    print(f"  optimizer:         {optimizer}")
    print(f"  lr0:               {lr0}")
    print(f"  imgsz:             {imgsz}")
    print(f"  batch:             {batch}")
    print()
    for name in ["B1", "B2", "B3", "B4"]:
        m50 = metrics.get(f"map50_{name}")
        m50_95 = metrics.get(f"map50_95_{name}")
        if m50 is not None:
            print(f"  {name}  mAP@0.5={_safe_float(m50):.4f}  mAP@0.5-0.95={_safe_float(m50_95):.4f}")
    if metrics.get("b2_b3_confusion") is not None:
        print(f"\n  B2/B3 confusion:   {_safe_float(metrics['b2_b3_confusion']):.4f}")
    if metrics.get("b4_precision") is not None:
        print(f"  B4 precision:      {_safe_float(metrics['b4_precision']):.4f}")
    if metrics.get("b4_recall") is not None:
        print(f"  B4 recall:         {_safe_float(metrics['b4_recall']):.4f}")

    if peak_vram_mb / 1024 > VRAM_CAP_GB:
        print(f"\n  *** WARNING: VRAM usage ({peak_vram_mb/1024:.1f} GB) exceeds {VRAM_CAP_GB} GB cap ***")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------
def run_from_train_spec(spec: dict[str, Any]) -> None:
    """Execute one experiment from a train.py spec."""
    state = ensure_experiment_files()
    _validate_spec(spec)
    changed_files = _changed_live_files()
    track = _classify_track(spec["track_hint"], changed_files)
    branch_name = (spec.get("exploration_name") or "").strip() or _slug(spec["title"])

    # --- Compilation and smoke gates ---
    try:
        _compile_gate()
        _smoke_gate(spec)
    except Exception:
        error_text = traceback.format_exc()
        _log_infra_failure(spec["title"], None, track, error_text)
        regenerate_summary(state)
        regenerate_batch_report(state)
        raise

    # --- Reserve experiment ID ---
    exp_id = _reserve_exp_id(state)
    batch_id = int(state["current_batch_id"])
    log_path = LOGS_DIR / f"{_now_stamp()}_exp{exp_id:03d}_{track}_{_slug(spec['title'])}.log"

    # --- Training ---
    try:
        model, train_args = _resolve_experiment(spec)

        shutil.rmtree(TRAIN_RUN_DIR, ignore_errors=True)

        peak_vram_mb = 0.0
        elapsed = 0.0
        with log_path.open("w", encoding="utf-8") as log_file:
            tee_stdout = TeeStream(log_file, sys.stdout)
            tee_stderr = TeeStream(log_file, sys.stderr)
            with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
                print(f"Experiment {exp_id}: {spec['title']}")
                print(f"Track: {track}")
                print(f"Hypothesis: {spec['hypothesis']}")
                print("Verifying dataset...")
                verify_dataset()
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA GPU required for this repository.")
                torch.cuda.reset_peak_memory_stats()
                t0 = datetime.now()
                repo_cwd = os.getcwd()
                os.chdir(Path(train_args["data"]).parent)
                try:
                    model.train(**train_args)
                finally:
                    os.chdir(repo_cwd)
                elapsed = (datetime.now() - t0).total_seconds()
                print("\nEvaluating best.pt...")
                metrics = evaluate_model(BEST_WEIGHTS)
                peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                _print_metrics(
                    metrics,
                    str(spec["model_ref"]),
                    str(train_args.get("optimizer", "")),
                    float(train_args.get("lr0", 0.0)),
                    int(train_args.get("imgsz", 0)),
                    int(train_args.get("batch", 0)),
                    peak_vram_mb,
                    elapsed,
                )
    except Exception:
        error_text = traceback.format_exc()
        _log_infra_failure(spec["title"], exp_id, track, error_text)
        row = {
            "exp_id": exp_id,
            "batch_id": batch_id,
            "track": track,
            "seed": spec["seed"],
            "commit": _head_commit(),
            "title": spec["title"],
            "hypothesis": spec["hypothesis"],
            "success_criterion": spec["success_criterion"],
            "val_map50": "0.000000",
            "val_map50_95": "0.000000",
            "precision": "0.000000",
            "recall": "0.000000",
            "time_minutes": "",
            "vram_gb": "0.0",
            "provisional_status": "infra_fail",
            "final_status": "infra_fail",
            "status": "infra_fail",
            "description": spec["title"],
        }
        _append_results_row(row)
        regenerate_summary(state)
        regenerate_batch_report(state)
        raise

    # --- Decision ---
    baseline_score = _baseline_for_track(state, track, branch_name)
    score = _safe_float(metrics.get("map50"))  # Decision metric = mAP@0.5
    provisional_status = _decide_provisional_status(track, baseline_score, score)
    row = _build_results_row(spec, exp_id, batch_id, track, metrics, peak_vram_mb, elapsed, provisional_status)
    _append_results_row(row)
    _log_success(spec, row, baseline_score, branch_name, changed_files, elapsed)
    _update_state_after_success(state, row, branch_name if track == "exploration" else "main")
    regenerate_summary(load_state())
    regenerate_batch_report(load_state())

    # Print decision
    print(f"\nProvisional: {provisional_status.upper()}")
    if provisional_status == "keep":
        print("Use `uv run orchestrator.py decide <exp_id> PARK <reason>` to override if needed.")
    else:
        print("Use `uv run orchestrator.py decide <exp_id> KEEP|PARK <reason>` to override.")


# ---------------------------------------------------------------------------
# Decision override command
# ---------------------------------------------------------------------------
def decide(exp_id_str: str, decision: str, justification: str) -> None:
    """Override the provisional status of an experiment.

    Usage: uv run orchestrator.py decide <exp_id> <KEEP|DISCARD|PARK> <justification>

    PARK means: set aside for later, idea not dead. The signal is interesting
    enough to revisit, possibly in combination with something else.
    """
    decision = decision.strip().lower()
    if decision not in ("keep", "discard", "park"):
        print(f"Invalid decision: {decision}. Must be one of: keep, discard, park")
        sys.exit(1)

    exp_id = int(exp_id_str)
    state = ensure_experiment_files()

    # Find the experiment in results
    rows = _read_results_rows()
    target_row = None
    for row in rows:
        if str(row.get("exp_id", "")) == str(exp_id):
            target_row = row
            break

    if target_row is None:
        print(f"Experiment {exp_id} not found in results.tsv")
        sys.exit(1)

    old_status = target_row.get("final_status") or target_row.get("status") or "unknown"
    if old_status == decision:
        print(f"Experiment {exp_id} already has status '{decision}'. No change.")
        return

    # Update results.tsv
    _update_results_row(exp_id, {
        "final_status": decision,
        "status": decision,
    })

    # Log the override
    _append_log_entry(
        f"Decision Override: Experiment {exp_id}",
        "\n".join([
            f"Previous: {old_status.upper()}",
            f"New: {decision.upper()}",
            f"Justification: {justification}",
        ]),
    )

    # Re-bootstrap state from results to ensure consistency
    rows = _read_results_rows()
    new_main_best = _bootstrap_main_best(rows)
    if new_main_best:
        state["main_best"] = new_main_best
    save_state(state)
    regenerate_summary(state)
    regenerate_batch_report(state)

    print(f"Experiment {exp_id}: {old_status.upper()} -> {decision.upper()}")
    print(f"Justification: {justification}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print("Usage:")
        print("  uv run train.py                                          # run experiment")
        print("  uv run orchestrator.py next-batch                        # advance batch")
        print("  uv run orchestrator.py decide <exp_id> <KEEP|DISCARD|PARK> <justification>")
        return

    cmd = argv[0]
    if cmd == "next-batch":
        next_batch()
    elif cmd == "decide":
        if len(argv) < 4:
            print("Usage: uv run orchestrator.py decide <exp_id> <KEEP|DISCARD|PARK> <justification>")
            sys.exit(1)
        decide(argv[1], argv[2], " ".join(argv[3:]))
    else:
        print(f"Unknown command: {cmd}")
        print("Available: next-batch, decide")
        sys.exit(1)


if __name__ == "__main__":
    main()

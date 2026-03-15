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

from prepare import BEST_WEIGHTS, REPO_ROOT, RESULTS_TSV, RUNS_ROOT, TRAIN_RUN_DIR, evaluate_model, verify_dataset

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
STATE_PATH = EXPERIMENTS_DIR / "state.json"
SUMMARY_PATH = EXPERIMENTS_DIR / "summary.md"
LOG_PATH = EXPERIMENTS_DIR / "log.md"
REPORTS_DIR = EXPERIMENTS_DIR / "reports"
LOGS_DIR = REPO_ROOT / "logs"

MEANINGFUL_IMPROVEMENT = 0.005
RESULTS_COLUMNS = [
    "commit",
    "val_map50",
    "val_map50_95",
    "precision",
    "recall",
    "memory_gb",
    "status",
    "description",
    "exp_id",
    "batch_id",
    "track",
    "seed",
    "title",
    "hypothesis",
    "provisional_status",
    "success_criterion",
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
]


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


def _changed_live_files() -> set[str]:
    names: set[str] = set()
    try:
        diff_output = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "--", "train.py", "research.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        names.update(line.strip() for line in diff_output.splitlines() if line.strip())

        status_output = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", "train.py", "research.py"],
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
    if track_hint in {"main", "exploration"}:
        return track_hint
    if "research.py" in changed_files:
        return "exploration"
    if changed_files == {"train.py"}:
        return "main"
    return "main"


def _read_results_rows() -> list[dict[str, str]]:
    if not RESULTS_TSV.exists():
        return []
    with RESULTS_TSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def ensure_results_schema() -> None:
    rows = _read_results_rows()
    if not RESULTS_TSV.exists():
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

    normalized_rows = []
    for row in rows:
        normalized = {column: row.get(column, "") for column in RESULTS_COLUMNS}
        if not normalized["description"]:
            normalized["description"] = row.get("title", "") or ""
        normalized_rows.append(normalized)

    with RESULTS_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(normalized_rows)


def _bootstrap_main_best(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    best_row = None
    best_score = float("-inf")
    for row in rows:
        status = (row.get("status") or "").strip().lower()
        description = (row.get("description") or row.get("title") or "").strip().lower()
        if status != "keep":
            continue
        if "single-class" in description or "two-stage" in description or "classifier only" in description:
            continue
        score = _safe_float(row.get("val_map50_95"))
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
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
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

    report_path = REPORTS_DIR / f"batch_{state['current_batch_id']:03d}_report.md"
    if not report_path.exists():
        report_path.write_text(
            f"# Batch {state['current_batch_id']:03d} Report\n\nNo completed experiments yet.\n",
            encoding="utf-8",
        )

    regenerate_summary(state)
    regenerate_batch_report(state)
    return state


def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _append_results_row(row: dict[str, Any]) -> None:
    ensure_results_schema()
    normalized = {column: row.get(column, "") for column in RESULTS_COLUMNS}
    with RESULTS_TSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, delimiter="\t")
        writer.writerow(normalized)


def _append_log_entry(title: str, body: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"## {title}\n\n{body.strip()}\n\n")


def regenerate_summary(state: dict[str, Any]) -> None:
    main_best = state.get("main_best") or {}
    exploration_best = state.get("exploration_best") or {}
    active_exploration = state.get("active_exploration")
    closed = state.get("closed_explorations") or []
    rows = _read_results_rows()
    recent = rows[-5:]
    key_findings = []
    for row in reversed(recent):
        if (row.get("status") or "").strip().lower() == "keep":
            key_findings.append(
                f"- Exp {row.get('exp_id') or '?'} ({row.get('track') or 'legacy'}): "
                f"{row.get('title') or row.get('description') or 'untitled'} -> "
                f"mAP50-95 {row.get('val_map50_95', '')}"
            )
        if len(key_findings) == 3:
            break

    if not key_findings:
        key_findings = ["- No confirmed kept improvements in the current tracked window."]

    dead_ends = [f"- {name}" for name in closed[-5:]] or ["- None."]
    open_hypotheses = []
    if active_exploration:
        open_hypotheses.append(f"- Active exploration: {active_exploration}")
    last_row = recent[-1] if recent else None
    if last_row and last_row.get("hypothesis"):
        open_hypotheses.append(f"- Last hypothesis: {last_row['hypothesis']}")
    if not open_hypotheses:
        open_hypotheses = ["- None."]

    content = "\n".join(
        [
            "# Summary",
            "",
            "## Current Best",
            (
                f"Main: {main_best.get('title', 'none')} "
                f"(score={main_best.get('score', 'n/a')}, exp={main_best.get('exp_id') or main_best.get('commit') or 'n/a'})"
                if main_best
                else "Main: none"
            ),
            (
                f"Exploration: {exploration_best.get('title', 'none')} "
                f"(score={exploration_best.get('score', 'n/a')}, branch={exploration_best.get('branch', 'n/a')})"
                if exploration_best
                else "Exploration: none"
            ),
            "",
            "## Active Exploration",
            f"- {active_exploration}" if active_exploration else "- None.",
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
        ]
    )
    SUMMARY_PATH.write_text(content, encoding="utf-8")


def regenerate_batch_report(state: dict[str, Any]) -> None:
    batch_id = state["current_batch_id"]
    rows = [row for row in _read_results_rows() if str(row.get("batch_id") or "") == str(batch_id)]
    report_path = REPORTS_DIR / f"batch_{batch_id:03d}_report.md"
    keep_rows = [row for row in rows if (row.get("status") or "").strip().lower() == "keep"]
    status_counts: dict[str, int] = {}
    for row in rows:
        status = (row.get("status") or "").strip().lower() or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

    findings = [
        f"- Exp {row.get('exp_id')}: {row.get('title') or row.get('description')} -> {row.get('val_map50_95')}"
        for row in keep_rows[-5:]
    ] or ["- No kept improvements in this batch yet."]

    infra = ", ".join(f"{name}={count}" for name, count in sorted(status_counts.items())) or "none"
    content = "\n".join(
        [
            f"# Batch {batch_id:03d} Report",
            "",
            f"- total rows: {len(rows)}",
            f"- active exploration: {state.get('active_exploration') or 'none'}",
            f"- last completed exp: {state.get('last_completed_exp') or 'none'}",
            f"- status counts: {infra}",
            "",
            "## Significant Findings",
            *findings,
            "",
        ]
    )
    report_path.write_text(content, encoding="utf-8")


def next_batch() -> None:
    state = ensure_experiment_files()
    state["current_batch_id"] += 1
    save_state(state)
    regenerate_summary(state)
    regenerate_batch_report(state)
    print(f"Moved to batch {state['current_batch_id']:03d}")


def _validate_spec(spec: dict[str, Any]) -> None:
    required = ["title", "hypothesis", "success_criterion", "track_hint", "seed", "model_ref", "train_args", "imgsz"]
    missing = [key for key in required if not spec.get(key) and spec.get(key) != 0]
    if missing:
        raise ValueError(f"Missing experiment spec fields: {missing}")
    if spec["track_hint"] not in {"auto", "main", "exploration"}:
        raise ValueError("TRACK_HINT must be one of: auto, main, exploration")


def _compile_gate() -> None:
    py_compile.compile(str(REPO_ROOT / "train.py"), doraise=True)
    py_compile.compile(str(REPO_ROOT / "research.py"), doraise=True)


def _import_module(name: str) -> ModuleType:
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def _resolve_experiment(spec: dict[str, Any], research_module: ModuleType) -> tuple[YOLO, dict[str, Any]]:
    train_args = deepcopy(spec["train_args"])
    model_ref = spec["model_ref"]

    configured = research_module.configure_experiment(model_ref, train_args)
    if not isinstance(configured, tuple) or len(configured) != 2:
        raise TypeError("research.configure_experiment() must return (model_ref_or_yolo, train_args)")

    resolved_model, resolved_args = configured
    if not isinstance(resolved_args, dict):
        raise TypeError("research.configure_experiment() must return a dict for train_args")

    if isinstance(resolved_model, YOLO):
        model = resolved_model
    else:
        model = YOLO(resolved_model)
    return model, resolved_args


def _smoke_gate(spec: dict[str, Any], research_module: ModuleType) -> None:
    verify_dataset()
    if hasattr(research_module, "validate_experiment"):
        research_module.validate_experiment()

    model, _ = _resolve_experiment(spec, research_module)
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


def _reserve_exp_id(state: dict[str, Any]) -> int:
    exp_id = int(state["next_exp_id"])
    state["next_exp_id"] = exp_id + 1
    save_state(state)
    return exp_id


def _baseline_for_track(state: dict[str, Any], track: str, branch_name: str) -> float:
    if track == "exploration" and state.get("exploration_best") and state["exploration_best"].get("branch") == branch_name:
        return _safe_float(state["exploration_best"].get("score"))
    if track == "main" and state.get("main_best"):
        return _safe_float(state["main_best"].get("score"))
    return float("-inf")


def _update_state_after_success(state: dict[str, Any], row: dict[str, Any], branch_name: str) -> None:
    track = row["track"]
    score = _safe_float(row["val_map50_95"])
    exp_id = row["exp_id"]
    title = row["title"]

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
                if state["exploration_miss_streak"] >= 5:
                    closed = state.setdefault("closed_explorations", [])
                    if branch_name not in closed:
                        closed.append(branch_name)
                    state["active_exploration"] = None
                    state["exploration_miss_streak"] = 0

    state["last_completed_exp"] = exp_id
    save_state(state)


def _build_results_row(
    spec: dict[str, Any],
    exp_id: int,
    batch_id: int,
    track: str,
    metrics: dict[str, Any],
    peak_vram_mb: float,
    provisional_status: str,
) -> dict[str, Any]:
    row = {
        "commit": _head_commit(),
        "val_map50": f"{_safe_float(metrics.get('map50')):.6f}",
        "val_map50_95": f"{_safe_float(metrics.get('map50_95')):.6f}",
        "precision": f"{_safe_float(metrics.get('precision')):.6f}",
        "recall": f"{_safe_float(metrics.get('recall')):.6f}",
        "memory_gb": f"{peak_vram_mb / 1024:.1f}",
        "status": provisional_status,
        "description": spec["title"],
        "exp_id": exp_id,
        "batch_id": batch_id,
        "track": track,
        "seed": spec["seed"],
        "title": spec["title"],
        "hypothesis": spec["hypothesis"],
        "provisional_status": provisional_status,
        "success_criterion": spec["success_criterion"],
        "b2_b3_confusion": _safe_ratio(metrics.get("b2_b3_confusion")),
        "b4_precision": _safe_ratio(metrics.get("b4_precision")),
        "b4_recall": _safe_ratio(metrics.get("b4_recall")),
    }
    for name in ["B1", "B2", "B3", "B4"]:
        row[f"map50_{name}"] = _safe_ratio(metrics.get(f"map50_{name}"))
        row[f"map50_95_{name}"] = _safe_ratio(metrics.get(f"map50_95_{name}"))
    return row


def _decide_provisional_status(track: str, baseline_score: float, score: float) -> str:
    if baseline_score == float("-inf"):
        return "keep"
    if track == "exploration":
        return "keep" if score >= baseline_score + MEANINGFUL_IMPROVEMENT else "discard"
    return "keep" if score > baseline_score else "discard"


def _log_success(spec: dict[str, Any], row: dict[str, Any], baseline_score: float, branch_name: str) -> None:
    body = "\n".join(
        [
            f"Track: {row['track']}",
            f"Branch: {branch_name or 'main'}",
            f"Hypothesis: {spec['hypothesis']}",
            f"Success criterion: {spec['success_criterion']}",
            f"Baseline score: {baseline_score:.6f}" if baseline_score != float("-inf") else "Baseline score: none",
            "",
            "Results:",
            (
                f"mAP50-95={row['val_map50_95']} | mAP50={row['val_map50']} | "
                f"B4 P/R={row['b4_precision'] or 'n/a'}/{row['b4_recall'] or 'n/a'} | "
                f"B2/B3 confusion={row['b2_b3_confusion'] or 'n/a'}"
            ),
            f"Decision: {row['provisional_status']}",
        ]
    )
    _append_log_entry(f"Experiment {row['exp_id']}: {spec['title']}", body)


def _log_infra_failure(title: str, exp_id: int | None, track: str, error_text: str) -> None:
    label = f"Experiment {exp_id}: {title}" if exp_id else f"Gate Failure: {title}"
    body = "\n".join(
        [
            f"Track: {track}",
            "Status: infra_fail",
            "",
            "Traceback:",
            "```text",
            error_text.strip(),
            "```",
        ]
    )
    _append_log_entry(label, body)


def _print_metrics(metrics: dict[str, Any], model_name: str, optimizer: str, lr0: float, imgsz: int, batch: int, peak_vram_mb: float, elapsed: float) -> None:
    print("\n---")
    print(f"val_map50:        {_safe_float(metrics.get('map50')):.6f}")
    print(f"val_map50_95:     {_safe_float(metrics.get('map50_95')):.6f}")
    print(f"precision:        {_safe_float(metrics.get('precision')):.6f}")
    print(f"recall:           {_safe_float(metrics.get('recall')):.6f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"total_seconds:    {elapsed:.1f}")
    print(f"model:            {model_name}")
    print(f"optimizer:        {optimizer}")
    print(f"lr0:              {lr0}")
    print(f"imgsz:            {imgsz}")
    print(f"batch:            {batch}")
    for name in ["B1", "B2", "B3", "B4"]:
        if metrics.get(f"map50_{name}") is not None:
            print(f"map50_{name}:        {_safe_float(metrics.get(f'map50_{name}')):.6f}")
        if metrics.get(f"map50_95_{name}") is not None:
            print(f"map50_95_{name}:     {_safe_float(metrics.get(f'map50_95_{name}')):.6f}")
    if metrics.get("b2_b3_confusion") is not None:
        print(f"b2_b3_confusion: {_safe_float(metrics['b2_b3_confusion']):.6f}")
    if metrics.get("b4_precision") is not None:
        print(f"b4_precision:    {_safe_float(metrics['b4_precision']):.6f}")
    if metrics.get("b4_recall") is not None:
        print(f"b4_recall:       {_safe_float(metrics['b4_recall']):.6f}")


def run_from_train_spec(spec: dict[str, Any]) -> None:
    state = ensure_experiment_files()
    _validate_spec(spec)
    changed_files = _changed_live_files()
    track = _classify_track(spec["track_hint"], changed_files)
    branch_name = (spec.get("exploration_name") or "").strip() or _slug(spec["title"])

    try:
        _compile_gate()
        research_module = _import_module(spec["research_module"])
        _smoke_gate(spec, research_module)
    except Exception:
        error_text = traceback.format_exc()
        _log_infra_failure(spec["title"], None, track, error_text)
        regenerate_summary(state)
        regenerate_batch_report(state)
        raise

    exp_id = _reserve_exp_id(state)
    batch_id = int(state["current_batch_id"])
    log_path = LOGS_DIR / f"{_now_stamp()}_exp{exp_id:03d}_{track}_{_slug(spec['title'])}.log"

    try:
        research_module = _import_module(spec["research_module"])
        model, train_args = _resolve_experiment(spec, research_module)

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
            "commit": _head_commit(),
            "val_map50": "0.000000",
            "val_map50_95": "0.000000",
            "precision": "0.000000",
            "recall": "0.000000",
            "memory_gb": "0.0",
            "status": "infra_fail",
            "description": spec["title"],
            "exp_id": exp_id,
            "batch_id": batch_id,
            "track": track,
            "seed": spec["seed"],
            "title": spec["title"],
            "hypothesis": spec["hypothesis"],
            "provisional_status": "infra_fail",
            "success_criterion": spec["success_criterion"],
        }
        _append_results_row(row)
        regenerate_summary(state)
        regenerate_batch_report(state)
        raise

    baseline_score = _baseline_for_track(state, track, branch_name)
    score = _safe_float(metrics.get("map50_95"))
    provisional_status = _decide_provisional_status(track, baseline_score, score)
    row = _build_results_row(spec, exp_id, batch_id, track, metrics, peak_vram_mb, provisional_status)
    _append_results_row(row)
    _log_success(spec, row, baseline_score, branch_name)
    _update_state_after_success(state, row, branch_name if track == "exploration" else "main")
    regenerate_summary(load_state())
    regenerate_batch_report(load_state())


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or sys.argv[1:])
    if argv and argv[0] == "next-batch":
        next_batch()
        return
    print("Usage:")
    print("  uv run train.py")
    print("  uv run orchestrator.py next-batch")


if __name__ == "__main__":
    main()

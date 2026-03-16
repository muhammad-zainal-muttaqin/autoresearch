"""Generate e0_progress.png — Karpathy-style progress chart.

Shows all E0 runs as gray dots, improvements (new running best) as green
circles with a step-line and diagonal annotations explaining what changed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

E0_CSV = Path("e0_results") / "results.csv"
OUTPUT_PATH = Path("e0_progress.png")

GREEN = "#2ca02c"
GRAY  = "#bbbbbb"


def _make_label(row: pd.Series, prev_best_row: pd.Series | None) -> str:
    """Generate a short human-readable label describing what changed."""
    model   = str(row.get("model", "")).replace(".pt", "")
    imgsz   = int(row.get("imgsz", 640))
    seed    = int(row.get("seed", 0))
    phase   = str(row.get("phase", "?"))
    aug     = str(row.get("aug", "medium"))
    lr0     = float(row.get("lr0", 0.001))
    batch   = int(row.get("batch", 16))

    if prev_best_row is None:
        return f"baseline\n{model} {imgsz}px"

    parts = []
    prev_model  = str(prev_best_row.get("model", "")).replace(".pt", "")
    prev_imgsz  = int(prev_best_row.get("imgsz", 640))
    prev_aug    = str(prev_best_row.get("aug", "medium"))
    prev_lr0    = float(prev_best_row.get("lr0", 0.001))
    prev_batch  = int(prev_best_row.get("batch", 16))

    if model != prev_model:
        parts.append(f"arch {prev_model}→{model}")
    if imgsz != prev_imgsz:
        parts.append(f"imgsz {prev_imgsz}→{imgsz}px")
    if aug != prev_aug:
        parts.append(f"aug {prev_aug}→{aug}")
    if abs(lr0 - prev_lr0) > 1e-6:
        parts.append(f"lr {prev_lr0}→{lr0}")
    if batch != prev_batch:
        parts.append(f"batch {prev_batch}→{batch}")
    if not parts:
        parts.append(f"seed {seed} (phase {phase})")

    return "\n".join(parts)


def main() -> None:
    if not E0_CSV.exists():
        print(f"{E0_CSV} not found — no E0 runs yet.")
        return

    df = pd.read_csv(E0_CSV)
    df = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    if df.empty or "map50" not in df.columns:
        print("No completed E0 runs yet.")
        return

    df["map50"] = pd.to_numeric(df["map50"], errors="coerce")
    df = df.dropna(subset=["map50"]).reset_index(drop=True)
    df["idx"] = range(1, len(df) + 1)

    # Compute running best and identify improvement points
    running_best  = -np.inf
    is_improvement = []
    for v in df["map50"]:
        if v > running_best:
            running_best = v
            is_improvement.append(True)
        else:
            is_improvement.append(False)
    df["improvement"] = is_improvement

    improved = df[df["improvement"]].copy()
    n_total  = len(df)
    n_kept   = len(improved)

    # Build step-line: for each improvement, the line stays flat until next
    step_x, step_y = [], []
    for i, (_, row) in enumerate(improved.iterrows()):
        x_next = improved.iloc[i + 1]["idx"] if i + 1 < len(improved) else n_total + 1
        step_x += [row["idx"], x_next]
        step_y += [row["map50"], row["map50"]]
        if i > 0:
            # Vertical connector from previous level
            step_x.insert(-2, improved.iloc[i - 1]["idx"] if False else row["idx"])
            step_y.insert(-2, step_y[-4] if len(step_y) >= 4 else row["map50"])

    # Rebuild step-line cleanly
    sx, sy = [], []
    prev_y = None
    for _, row in improved.iterrows():
        if prev_y is not None:
            sx.append(row["idx"])
            sy.append(prev_y)
        sx.append(row["idx"])
        sy.append(row["map50"])
        prev_y = row["map50"]
    # Extend flat line to end
    if sx:
        sx.append(n_total + 0.6)
        sy.append(sy[-1])

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # All runs — gray dots
    ax.scatter(df[~df["improvement"]]["idx"],
               df[~df["improvement"]]["map50"],
               s=28, color=GRAY, alpha=0.65, zorder=2, label="Not improved")

    # Step-line
    if sx:
        ax.plot(sx, sy, color=GREEN, linewidth=1.6, zorder=3, solid_capstyle="round")

    # Improvement dots
    ax.scatter(improved["idx"], improved["map50"],
               s=80, color=GREEN, edgecolors="white", linewidths=1.2,
               zorder=4, label="New best (kept)")

    # Diagonal annotations
    prev_best_row = None
    for _, row in improved.iterrows():
        label = _make_label(row, prev_best_row)
        prev_best_row = row

        txt = ax.annotate(
            f"{label}\n({row['map50']:.4f})",
            xy=(row["idx"], row["map50"]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=7, color=GREEN, va="bottom", ha="left",
            rotation=35,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            zorder=5,
        )

    # Axes styling
    all_vals = df["map50"].values
    lo = max(0.0, float(all_vals.min()) - 0.01)
    hi = min(1.0, float(all_vals.max()) + 0.04)
    ax.set_ylim(lo, hi)
    ax.set_xlim(0, n_total + 1)

    ax.set_title(
        f"E0 Baseline Progress — {n_total} runs, {n_kept} improvements",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.set_xlabel("E0 Run #", fontsize=9)
    ax.set_ylabel("Validation mAP@0.5 (higher is better)", fontsize=9)
    ax.grid(True, color="#e8e8e8", linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    legend = ax.legend(loc="lower right", fontsize=8, frameon=True,
                       framealpha=0.9, edgecolor="#cccccc")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}  ({n_total} runs, {n_kept} improvements)")


if __name__ == "__main__":
    main()

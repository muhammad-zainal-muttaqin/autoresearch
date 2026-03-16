"""Generate e0_progress.png from E0 results CSV."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

E0_CSV = Path("e0_results") / "results.csv"
OUTPUT_PATH = Path("e0_progress.png")

PHASE_COLORS = {
    "0B": "#4e79a7",
    "0C": "#f28e2b",
    "1B": "#59a14f",
    "2": "#e15759",
}


def main() -> None:
    if not E0_CSV.exists():
        print(f"{E0_CSV} not found — no E0 runs yet.")
        return

    df = pd.read_csv(E0_CSV)
    if df.empty or "map50" not in df.columns:
        print("No completed E0 runs yet.")
        return

    df["map50"] = pd.to_numeric(df["map50"], errors="coerce").fillna(0.0)
    df["phase"] = df.get("phase", pd.Series(["?" * len(df)]))
    df["run_id"] = df.get("run_id", pd.Series(range(len(df)))).astype(str)
    df["idx"] = range(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)

    for phase, grp in df.groupby("phase"):
        color = PHASE_COLORS.get(str(phase), "#aaaaaa")
        ax.scatter(grp["idx"], grp["map50"], s=30, color=color,
                   edgecolors="white", linewidths=0.4,
                   label=f"Phase {phase}", zorder=3)

    # Best per arch annotation
    if "arch" in df.columns:
        best_per_arch = df.loc[df.groupby("arch")["map50"].idxmax()]
        for _, row in best_per_arch.iterrows():
            ax.annotate(
                f"{row['arch']}\n{row['map50']:.3f}",
                (row["idx"], row["map50"]),
                textcoords="offset points", xytext=(4, 4),
                ha="left", va="bottom", fontsize=6,
                color="#333333", alpha=0.85, zorder=4,
            )

    max_v = float(df["map50"].max()) if not df.empty else 0.6
    min_v = float(df["map50"].min()) if not df.empty else 0.0
    span = max(max_v - min_v, 0.05)

    ax.set_title(f"E0 Protocol Progress — {len(df)} runs", fontsize=12)
    ax.set_xlabel("E0 Run #", fontsize=9)
    ax.set_ylabel("Validation mAP@0.5", fontsize=9)
    ax.set_ylim(max(0, min_v - span * 0.05), min(1.0, max_v + span * 0.15))
    ax.set_xlim(0, len(df) + 1)
    ax.grid(True, color="#e9ecef", linewidth=0.8, alpha=0.8)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    tick_step = max(1, math.ceil(len(df) / 15))
    ax.set_xticks(range(0, len(df) + 2, tick_step))
    ax.tick_params(axis="both", labelsize=8)

    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved E0 plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

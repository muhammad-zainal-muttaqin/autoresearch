from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("results.tsv")
OUTPUT_PATH = Path("progress.png")
REQUIRED_COLUMNS = [
    "commit",
    "val_map50",
    "val_map50_95",
    "precision",
    "recall",
    "memory_gb",
    "status",
]


def human_label(row: pd.Series) -> str:
    label = str(row.get("title", "") or "").strip()
    if not label:
        label = str(row.get("description", "") or "").strip()
    if not label:
        label = f"{row['status']} run"

    label = label.replace("_", " ")
    return " ".join(textwrap.wrap(label, width=28)) or label


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found")

    df = pd.read_csv(RESULTS_PATH, sep="\t")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"results.tsv missing columns: {missing}")

    if df.empty:
        print("results.tsv only contains the header so far.")
        return

    if "description" not in df.columns:
        df["description"] = ""
    if "title" not in df.columns:
        df["title"] = ""

    df["status"] = df["status"].fillna("").astype(str).str.strip().str.lower()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["val_map50_95"] = pd.to_numeric(df["val_map50_95"], errors="coerce").fillna(0.0)
    df["iteration"] = range(1, len(df) + 1)

    keep_df = df[df["status"] == "keep"].copy()
    discarded_df = df[df["status"] != "keep"].copy()
    keep_df["running_best"] = keep_df["val_map50_95"].cummax()
    running_best_df = keep_df[keep_df["val_map50_95"] == keep_df["running_best"]].copy()

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

    if not discarded_df.empty:
        ax.scatter(
            discarded_df["iteration"],
            discarded_df["val_map50_95"],
            s=10,
            color="#cfcfcf",
            alpha=0.8,
            edgecolors="none",
            label="Discarded / infra",
            zorder=1,
        )

    if not keep_df.empty:
        ax.scatter(
            keep_df["iteration"],
            keep_df["val_map50_95"],
            s=22,
            color="#37b26c",
            edgecolors="#1f7a4c",
            linewidths=0.6,
            label="Kept",
            zorder=3,
        )
        ax.step(
            keep_df["iteration"],
            keep_df["running_best"],
            where="post",
            color="#37b26c",
            linewidth=1.2,
            alpha=0.9,
            label="Running best",
            zorder=2,
        )

    max_value = float(df["val_map50_95"].max())
    min_value = float(df["val_map50_95"].min())
    value_span = max(max_value - min_value, 0.02)

    last_labeled_x = -10
    for _, row in running_best_df.iterrows():
        x = int(row["iteration"])
        y = float(row["val_map50_95"])
        if x - last_labeled_x < 2 and y < max_value - 0.01:
            continue
        ax.annotate(
            human_label(row),
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            va="bottom",
            rotation=28,
            fontsize=6,
            color="#2f9e61",
            alpha=0.95,
            zorder=4,
        )
        last_labeled_x = x

    ax.set_title(
        f"Autoresearch Progress: {len(df)} Experiments, {len(keep_df)} Kept Improvements",
        fontsize=12,
    )
    ax.set_xlabel("Experiment #", fontsize=9)
    ax.set_ylabel("Validation mAP50-95 (higher is better)", fontsize=9)
    ax.grid(True, color="#e9ecef", linewidth=0.8, alpha=0.8)

    tick_step = max(1, math.ceil(len(df) / 12))
    ax.set_xticks(list(range(0, len(df) + 1, tick_step)))
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlim(-1, len(df) + 1)
    ax.set_ylim(min(-0.01, min_value - value_span * 0.03), max_value + value_span * 0.10)
    ax.legend(loc="upper right", fontsize=7, frameon=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

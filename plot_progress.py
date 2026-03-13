from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("results.tsv")
OUTPUT_PATH = Path("progress.png")
EXPECTED_COLUMNS = [
    "commit",
    "val_map50",
    "val_map50_95",
    "precision",
    "recall",
    "memory_gb",
    "status",
    "description",
]


def human_label(row: pd.Series) -> str:
    description = str(row.get("description", "") or "").strip()
    if not description:
        description = f"{row['status']} run"

    description = description.replace("_", " ")
    return "\n".join(textwrap.wrap(description, width=22)) or description


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found")

    df = pd.read_csv(RESULTS_PATH, sep="\t")
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"results.tsv missing columns: {missing}")

    if df.empty:
        print("results.tsv only contains the header so far.")
        return

    df["status"] = df["status"].fillna("").astype(str).str.strip().str.lower()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["val_map50_95"] = pd.to_numeric(df["val_map50_95"], errors="coerce").fillna(0.0)
    df["iteration"] = range(1, len(df) + 1)

    status_colors = {
        "keep": "#2e8b57",
        "discard": "#d23b3b",
        "crash": "#666666",
    }

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    ax.plot(df["iteration"], df["val_map50_95"], color="#8da2ad", linewidth=2.0, alpha=0.9, zorder=1)

    for status, group in df.groupby("status", dropna=False):
        color = status_colors.get(status, "#4b6a88")
        ax.scatter(
            group["iteration"],
            group["val_map50_95"],
            c=color,
            s=150,
            edgecolors="white",
            linewidths=1.6,
            zorder=3,
        )

    for _, row in df.iterrows():
        offset_y = 8 if row["status"] != "crash" else 6
        va = "bottom"
        if row["status"] == "crash" and row["val_map50_95"] <= 0.01:
            offset_y = 8
            va = "bottom"

        ax.annotate(
            human_label(row),
            (row["iteration"], row["val_map50_95"]),
            textcoords="offset points",
            xytext=(0, offset_y),
            ha="center",
            va=va,
            fontsize=9,
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
            },
            zorder=4,
        )

    ax.set_title("Autoresearch val mAP50-95 by Iteration", fontsize=20)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("val_map50_95", fontsize=16)
    ax.grid(True, linestyle="--", linewidth=1, alpha=0.35)
    ax.set_xticks(df["iteration"])

    max_value = float(df["val_map50_95"].max())
    margin = max(0.01, max_value * 0.06)
    ax.set_ylim(-0.01, max_value + margin)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

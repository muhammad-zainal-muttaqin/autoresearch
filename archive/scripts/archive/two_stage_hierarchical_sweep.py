"""Sweep a small grid of detector/integration settings for the hierarchical pipeline."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare import DATA_YAML
from stage2_models import load_stage2_classifier
from two_stage_hierarchical_eval import (
    BINARY_CLASSIFIER_PATH,
    COARSE_CLASSIFIER_PATH,
    DETECTOR_PATH,
    DEVICE,
    evaluate_pipeline,
)


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-path", default=str(DETECTOR_PATH))
    parser.add_argument("--coarse-classifier-path", default=str(COARSE_CLASSIFIER_PATH))
    parser.add_argument("--binary-classifier-path", default=str(BINARY_CLASSIFIER_PATH))
    parser.add_argument("--det-conf-grid", default="0.05,0.10,0.20")
    parser.add_argument("--det-iou-grid", default="0.40,0.50,0.60")
    parser.add_argument("--pad-ratio-grid", default="0.10,0.20,0.30")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    os.chdir(DATA_YAML.parent)
    print("Loading detector...")
    detector = YOLO(str(Path(args.detector_path)))
    print("Loading classifiers...")
    coarse_loaded = load_stage2_classifier(Path(args.coarse_classifier_path), DEVICE)
    binary_loaded = load_stage2_classifier(Path(args.binary_classifier_path), DEVICE)

    val_img_dir = DATA_YAML.parent / "images" / "val"
    val_lbl_dir = DATA_YAML.parent / "labels" / "val"
    det_conf_grid = parse_csv_floats(args.det_conf_grid)
    det_iou_grid = parse_csv_floats(args.det_iou_grid)
    pad_ratio_grid = parse_csv_floats(args.pad_ratio_grid)

    results = []
    for det_conf, det_iou, pad_ratio in itertools.product(det_conf_grid, det_iou_grid, pad_ratio_grid):
        print(
            f"\n=== Sweep config: det_conf={det_conf:.2f}, det_iou={det_iou:.2f}, "
            f"pad_ratio={pad_ratio:.2f} ==="
        )
        map50, map50_95, _ = evaluate_pipeline(
            detector,
            coarse_loaded,
            binary_loaded,
            val_img_dir,
            val_lbl_dir,
            detector_conf=det_conf,
            detector_iou=det_iou,
            imgsz=args.imgsz,
            pad_ratio=pad_ratio,
        )
        results.append(
            {
                "det_conf": det_conf,
                "det_iou": det_iou,
                "pad_ratio": pad_ratio,
                "map50": map50,
                "map50_95": map50_95,
            }
        )

    results.sort(key=lambda row: row["map50_95"], reverse=True)
    print("\n=== Sweep summary (best first) ===")
    for row in results:
        print(
            f"det_conf={row['det_conf']:.2f} det_iou={row['det_iou']:.2f} "
            f"pad_ratio={row['pad_ratio']:.2f} map50={row['map50']:.6f} "
            f"map50_95={row['map50_95']:.6f}"
        )
    best = results[0]
    print(
        "\nBEST: "
        f"det_conf={best['det_conf']:.2f} det_iou={best['det_iou']:.2f} "
        f"pad_ratio={best['pad_ratio']:.2f} map50={best['map50']:.6f} "
        f"map50_95={best['map50_95']:.6f}"
    )


if __name__ == "__main__":
    main()

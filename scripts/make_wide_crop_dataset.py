"""
Extract WIDER context crops from training images using GT bounding boxes.

This is the context-aware variant: PAD_RATIO=0.6 gives much more surrounding
context around each bunch, potentially helping B2/B3 classification where
surrounding foliage/tree context matters for ripeness determination.

Creates:
- Dataset-Crops-Wide/train/B1/*.jpg  (B2, B3, B4 same)
- Dataset-Crops-Wide/val/B1/*.jpg

Then makes hierarchical variants:
- Dataset-Crops-Wide-Coarse3/ (B1, B23, B4)
- Dataset-Crops-Wide-B23/    (B2, B3 only)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from PIL import Image


CLASS_NAMES = {0: "B1", 1: "B2", 2: "B3", 3: "B4"}
SRC = Path("/workspace/autoresearch/Dataset-YOLO")


def read_label(label_path: Path) -> list:
    boxes = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4])))
    return boxes


def extract_crops(split: str, dst: Path, pad_ratio: float) -> dict:
    src_img = SRC / "images" / split
    src_lbl = SRC / "labels" / split

    for cls_name in CLASS_NAMES.values():
        (dst / split / cls_name).mkdir(parents=True, exist_ok=True)

    counts = {cls_id: 0 for cls_id in CLASS_NAMES}
    n_imgs_processed = 0

    for img_path in sorted(src_img.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        lbl_path = src_lbl / (img_path.stem + ".txt")
        boxes = read_label(lbl_path)
        if not boxes:
            continue

        img = Image.open(str(img_path)).convert("RGB")
        img_w, img_h = img.size

        for i, (cls_id, cx_norm, cy_norm, w_norm, h_norm) in enumerate(boxes):
            if cls_id not in CLASS_NAMES:
                continue

            cx = cx_norm * img_w
            cy = cy_norm * img_h
            bw = w_norm * img_w
            bh = h_norm * img_h

            pad_w = bw * pad_ratio
            pad_h = bh * pad_ratio

            x1 = max(0, int(cx - bw / 2 - pad_w))
            y1 = max(0, int(cy - bh / 2 - pad_h))
            x2 = min(img_w, int(cx + bw / 2 + pad_w))
            y2 = min(img_h, int(cy + bh / 2 + pad_h))

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            crop = img.crop((x1, y1, x2, y2))
            # Resize to 224x224 for classifier
            crop = crop.resize((224, 224), Image.LANCZOS)

            cls_name = CLASS_NAMES[cls_id]
            out_path = dst / split / cls_name / f"{img_path.stem}_box{i}.jpg"
            crop.save(str(out_path), quality=90)
            counts[cls_id] += 1

        n_imgs_processed += 1

    print(f"  {split}: {n_imgs_processed} images processed")
    print(f"         B1={counts[0]}, B2={counts[1]}, B3={counts[2]}, B4={counts[3]}")
    return counts


def build_coarse3(src: Path, dst: Path) -> None:
    print(f"\nBuilding coarse3 dataset: {dst}")
    if dst.exists():
        shutil.rmtree(dst)
    for split in ("train", "val"):
        b1_src = src / split / "B1"
        b2_src = src / split / "B2"
        b3_src = src / split / "B3"
        b4_src = src / split / "B4"

        b1_dst = dst / split / "B1"
        b23_dst = dst / split / "B23"
        b4_dst = dst / split / "B4"

        b1_dst.mkdir(parents=True, exist_ok=True)
        b23_dst.mkdir(parents=True, exist_ok=True)
        b4_dst.mkdir(parents=True, exist_ok=True)

        counts = {"B1": 0, "B2": 0, "B3": 0, "B4": 0}
        for img in sorted(b1_src.glob("*")):
            (b1_dst / img.name).symlink_to(img.resolve())
            counts["B1"] += 1
        for img in sorted(b2_src.glob("*")):
            (b23_dst / f"B2__{img.name}").symlink_to(img.resolve())
            counts["B2"] += 1
        for img in sorted(b3_src.glob("*")):
            (b23_dst / f"B3__{img.name}").symlink_to(img.resolve())
            counts["B3"] += 1
        for img in sorted(b4_src.glob("*")):
            (b4_dst / img.name).symlink_to(img.resolve())
            counts["B4"] += 1
        print(f"  {split}: B1={counts['B1']}, B23={counts['B2']+counts['B3']}, B4={counts['B4']}")


def build_b23(src: Path, dst: Path) -> None:
    print(f"\nBuilding B23 dataset: {dst}")
    if dst.exists():
        shutil.rmtree(dst)
    for split in ("train", "val"):
        b2_src = src / split / "B2"
        b3_src = src / split / "B3"

        (dst / split / "B2").mkdir(parents=True, exist_ok=True)
        (dst / split / "B3").mkdir(parents=True, exist_ok=True)

        b2_count = 0
        b3_count = 0
        for img in sorted(b2_src.glob("*")):
            (dst / split / "B2" / img.name).symlink_to(img.resolve())
            b2_count += 1
        for img in sorted(b3_src.glob("*")):
            (dst / split / "B3" / img.name).symlink_to(img.resolve())
            b3_count += 1
        print(f"  {split}: B2={b2_count}, B3={b3_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pad-ratio", type=float, default=0.6,
                        help="Padding ratio around bounding box (default=0.6 for wide context)")
    parser.add_argument("--suffix", type=str, default="Wide",
                        help="Dataset name suffix (default=Wide)")
    args = parser.parse_args()

    dst = Path(f"/workspace/autoresearch/Dataset-Crops-{args.suffix}")
    print(f"Extracting crops with pad_ratio={args.pad_ratio}")
    print(f"Output directory: {dst}")

    if dst.exists():
        print(f"  Removing existing {dst}...")
        shutil.rmtree(dst)

    for split in ("train", "val"):
        print(f"\nProcessing {split}...")
        extract_crops(split, dst, args.pad_ratio)

    # Build hierarchical variants
    coarse_dst = Path(f"/workspace/autoresearch/Dataset-Crops-{args.suffix}-Coarse3")
    b23_dst = Path(f"/workspace/autoresearch/Dataset-Crops-{args.suffix}-B23")

    build_coarse3(dst, coarse_dst)
    build_b23(dst, b23_dst)

    print(f"\nDone! Created:")
    print(f"  {dst}")
    print(f"  {coarse_dst}")
    print(f"  {b23_dst}")


if __name__ == "__main__":
    main()

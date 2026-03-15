"""
Train RF-DETR (DINOv2 backbone) on palm oil bunch detection dataset.

RF-DETR uses a DINOv2 ViT backbone with deformable attention, achieving
54.7% mAP50-95 on COCO. For fine-grained B2/B3 disambiguation, the
ViT self-attention should capture global context better than CNN.

Key difference from RT-DETR: RF-DETR uses DINOv2 as backbone (much stronger
visual representations than ResNet-based RT-DETR).

Dataset format: RF-DETR expects YOLO format with "train" and "valid" subdirs.
"""

import os
import shutil
import time
import torch
from pathlib import Path
from prepare import DATA_YAML, evaluate_model, RUNS_ROOT, BEST_WEIGHTS, TRAIN_RUN_DIR

# ── Config ────────────────────────────────────────────────────────────────────
TIME_BUDGET = 0.33 * 3600  # 20 minutes
BATCH_SIZE = 4
EPOCHS = 40
LR = 1e-4
IMGSZ = 640  # RF-DETR typically trained at 640

# Source dataset
SRC_DIR = Path("/workspace/autoresearch/Dataset-YOLO")
# RF-DETR needs "valid" not "val"
RFDETR_DIR = Path("/workspace/autoresearch/Dataset-RFDETR")

RFDETR_OUTPUT = Path("/workspace/autoresearch/rfdetr_output")


def create_rfdetr_dataset():
    """Create RF-DETR compatible dataset (symlink val → valid)."""
    print("Creating RF-DETR compatible dataset structure...")

    # RF-DETR needs: train/ and valid/ with images/ and labels/ subdirs
    for split in ['train', 'valid']:
        (RFDETR_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (RFDETR_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Copy train split
    src_split = 'train'
    for img_path in sorted((SRC_DIR / 'images' / src_split).glob('*')):
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            dst = RFDETR_DIR / 'train' / 'images' / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)
    for lbl_path in sorted((SRC_DIR / 'labels' / src_split).glob('*.txt')):
        dst = RFDETR_DIR / 'train' / 'labels' / lbl_path.name
        if not dst.exists():
            shutil.copy2(lbl_path, dst)
    n_train = len(list((RFDETR_DIR / 'train' / 'images').glob('*')))
    print(f"  train: {n_train} images")

    # Copy val split as 'valid'
    src_split = 'val'
    for img_path in sorted((SRC_DIR / 'images' / src_split).glob('*')):
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            dst = RFDETR_DIR / 'valid' / 'images' / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)
    for lbl_path in sorted((SRC_DIR / 'labels' / src_split).glob('*.txt')):
        dst = RFDETR_DIR / 'valid' / 'labels' / lbl_path.name
        if not dst.exists():
            shutil.copy2(lbl_path, dst)
    n_val = len(list((RFDETR_DIR / 'valid' / 'images').glob('*')))
    print(f"  valid: {n_val} images")

    # Create data.yaml for RF-DETR (needs 'names' as list)
    yaml_content = """path: /workspace/autoresearch/Dataset-RFDETR
train: train/images
val: valid/images

nc: 4
names:
  - B1
  - B2
  - B3
  - B4
"""
    with open(RFDETR_DIR / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    print("  data.yaml created")


def train_rfdetr():
    """Train RF-DETR model."""
    from rfdetr import RFDETRBase

    print("\nInitializing RF-DETR Base (DINOv2-S backbone)...")
    model = RFDETRBase()

    RFDETR_OUTPUT.mkdir(parents=True, exist_ok=True)

    print(f"Training RF-DETR for {TIME_BUDGET/60:.0f} minutes...")
    print(f"  dataset_dir: {RFDETR_DIR}")
    print(f"  batch_size: {BATCH_SIZE}, epochs: {EPOCHS}")

    t0 = time.time()

    # RF-DETR training API
    model.train(
        dataset_dir=str(RFDETR_DIR),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        output_dir=str(RFDETR_OUTPUT),
        num_workers=4,
        early_stopping=True,
        early_stopping_patience=10,
        progress_bar=True,
    )

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.0f}s")


def evaluate_rfdetr():
    """Evaluate RF-DETR on val set and convert to standard metrics."""
    from rfdetr import RFDETRBase
    from ultralytics import YOLO
    import json

    # Load best RF-DETR model
    output_dir = RFDETR_OUTPUT
    checkpoint_files = list(output_dir.glob("*.pth"))
    if not checkpoint_files:
        print("No checkpoint found!")
        return None

    # Find best checkpoint
    best_ckpt = sorted(checkpoint_files)[-1]
    print(f"Evaluating checkpoint: {best_ckpt}")

    model = RFDETRBase(pretrain_weights=str(best_ckpt))

    # Run evaluation
    # RF-DETR evaluate method
    val_results = model.eval(
        dataset_dir=str(RFDETR_DIR),
        batch_size=BATCH_SIZE,
    )

    print(f"\nRF-DETR evaluation results:")
    print(val_results)
    return val_results


if __name__ == '__main__':
    print("=== RF-DETR Training Pipeline ===")

    # Create dataset
    create_rfdetr_dataset()

    # Train
    train_rfdetr()

    # Evaluate
    print("\nEvaluation...")
    # Evaluate using our standard pipeline for fair comparison
    # Find best weights
    best_weights = sorted(RFDETR_OUTPUT.glob("*.pth"))
    if best_weights:
        print(f"Best weights found: {best_weights[-1]}")
    else:
        print("No weights found — training may have failed")

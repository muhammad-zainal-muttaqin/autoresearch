"""
Train RF-DETR Small on palm oil bunch detection.

RF-DETR Small uses DINOv2 ViT-S backbone with deformable attention.
End-to-end detection + classification — no cascade error.

Key differences from previous RF-DETR Base run:
- RFDETRSmall (faster, less params)
- Dataset-RFDETR-TT: 3388 train images (vs 2764)
- 40 epochs, early stopping patience 15
- Gradient accumulation steps 4 (effective batch=32)
"""

import time
from pathlib import Path
from rfdetr import RFDETRSmall

RFDETR_TT_DIR = Path("/workspace/autoresearch/Dataset-RFDETR-TT")
RFDETR_OUTPUT = Path("/workspace/autoresearch/rfdetr_small_output")
EPOCHS = 40
BATCH_SIZE = 8
LR = 1e-4

print("=== RF-DETR Small Training ===")
print(f"Dataset: {RFDETR_TT_DIR}")
print(f"Train images: 3388 | Val images: 604")
print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")

t0 = time.time()

model = RFDETRSmall()

RFDETR_OUTPUT.mkdir(parents=True, exist_ok=True)

model.train(
    dataset_dir=str(RFDETR_TT_DIR),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    output_dir=str(RFDETR_OUTPUT),
    num_workers=4,
    early_stopping=True,
    early_stopping_patience=15,
    progress_bar=True,
    gradient_accumulation_steps=4,
)

elapsed = time.time() - t0
print(f"\nTraining completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

# Find best checkpoint
checkpoints = sorted(RFDETR_OUTPUT.glob("*.pth"))
if checkpoints:
    print(f"Best checkpoint: {checkpoints[-1]}")
else:
    print("No checkpoint found.")

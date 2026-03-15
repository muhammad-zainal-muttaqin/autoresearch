"""Train a DINOv2 crop classifier with CORN ordinal loss."""

from pathlib import Path
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from stage2_models import (
    CLASS_NAMES,
    N_CLASSES,
    DINOv2CORNClassifier,
    build_train_transforms,
    build_val_transforms,
    classifier_logits_to_probs,
)


CROP_DIR = Path("/workspace/autoresearch/Dataset-Crops")
SAVE_PATH = Path("/workspace/autoresearch/stage2_dinov2_corn_classifier.pth")
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-3
TIME_BUDGET = 2.0 * 3600
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINOV2_MODEL = "facebook/dinov2-base"


def compute_class_weights(dataset):
    counts = [0] * N_CLASSES
    for _, label in dataset:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (N_CLASSES * c) if c > 0 else 1.0 for c in counts]
    print(f"  Class counts: {dict(zip(CLASS_NAMES, counts))}")
    print(f"  Class weights: {[f'{w:.3f}' for w in weights]}")
    return torch.FloatTensor(weights).to(DEVICE)


def corn_loss(logits, labels, class_weights):
    loss_sum = torch.zeros((), device=labels.device)
    count = 0
    sample_weights = class_weights[labels]
    for idx in range(N_CLASSES - 1):
        if idx == 0:
            mask = torch.ones_like(labels, dtype=torch.bool)
        else:
            mask = labels > (idx - 1)
        if not torch.any(mask):
            continue
        task_targets = (labels[mask] > idx).float()
        task_loss = F.binary_cross_entropy_with_logits(
            logits[mask, idx],
            task_targets,
            reduction="none",
        )
        loss_sum = loss_sum + (task_loss * sample_weights[mask]).sum()
        count += int(mask.sum().item())
    return loss_sum / max(count, 1)


def train_epoch(model, loader, optimizer, class_weights):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = corn_loss(logits, labels, class_weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        probs = classifier_logits_to_probs("dinov2_corn", logits)
        predicted = probs.argmax(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


def val_epoch(model, loader, class_weights):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * N_CLASSES
    class_total = [0] * N_CLASSES
    confusion = torch.zeros((N_CLASSES, N_CLASSES), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = corn_loss(logits, labels, class_weights)
            probs = classifier_logits_to_probs("dinov2_corn", logits)
            predicted = probs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

            for i in range(len(labels)):
                true_cls = labels[i].item()
                pred_cls = predicted[i].item()
                class_correct[true_cls] += int(pred_cls == true_cls)
                class_total[true_cls] += 1
                confusion[true_cls, pred_cls] += 1

    per_class_acc = [
        100.0 * c / t if t > 0 else 0.0
        for c, t in zip(class_correct, class_total)
    ]
    return total_loss / total, 100.0 * correct / total, per_class_acc, confusion


def main():
    print(f"Device: {DEVICE}")
    print(f"Crop directory: {CROP_DIR}")

    train_dataset = datasets.ImageFolder(
        str(CROP_DIR / "train"),
        transform=build_train_transforms(),
    )
    val_dataset = datasets.ImageFolder(
        str(CROP_DIR / "val"),
        transform=build_val_transforms(),
    )
    assert train_dataset.classes == CLASS_NAMES, (
        f"Class mismatch: {train_dataset.classes} vs {CLASS_NAMES}"
    )

    print(f"\nDataset sizes: train={len(train_dataset)}, val={len(val_dataset)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print("\nBuilding CORN DINOv2 classifier...")
    model = DINOv2CORNClassifier(
        n_classes=N_CLASSES,
        model_name=DINOV2_MODEL,
    ).to(DEVICE)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params/1e6:.1f}M | Trainable: {trainable_params/1e6:.1f}M")

    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_dataset)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6,
    )

    best_val_acc = 0.0
    best_epoch = 0
    best_confusion = None
    best_per_class = None
    t0 = time.time()

    print(f"\nTraining for up to {NUM_EPOCHS} epochs ({TIME_BUDGET/3600:.1f}h budget)...")
    for epoch in range(NUM_EPOCHS):
        elapsed = time.time() - t0
        if elapsed > TIME_BUDGET:
            print(f"\nTime budget exceeded at epoch {epoch + 1}. Stopping.")
            break

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, class_weights)
        val_loss, val_acc, per_class_acc, confusion = val_epoch(model, val_loader, class_weights)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | "
            f"train_acc={train_acc:.1f}% | val_acc={val_acc:.1f}% | "
            f"B1={per_class_acc[0]:.1f}% B2={per_class_acc[1]:.1f}% "
            f"B3={per_class_acc[2]:.1f}% B4={per_class_acc[3]:.1f}% | "
            f"lr={lr:.2e} | {elapsed:.0f}s elapsed"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_confusion = confusion.clone()
            best_per_class = per_class_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "per_class_acc": per_class_acc,
                    "confusion_matrix": confusion.tolist(),
                    "epoch": epoch + 1,
                    "class_names": CLASS_NAMES,
                    "dinov2_model": DINOV2_MODEL,
                    "classifier_type": "dinov2_corn",
                },
                str(SAVE_PATH),
            )
            print(f"  *** Saved best model (val_acc={val_acc:.2f}%) ***")

    print(f"\nBest val_acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"CORN DINOv2 classifier saved to {SAVE_PATH}")

    if best_per_class is not None:
        print("\nBest per-class accuracy:")
        for i, cls_name in enumerate(CLASS_NAMES):
            print(f"  {cls_name}: {best_per_class[i]:.1f}%")
    if best_confusion is not None:
        print("\nBest confusion matrix (rows=true, cols=pred):")
        for i, cls_name in enumerate(CLASS_NAMES):
            row = " ".join(f"{int(v):4d}" for v in best_confusion[i].tolist())
            print(f"  {cls_name}: {row}")

    return best_val_acc


if __name__ == "__main__":
    main()

"""Train a frozen-backbone DINOv2 classifier with CE + supervised contrastive loss."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from stage2_models import DINOv2SupConClassifier, build_train_transforms, build_val_transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="ImageFolder dataset root with train/ and val/")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--time-hours", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-name", default="facebook/dinov2-base")
    parser.add_argument("--supcon-weight", type=float, default=0.15)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--supcon-start-epoch", type=int, default=3)
    return parser.parse_args()


def compute_class_weights(dataset: datasets.ImageFolder, n_classes: int) -> torch.Tensor:
    counts = [0] * n_classes
    for _, label in dataset.samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (n_classes * c) if c > 0 else 1.0 for c in counts]
    print(f"  Class counts: {dict(zip(dataset.classes, counts))}")
    print(f"  Class weights: {[f'{w:.3f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T)
    logits_mask = torch.ones_like(mask, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)
    mask = mask & logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_counts = mask.sum(dim=1)
    valid = positive_counts > 0
    if not torch.any(valid):
        return torch.zeros((), device=features.device)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


def train_epoch(model, loader, ce_criterion, optimizer, supcon_weight, supcon_temperature):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_supcon = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits, proj = model.forward_with_projection(images)
        ce_loss = ce_criterion(logits, labels)
        supcon_loss = supervised_contrastive_loss(proj, labels, temperature=supcon_temperature)
        loss = ce_loss + supcon_weight * supcon_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_ce += ce_loss.item() * images.size(0)
        total_supcon += supcon_loss.item() * images.size(0)
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        total += images.size(0)
    denom = max(total, 1)
    return total_loss / denom, total_ce / denom, total_supcon / denom, 100.0 * correct / denom


def val_epoch(model, loader, criterion, n_classes: int):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * n_classes
    class_total = [0] * n_classes
    confusion = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)
            for idx in range(len(labels)):
                true_cls = labels[idx].item()
                pred_cls = preds[idx].item()
                class_correct[true_cls] += int(pred_cls == true_cls)
                class_total[true_cls] += 1
                confusion[true_cls, pred_cls] += 1

    per_class_acc = [
        100.0 * c / t if t > 0 else 0.0
        for c, t in zip(class_correct, class_total)
    ]
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1), per_class_acc, confusion


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    save_path = Path(args.save_path)
    time_budget = args.time_hours * 3600

    print(f"Device: {DEVICE}")
    print(f"Dataset root: {dataset_root}")

    train_dataset = datasets.ImageFolder(str(dataset_root / "train"), transform=build_train_transforms())
    val_dataset = datasets.ImageFolder(str(dataset_root / "val"), transform=build_val_transforms())
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(f"Train/val class mismatch: {train_dataset.classes} vs {val_dataset.classes}")
    n_classes = len(train_dataset.classes)

    print(f"\nDataset sizes: train={len(train_dataset)}, val={len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("\nBuilding DINOv2 SupCon classifier...")
    model = DINOv2SupConClassifier(n_classes=n_classes, model_name=args.model_name).to(DEVICE)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params/1e6:.1f}M | Trainable: {trainable_params/1e6:.1f}M")

    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_dataset, n_classes)
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_epoch = 0
    best_confusion = None
    best_per_class = None
    t0 = time.time()

    print(f"\nTraining for up to {args.epochs} epochs ({args.time_hours:.1f}h budget)...")
    for epoch in range(args.epochs):
        elapsed = time.time() - t0
        if elapsed > time_budget:
            print(f"\nTime budget exceeded at epoch {epoch + 1}. Stopping.")
            break

        supcon_weight = args.supcon_weight if (epoch + 1) >= args.supcon_start_epoch else 0.0
        train_loss, train_ce, train_supcon, train_acc = train_epoch(
            model,
            train_loader,
            ce_criterion,
            optimizer,
            supcon_weight=supcon_weight,
            supcon_temperature=args.supcon_temperature,
        )
        val_loss, val_acc, per_class_acc, confusion = val_epoch(model, val_loader, ce_criterion, n_classes)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        per_class_str = " ".join(
            f"{name}={acc:.1f}%"
            for name, acc in zip(train_dataset.classes, per_class_acc)
        )
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | ce={train_ce:.4f} | supcon={train_supcon:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.1f}% | "
            f"{per_class_str} | supcon_w={supcon_weight:.2f} | lr={lr:.2e} | {elapsed:.0f}s elapsed"
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
                    "class_names": train_dataset.classes,
                    "n_classes": n_classes,
                    "dinov2_model": args.model_name,
                    "classifier_type": "dinov2_supcon",
                    "dataset_root": str(dataset_root),
                    "supcon_weight": args.supcon_weight,
                    "supcon_temperature": args.supcon_temperature,
                },
                str(save_path),
            )
            print(f"  *** Saved best model (val_acc={val_acc:.2f}%) ***")

    print(f"\nBest val_acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"SupCon DINOv2 classifier saved to {save_path}")
    if best_per_class is not None:
        print("\nBest per-class accuracy:")
        for cls_name, acc in zip(train_dataset.classes, best_per_class):
            print(f"  {cls_name}: {acc:.1f}%")
    if best_confusion is not None:
        print("\nBest confusion matrix (rows=true, cols=pred):")
        for idx, cls_name in enumerate(train_dataset.classes):
            row = " ".join(f"{int(v):4d}" for v in best_confusion[idx].tolist())
            print(f"  {cls_name}: {row}")


if __name__ == "__main__":
    main()

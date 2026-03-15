"""
Train DINOv2 classifier with PARTIAL backbone fine-tuning.

Unfreezes the last N transformer blocks of DINOv2 to enable domain adaptation
while keeping early layers frozen (which capture general visual features).

This is different from all prior work which used fully frozen backbones.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2_models import build_train_transforms, build_val_transforms, ensure_transformers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--time-hours", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="LR for unfrozen backbone layers")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="LR for classification head")
    parser.add_argument("--unfreeze-last-n", type=int, default=4,
                        help="Number of last transformer blocks to unfreeze (default=4)")
    parser.add_argument("--model-name", default="facebook/dinov2-base")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


class DINOv2FineTuneClassifier(nn.Module):
    """DINOv2 with configurable number of unfrozen blocks + MLP head."""

    def __init__(self, n_classes: int, model_name: str, unfreeze_last_n: int):
        super().__init__()
        transformers = ensure_transformers()
        self.backbone = transformers.AutoModel.from_pretrained(model_name)

        # Freeze all backbone params first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last N transformer blocks (encoder layers)
        n_total = len(self.backbone.encoder.layer)
        n_unfreeze = min(unfreeze_last_n, n_total)
        for i in range(n_total - n_unfreeze, n_total):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True

        # Always unfreeze the final layernorm
        if hasattr(self.backbone, "layernorm"):
            for param in self.backbone.layernorm.parameters():
                param.requires_grad = True

        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

        n_trainable_backbone = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        n_trainable_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total_params = sum(p.numel() for p in self.parameters())
        print(f"  DINOv2 backbone: {n_total} blocks, {n_unfreeze} unfrozen")
        print(f"  Backbone trainable: {n_trainable_backbone/1e6:.2f}M")
        print(f"  Total trainable: {n_trainable_total/1e6:.2f}M / {n_total_params/1e6:.1f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_embedding)

    def get_param_groups(self, lr_backbone: float, lr_head: float) -> list:
        """Split params into backbone (low lr) and head (high lr) groups."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
            {"params": head_params, "lr": lr_head, "weight_decay": 1e-4},
        ]


def compute_class_weights(dataset: datasets.ImageFolder, n_classes: int) -> torch.Tensor:
    counts = [0] * n_classes
    for _, label in dataset.samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (n_classes * c) if c > 0 else 1.0 for c in counts]
    print(f"  Class counts: {dict(zip(dataset.classes, counts))}")
    print(f"  Class weights: {[f'{w:.3f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


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
    print(f"Dataset: {dataset_root}")
    print(f"Unfreeze last {args.unfreeze_last_n} blocks | LR backbone={args.lr_backbone} | LR head={args.lr_head}")

    train_dataset = datasets.ImageFolder(str(dataset_root / "train"), transform=build_train_transforms())
    val_dataset = datasets.ImageFolder(str(dataset_root / "val"), transform=build_val_transforms())
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(f"Class mismatch: {train_dataset.classes} vs {val_dataset.classes}")
    n_classes = len(train_dataset.classes)
    print(f"Classes: {train_dataset.classes} | train={len(train_dataset)}, val={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print("\nBuilding DINOv2 fine-tune classifier...")
    model = DINOv2FineTuneClassifier(n_classes, args.model_name, args.unfreeze_last_n).to(DEVICE)

    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_dataset, n_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    param_groups = model.get_param_groups(args.lr_backbone, args.lr_head)
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    best_val_acc = 0.0
    best_epoch = 0
    best_per_class = None
    best_confusion = None
    t0 = time.time()

    print(f"\nTraining for up to {args.epochs} epochs ({args.time_hours:.1f}h budget)...")
    for epoch in range(args.epochs):
        elapsed = time.time() - t0
        if elapsed > time_budget:
            print(f"\nTime budget exceeded at epoch {epoch + 1}. Stopping.")
            break

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, per_class_acc, confusion = val_epoch(model, val_loader, criterion, n_classes)
        scheduler.step()
        lr_head = scheduler.get_last_lr()[0]  # approx
        per_class_str = " ".join(
            f"{name}={acc:.1f}%"
            for name, acc in zip(train_dataset.classes, per_class_acc)
        )
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.1f}% | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.1f}% | "
            f"{per_class_str} | lr_head={lr_head:.2e} | {elapsed:.0f}s elapsed"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_per_class = per_class_acc
            best_confusion = confusion.clone()
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
                    "classifier_type": "dinov2_ce",
                    "dataset_root": str(dataset_root),
                    "unfreeze_last_n": args.unfreeze_last_n,
                    "lr_backbone": args.lr_backbone,
                    "lr_head": args.lr_head,
                },
                str(save_path),
            )
            print(f"  *** Saved best model (val_acc={val_acc:.2f}%) ***")

    print(f"\nBest val_acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"Saved to: {save_path}")
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

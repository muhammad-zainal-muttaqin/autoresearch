"""
Train a DINOv2-based crop classifier for two-stage pipeline stage 2.

Replaces EfficientNet-B0 (62.7% val acc) with DINOv2-large as frozen backbone
and a linear head trained on top.

DINOv2 has much stronger visual representations for fine-grained discrimination.
Expected: >80% val accuracy vs EfficientNet's 62.7%.

Usage: uv run train_dinov2_classifier.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

CROP_DIR = Path("/workspace/autoresearch/Dataset-Crops")
SAVE_PATH = Path("/workspace/autoresearch/stage2_dinov2_classifier.pth")
CLASS_NAMES = ["B1", "B2", "B3", "B4"]
N_CLASSES = 4
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-3
TIME_BUDGET = 2.0 * 3600  # 2 hours (can be shortened)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DINOv2 model variant: dinov2-base (~86M params) or dinov2-large (~307M params)
# Use base for speed, large for accuracy
DINOV2_MODEL = "facebook/dinov2-base"  # swap to dinov2-large for +accuracy


class DINOv2Classifier(nn.Module):
    """DINOv2 frozen backbone + trainable classification head."""

    def __init__(self, n_classes=4, model_name=DINOV2_MODEL):
        super().__init__()
        from transformers import AutoModel
        print(f"  Loading DINOv2 backbone: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size  # 768 for base, 1024 for large
        print(f"  DINOv2 hidden size: {hidden_size}")

        # Two-layer MLP head (more expressive than linear for fine-grained task)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        # DINOv2 returns BaseModelOutputWithPooling
        outputs = self.backbone(pixel_values=x)
        # Use CLS token embedding (index 0 of last hidden state)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_embedding)


def get_transforms():
    """DINOv2 expects 224x224 images normalized with ImageNet stats."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def compute_class_weights(dataset):
    """Compute inverse-frequency class weights."""
    counts = [0] * N_CLASSES
    for _, label in dataset:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (N_CLASSES * c) if c > 0 else 1.0 for c in counts]
    print(f"  Class counts: {dict(zip(CLASS_NAMES, counts))}")
    print(f"  Class weights: {[f'{w:.3f}' for w in weights]}")
    return torch.FloatTensor(weights).to(DEVICE)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    # Only head is trainable; backbone is frozen
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * N_CLASSES
    class_total = [0] * N_CLASSES
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)
            for i in range(len(labels)):
                cls = labels[i].item()
                class_correct[cls] += predicted[i].eq(labels[i]).item()
                class_total[cls] += 1
    per_class_acc = [100.0 * c / t if t > 0 else 0.0
                     for c, t in zip(class_correct, class_total)]
    return total_loss / total, 100.0 * correct / total, per_class_acc


def main():
    print(f"Device: {DEVICE}")
    print(f"Crop directory: {CROP_DIR}")

    # Install transformers if not available
    try:
        import transformers
        print(f"transformers version: {transformers.__version__}")
    except ImportError:
        print("Installing transformers...")
        os.system("pip install transformers -q")
        import transformers

    train_tf, val_tf = get_transforms()
    train_dataset = datasets.ImageFolder(str(CROP_DIR / 'train'), transform=train_tf)
    val_dataset = datasets.ImageFolder(str(CROP_DIR / 'val'), transform=val_tf)

    assert train_dataset.classes == CLASS_NAMES, \
        f"Class mismatch: {train_dataset.classes} vs {CLASS_NAMES}"

    print(f"\nDataset sizes: train={len(train_dataset)}, val={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    print("\nBuilding DINOv2 classifier...")
    model = DINOv2Classifier(n_classes=N_CLASSES).to(DEVICE)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params/1e6:.1f}M | Trainable: {trainable_params/1e6:.1f}M")

    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Only optimize the head (backbone is frozen)
    optimizer = optim.AdamW(model.head.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    best_epoch = 0
    history = []
    t0 = time.time()

    print(f"\nTraining for up to {NUM_EPOCHS} epochs ({TIME_BUDGET/3600:.1f}h budget)...")
    for epoch in range(NUM_EPOCHS):
        elapsed = time.time() - t0
        if elapsed > TIME_BUDGET:
            print(f"\nTime budget exceeded at epoch {epoch+1}. Stopping.")
            break

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, per_class_acc = val_epoch(model, val_loader, criterion)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        history.append({'epoch': epoch + 1, 'train_acc': train_acc, 'val_acc': val_acc,
                        'per_class': per_class_acc})

        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"train_acc={train_acc:.1f}% | val_acc={val_acc:.1f}% | "
              f"B1={per_class_acc[0]:.1f}% B2={per_class_acc[1]:.1f}% "
              f"B3={per_class_acc[2]:.1f}% B4={per_class_acc[3]:.1f}% | "
              f"lr={lr:.2e} | {elapsed:.0f}s elapsed")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'per_class_acc': per_class_acc,
                'epoch': epoch + 1,
                'class_names': CLASS_NAMES,
                'dinov2_model': DINOV2_MODEL,
            }, str(SAVE_PATH))
            print(f"  *** Saved best model (val_acc={val_acc:.2f}%) ***")

    print(f"\nBest val_acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"DINOv2 classifier saved to {SAVE_PATH}")

    if history:
        best_record = max(history, key=lambda x: x['val_acc'])
        print(f"\nBest per-class accuracy:")
        for i, cls_name in enumerate(CLASS_NAMES):
            print(f"  {cls_name}: {best_record['per_class'][i]:.1f}%")

    return best_val_acc


if __name__ == '__main__':
    main()

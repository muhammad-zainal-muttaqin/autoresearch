"""
Train a crop classifier for two-stage pipeline stage 2.

Uses EfficientNet-B0 (torchvision) on GT crops from make_crop_dataset.py.
The classifier takes detected crop → predicts B1/B2/B3/B4.

Training: 20 minutes budget (same as detector)
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import json

CROP_DIR = Path("/workspace/autoresearch/Dataset-Crops")
SAVE_PATH = Path("/workspace/autoresearch/stage2_classifier.pth")
CLASS_NAMES = ["B1", "B2", "B3", "B4"]
N_CLASSES = 4
BATCH_SIZE = 64
NUM_EPOCHS = 50
LR = 1e-3
TIME_BUDGET = 0.33 * 3600  # 20 minutes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
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


def build_model():
    """Build EfficientNet-B0 with custom head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Replace final classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, N_CLASSES),
    )
    return model


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

    return total_loss / total, 100. * correct / total


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

    per_class_acc = [100. * c / t if t > 0 else 0.0
                     for c, t in zip(class_correct, class_total)]
    return total_loss / total, 100. * correct / total, per_class_acc


def main():
    print(f"Device: {DEVICE}")
    print(f"Crop directory: {CROP_DIR}")

    # Data
    train_tf, val_tf = get_transforms()
    train_dataset = datasets.ImageFolder(str(CROP_DIR / 'train'), transform=train_tf)
    val_dataset = datasets.ImageFolder(str(CROP_DIR / 'val'), transform=val_tf)

    # Verify class mapping
    assert train_dataset.classes == CLASS_NAMES, f"Class mismatch: {train_dataset.classes} vs {CLASS_NAMES}"

    print(f"\nDataset sizes: train={len(train_dataset)}, val={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = build_model().to(DEVICE)
    print(f"Model: EfficientNet-B0, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Class-weighted loss to handle B3 dominance
    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with cosine LR
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    best_val_acc = 0
    best_epoch = 0
    history = []
    t0 = time.time()

    print(f"\nTraining for up to {NUM_EPOCHS} epochs ({TIME_BUDGET/60:.0f} min budget)...")
    for epoch in range(NUM_EPOCHS):
        # Check time budget
        elapsed = time.time() - t0
        if elapsed > TIME_BUDGET:
            print(f"\nTime budget exceeded at epoch {epoch+1}. Stopping.")
            break

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, per_class_acc = val_epoch(model, val_loader, criterion)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        history.append({'epoch': epoch+1, 'train_acc': train_acc, 'val_acc': val_acc,
                        'per_class': per_class_acc})

        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"train_acc={train_acc:.1f}% | val_acc={val_acc:.1f}% | "
              f"B1={per_class_acc[0]:.1f}% B2={per_class_acc[1]:.1f}% "
              f"B3={per_class_acc[2]:.1f}% B4={per_class_acc[3]:.1f}% | "
              f"lr={lr:.2e} | {elapsed:.0f}s elapsed")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({'model_state_dict': model.state_dict(),
                        'val_acc': val_acc,
                        'per_class_acc': per_class_acc,
                        'epoch': epoch + 1,
                        'class_names': CLASS_NAMES}, str(SAVE_PATH))
            print(f"  *** Saved best model (val_acc={val_acc:.2f}%) ***")

    print(f"\nBest val_acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"Classifier saved to {SAVE_PATH}")

    # Print final per-class accuracy
    if history:
        best_record = max(history, key=lambda x: x['val_acc'])
        print(f"\nBest per-class accuracy:")
        for i, cls in enumerate(CLASS_NAMES):
            print(f"  {cls}: {best_record['per_class'][i]:.1f}%")

    return best_val_acc


if __name__ == '__main__':
    main()

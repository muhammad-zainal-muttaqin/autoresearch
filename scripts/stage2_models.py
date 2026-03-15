"""
Reusable stage-2 classifier models for the two-stage pipeline.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models, transforms


CLASS_NAMES = ["B1", "B2", "B3", "B4"]
N_CLASSES = len(CLASS_NAMES)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_transformers():
    try:
        import transformers
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "transformers", "-q"],
            check=True,
        )
        import transformers
    return transformers


def build_train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_val_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class EfficientNetB0Classifier(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DINOv2Classifier(nn.Module):
    """Frozen DINOv2 backbone with a trainable MLP head."""

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        model_name: str = "facebook/dinov2-base",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        transformers = ensure_transformers()
        self.backbone = transformers.AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_embedding)


class CORALOrdinalHead(nn.Module):
    """Shared-weight ordinal head with K-1 thresholds."""

    def __init__(self, in_features: int, num_classes: int = N_CLASSES):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, 1))
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class DINOv2OrdinalClassifier(nn.Module):
    """Frozen DINOv2 backbone with ordinal CORAL-style head."""

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        model_name: str = "facebook/dinov2-base",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        transformers = ensure_transformers()
        self.backbone = transformers.AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.ordinal_head = CORALOrdinalHead(512, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        features = self.proj(cls_embedding)
        return self.ordinal_head(features)


def coral_levels_from_labels(labels: torch.Tensor, num_classes: int = N_CLASSES) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def ordinal_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    survival = torch.sigmoid(logits)
    survival = torch.cummin(survival, dim=1).values
    p0 = 1.0 - survival[:, :1]
    middle = survival[:, :-1] - survival[:, 1:]
    plast = survival[:, -1:]
    probs = torch.cat([p0, middle, plast], dim=1)
    probs = probs.clamp_min(0.0)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


def classifier_logits_to_probs(classifier_type: str, logits: torch.Tensor) -> torch.Tensor:
    if classifier_type == "dinov2_coral":
        return ordinal_logits_to_probs(logits)
    return torch.softmax(logits, dim=1)


def infer_classifier_type(checkpoint: dict) -> str:
    classifier_type = checkpoint.get("classifier_type")
    if classifier_type:
        return str(classifier_type)

    model_state = checkpoint.get("model_state_dict", {})
    if any(key.startswith("backbone.") for key in model_state):
        return "dinov2_ce"
    return "efficientnet_b0"


def adapt_state_dict_for_model(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model_keys = set(model.state_dict().keys())
    state_keys = set(state_dict.keys())

    if state_keys == model_keys:
        return state_dict

    if any(key.startswith("model.") for key in model_keys) and not any(
        key.startswith("model.") for key in state_keys
    ):
        return {f"model.{key}": value for key, value in state_dict.items()}

    if not any(key.startswith("model.") for key in model_keys) and any(
        key.startswith("model.") for key in state_keys
    ):
        return {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
        }

    return state_dict


@dataclass
class LoadedClassifier:
    model: nn.Module
    classifier_type: str
    checkpoint: dict


def load_stage2_classifier(checkpoint_path, device: torch.device) -> LoadedClassifier:
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    classifier_type = infer_classifier_type(checkpoint)
    model_name = checkpoint.get("dinov2_model", "facebook/dinov2-base")

    if classifier_type == "efficientnet_b0":
        model = EfficientNetB0Classifier()
    elif classifier_type == "dinov2_coral":
        model = DINOv2OrdinalClassifier(model_name=model_name)
    elif classifier_type == "dinov2_ce":
        model = DINOv2Classifier(model_name=model_name)
    else:
        raise ValueError(f"Unsupported classifier_type: {classifier_type}")

    state_dict = adapt_state_dict_for_model(model, checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return LoadedClassifier(model=model, classifier_type=classifier_type, checkpoint=checkpoint)

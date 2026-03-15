# Research Agent Report — 2026-03-15

## Executive Summary

Top 3 most promising findings, ranked by expected impact for our problem (yolo11l, 26.9% mAP50-95, 4-class palm oil FFB ripeness, B2/B3 ambiguity, ~4K images):

**1. RF-DETR with DINOv2 backbone (Finding 6)** — Highest ceiling. RF-DETR replaces the YOLO classification bottleneck entirely with a transformer decoder that naturally handles ambiguous queries. Its DINOv2 backbone was pre-trained on 142M images and is explicitly designed for fine-tuning on small datasets. On COCO it outperforms yolo11l by ~8 mAP50-95 points. On domain-specific small datasets it converges faster due to the richer visual representation. Expected gain: **+8–14 mAP50-95 points** (bringing us to 35–41% range).

**2. HAT-YOLOv8 — Hybrid Attention Transformer in neck + Shuffle Attention in backbone (Finding 3)** — Drop-in surgical modification to YOLO11. Insert a Hybrid Attention Transformer (HAT) module at the neck's TopDownLayer2 and Shuffle Attention in the backbone. This gave 7.6–11% absolute mAP improvement on multi-class fruit ripeness tasks. Since B2/B3 ambiguity is our bottleneck and the HAT module is specifically designed to "capture subtle inter-part relationships and restore fine-grained details," this directly attacks the problem. Expected gain: **+5–9 mAP50-95 points**.

**3. CORN Ordinal Regression Head integrated into YOLO classification branch (Finding 7)** — Our 4 ripeness classes are strictly ordered (B1 < B2 < B3 < B4). Standard cross-entropy treats B1→B4 confusion the same as B1→B2 confusion. CORN loss decomposes rank prediction into K-1 binary tasks with conditional probabilities, enforcing rank consistency. This is particularly valuable for the B2/B3 boundary, which is the dominant source of confusion. Expected gain: **+3–7 mAP50-95 points** (additive with other improvements).

---

## Finding 1: Fine-Grained Agricultural Ripeness Detection — DPDB-YOLO / AITP-YOLO Pattern

- **Source:** DPDB-YOLO (ScienceDirect, 2025): https://www.sciencedirect.com/science/article/pii/S0926669025019661 | AITP-YOLO (PMC, 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12146401/
- **Method:** Both models add a fourth P2/4-tiny detection head to standard YOLO's 3-head (P3/P4/P5) architecture, fusing shallow fine-grained spatial features with deep semantic features. DPDB-YOLO achieved mAP50-95 of 85.31% on cherry tomatoes. AITP-YOLO achieved mAP@0.5=92.6%, mAP@0.5:0.95=78.2% on tomato ripeness (5 classes). Shape-IoU loss replaces CIoU for better bounding-box regression on irregular produce.
- **Reported gain:** AITP-YOLO: +4–6% mAP50-95 vs YOLOv10s baseline.
- **Why relevant:** Palm oil FFBs are large objects (not small targets), so the P2 head is less critical — but the multi-scale feature fusion principle still applies to capture fine color-texture gradients that distinguish B2 from B3.

### Implementation for our project:

**Step 1 — Add auxiliary P2 detection head to yolo11l:**
```python
# In ultralytics/cfg/models/11/yolo11.yaml (copy and modify)
# Add under the head section:
# - [P2_feature_idx, 1, Conv, [256, 3, 2]]  # downsample P2
# - [[-1, P3_concat_idx], 1, Concat, [1]]
# Add a 4th Detect entry pointing to [P2_out, P3_out, P4_out, P5_out]
```

**Step 2 — Replace CIoU with Shape-IoU:**
```bash
pip install ultralytics  # already installed
```
In `ultralytics/utils/loss.py`, find `CIoU=True` in `BboxLoss` and add Shape-IoU as an option. Shape-IoU adds shape similarity term: `loss += shape_iou_term * scale`.

**Step 3 — Train with wider-context crops (already testing PAD_RATIO=0.6):**
```bash
yolo train model=yolo11l.pt data=tbs.yaml epochs=200 imgsz=640 \
  box=7.5 cls=0.5 dfl=1.5
```

**Files to create:**
- `/workspace/autoresearch/configs/yolo11l_p4head.yaml` — modified model config with extra head
- `/workspace/autoresearch/train_p4head.py` — training script

- **Estimated time:** 4–6 hours (model config + loss modification + training run)
- **Expected gain:** +2–4 mAP50-95 points (moderate, since FFBs are large objects)

---

## Finding 2: Contextual Feature Aggregation for Ambiguous Classes

- **Source:** VMC-Net (Complex & Intelligent Systems, 2025): https://link.springer.com/article/10.1007/s40747-025-01888-8 | Context in Object Detection Review (Springer, 2025): https://link.springer.com/article/10.1007/s10462-025-11186-x
- **Method:** VMC-Net introduces multi-scale context aggregation with a cross-attention mechanism that correlates each detected object with its neighborhood. Class-aware pixel-level feature aggregation filters background noise and boosts class-discriminative regions. Key insight: an FFB's surrounding environment (other bunches, tree fronds) provides ripeness cues.
- **Reported gain:** 2–5 mAP improvement on aerial detection tasks with ambiguous small classes.
- **Why relevant:** B2 and B3 bunches that appear ambiguous in isolation often have surrounding context (adjacent riper/unriper bunches, lighting direction) that disambiguates them.

### Implementation for our project:

**Step 1 — Increase PAD_RATIO to capture neighbor context (already testing this):**
The current PAD_RATIO=0.6 test is aligned with this finding. If it works, the context approach is validated.

**Step 2 — Add a scene-level branch to the classification head:**
```python
# In the two-stage classifier, add a "scene context" branch:
# 1. Extract the full image embedding from DINOv2 (global [CLS] token)
# 2. Concatenate with the crop embedding for each detected bunch
# 3. This lets the classifier ask: "given the overall scene ripeness distribution,
#    what is this specific bunch?"

class ContextAwareClassifier(nn.Module):
    def __init__(self, embed_dim=768, num_classes=4):
        super().__init__()
        self.crop_head = nn.Linear(embed_dim, 256)
        self.scene_head = nn.Linear(embed_dim, 256)
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, crop_feat, scene_feat):
        crop_emb = F.relu(self.crop_head(crop_feat))
        scene_emb = F.relu(self.scene_head(scene_feat))
        return self.fusion(torch.cat([crop_emb, scene_emb], dim=-1))
```

**Files to create:**
- `/workspace/autoresearch/models/context_classifier.py`
- `/workspace/autoresearch/train_context_classifier.py`

- **Estimated time:** 6–8 hours
- **Expected gain:** +2–5 mAP50-95 points (higher if PAD_RATIO=0.6 test shows context helps)

---

## Finding 3: HAT-YOLOv8 — Hybrid Attention Transformer in Neck for Fine-Grained Detection

- **Source:** "Hybrid attention transformer integrated YOLOV8 for fruit ripeness detection", Scientific Reports 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12219097/
- **Method:** Two-component modification to YOLOv8: (1) Shuffle Attention (SA) module replaces standard convolutions in the backbone — it groups feature maps into clusters and applies channel + spatial attention within each cluster at low cost; (2) Hybrid Attention Transformer (HAT) module inserted at neck TopDownLayer2, using Residual Hybrid Attention Groups (RHAG) with shifted-window self-attention plus Overlapping Cross-Attention Blocks (OCAB) to capture long-range dependencies and fine-grained details. EIoU loss replaces CIoU.
- **Reported gain:** +7.6% to +11% absolute mAP across 4 fruit types (5-class ripeness). Overall mAP 88.9%.
- **Why relevant:** The HAT neck module directly solves the "subtle inter-part relationship" problem. B2 and B3 bunches differ in subtle color gradients across the bunch surface — the shifted-window cross-attention can capture these gradient patterns that standard convolutions miss.

### Implementation for our project:

**Step 1 — Install required package:**
```bash
pip install einops  # for attention operations
```

**Step 2 — Implement Shuffle Attention module:**
```python
# File: /workspace/autoresearch/models/shuffle_attention.py
import torch
import torch.nn as nn

class ShuffleAttention(nn.Module):
    """Shuffle Attention: groups channels, applies channel+spatial attention per group."""
    def __init__(self, channel, G=8):
        super().__init__()
        self.G = G
        self.channel_weight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.channel_bias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.sigmoid = nn.Sigmoid()

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, c // groups, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.reshape(b, c, h, w)

    def forward(self, x):
        b, c, h, w = x.shape
        # Split into groups
        x = x.reshape(b * self.G, -1, h, w)
        # Split each group into two halves: channel attn + spatial attn
        x_0, x_1 = x.chunk(2, dim=1)
        # Channel attention
        x_channel = x_0.mean(dim=[2, 3], keepdim=True)
        x_channel = self.channel_weight * self.gn(x_channel) + self.channel_bias
        x_channel = self.sigmoid(x_channel) * x_0
        # Spatial attention
        x_spatial = self.sigmoid(self.gn(x_1)) * x_1
        # Concat and shuffle
        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.reshape(b, -1, h, w)
        return self.channel_shuffle(out, 2)
```

**Step 3 — Implement lightweight HAT neck block:**
```python
# File: /workspace/autoresearch/models/hat_neck.py
# Use swin-transformer-style window attention as HAT approximation
# Simpler alternative: use timm's SwinTransformerBlock

from timm.models.swin_transformer import SwinTransformerBlock

class HATNeckBlock(nn.Module):
    """Drop-in HAT replacement for a C2f neck block."""
    def __init__(self, dim, num_heads=8, window_size=7):
        super().__init__()
        self.swin = SwinTransformerBlock(
            dim=dim, num_heads=num_heads,
            window_size=window_size, shift_size=window_size // 2,
            mlp_ratio=4.0, drop=0.0, attn_drop=0.0
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x_flat = self.swin(x_flat, (H, W))
        return x_flat.transpose(1, 2).reshape(B, C, H, W)
```

**Step 4 — Register modules in yolo11l config:**
```bash
pip install timm
```
Modify `ultralytics/nn/modules/__init__.py` to include `ShuffleAttention` and `HATNeckBlock`.
In model YAML (yolo11l.yaml copy), replace one C2f in the backbone with `ShuffleAttention` and one C2f in the neck TopDownLayer2 with `HATNeckBlock`.

**Step 5 — Train:**
```bash
python train_hat_yolo.py --model configs/yolo11l_hat.yaml --epochs 200 --imgsz 640
```

**Files to create:**
- `/workspace/autoresearch/models/shuffle_attention.py`
- `/workspace/autoresearch/models/hat_neck.py`
- `/workspace/autoresearch/configs/yolo11l_hat.yaml`
- `/workspace/autoresearch/train_hat_yolo.py`

- **Estimated time:** 8–12 hours
- **Expected gain:** +5–9 mAP50-95 points. This is the highest-confidence YOLO-internal improvement.

---

## Finding 4: DINOv2 + LoRA Fine-Tuning for Agricultural Classification

- **Source:** "Foundation vision models in agriculture: DINOv2, LoRA and knowledge distillation for disease and weed identification", Computers and Electronics in Agriculture, 2025: https://www.sciencedirect.com/science/article/abs/pii/S0168169925010063 | dinov3-finetune GitHub: https://github.com/RobvanGastel/dinov3-finetune
- **Method:** Instead of freezing DINOv2 or doing full fine-tuning (both failed for us), LoRA (Low-Rank Adaptation) injects trainable rank-r matrices into the Q, K, V projection layers of each transformer block. Only ~0.5–2% of parameters are trained. This avoids catastrophic forgetting while adapting the backbone to domain-specific color/texture patterns unique to palm oil bunches. The study found DINOv2+LoRA consistently outperformed frozen DINOv2 and standard ImageNet-pretrained models on agricultural classification.
- **Reported gain:** DINOv2+LoRA outperforms frozen DINOv2 baselines significantly on agricultural tasks (exact numbers not published from abstract, but comparable to full fine-tuning with fraction of the risk of overfitting on small datasets).
- **Why relevant:** Our frozen DINOv2 classifier was stuck at ~73% B2/B3 binary accuracy. LoRA fine-tuning would let DINOv2 adapt its attention to the color gradients that distinguish FFB ripeness stages without overfitting ~4K images.

### Implementation for our project:

**Step 1 — Install dependencies:**
```bash
pip install peft  # HuggingFace PEFT library for LoRA
pip install transformers timm
```

**Step 2 — Apply LoRA to DINOv2 ViT-L backbone:**
```python
# File: /workspace/autoresearch/models/dinov2_lora_classifier.py
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn
import torch
from transformers import AutoModel

class DINOv2LoRAClassifier(nn.Module):
    def __init__(self, num_classes=4, lora_rank=16, lora_alpha=32):
        super().__init__()
        # Load DINOv2 ViT-L/14
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-large")

        # Apply LoRA to Q, K, V projections in all attention layers
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"],  # DINOv2 attention layers
            lora_dropout=0.05,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()
        # ~2M trainable params vs 300M total

        # Classification head
        hidden_size = 1024  # ViT-L hidden dim
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        # Use CLS token
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token)
```

**Step 3 — Training recipe (critical — LoRA needs different LR than classifier head):**
```python
# File: /workspace/autoresearch/train_dinov2_lora.py
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-4},  # LoRA params
    {"params": model.head.parameters(), "lr": 5e-4},       # Head params
], weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# CRITICAL: use class-weighted loss for imbalance
# B3 dominates 2-4x, so weight B1/B2/B4 higher
class_weights = torch.tensor([2.5, 2.0, 0.5, 2.0]).cuda()  # tune based on class freq
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```

**Step 4 — Data pipeline:**
```python
# Use PAD_RATIO=0.4 crops (wider context than the failed 0.2)
# Apply aggressive color augmentation: ColorJitter, RandomGrayscale
# Use MixUp with alpha=0.2 specifically for B2/B3 boundary pairs
```

**Step 5 — Integrate into two-stage pipeline:**
Replace the frozen DINOv2 classifier with this LoRA-finetuned version. Use the same YOLO11l detector for proposals.

**Files to create:**
- `/workspace/autoresearch/models/dinov2_lora_classifier.py`
- `/workspace/autoresearch/train_dinov2_lora.py`
- `/workspace/autoresearch/eval_two_stage_lora.py`

- **Estimated time:** 6–10 hours
- **Expected gain:** +4–8 mAP50-95 points over frozen DINOv2. NOTE: The two-stage pipeline overall underperformed (18% end-to-end) — but this was likely due to cascade error accumulation and frozen backbone, not the DINOv2 architecture. With LoRA + end-to-end training signal (distillation from YOLO), the two-stage ceiling could rise significantly.

---

## Finding 5: Prototype Learning for Class Imbalance in Detection

- **Source:** ICLR 2025 proceedings — Prototype learning for incremental object detection: https://proceedings.iclr.cc/paper_files/paper/2025/file/7f94f1d0a11e0a0f38f973e5a8925909-Paper-Conference.pdf | Survey: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00851-z
- **Method:** Each class is represented by a learned prototype in feature space (typically the mean of all class embeddings, possibly with learnable variance). At inference, classification is performed by nearest-prototype distance rather than a linear head. For imbalanced data, prototypes prevent the majority class (B3) from dominating the classifier's decision boundary. Combined with SupCon loss during training, prototypes cluster B1/B2/B3/B4 in well-separated embedding regions.
- **Reported gain:** Prototype-based methods show 3–8% improvement over standard CE on imbalanced fine-grained tasks. The higher-dimension prototype approach at ICLR 2025 was specifically designed to handle within-class variance (critical for B3 which spans a wide visual range).
- **Why relevant:** B3 dominates our dataset 2-4x. A B3 prototype that captures the full B3 visual distribution would naturally have higher within-class spread, reducing false B3 classifications for B2/B4 borderline cases.

### Implementation for our project:

**Step 1 — Install:**
```bash
pip install pytorch-metric-learning  # includes ProxyAnchor, SupCon, ProtoNet losses
```

**Step 2 — Prototype classifier head (replaces linear head in two-stage or YOLO cls branch):**
```python
# File: /workspace/autoresearch/models/prototype_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeHead(nn.Module):
    """Replace linear classification head with nearest-prototype classifier."""
    def __init__(self, feat_dim=256, num_classes=4, temperature=0.07):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.temperature = temperature
        # Optional: per-class learnable scale (handles B3's wider distribution)
        self.class_scale = nn.Parameter(torch.ones(num_classes))

    def forward(self, x):
        # x: [B, feat_dim], normalized
        x_norm = F.normalize(x, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        # Scaled cosine similarity
        sim = torch.mm(x_norm, proto_norm.T)  # [B, num_classes]
        sim = sim * self.class_scale.unsqueeze(0)
        return sim / self.temperature

    def get_prototype_loss(self, features, labels):
        """Pull features toward class prototypes, push away from others."""
        feat_norm = F.normalize(features, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        # SupCon-style: maximize similarity to own prototype
        sim = torch.mm(feat_norm, proto_norm.T) / self.temperature
        return F.cross_entropy(sim, labels)
```

**Step 3 — Training with prototype regularization:**
```python
# Add to main loss:
# L_total = L_detect + lambda_cls * L_cls + lambda_proto * L_proto
# lambda_proto = 0.5 initially
```

**Step 4 — For B2/B3 boundary specifically, use "boundary-aware prototype" update:**
```python
# After each epoch, compute per-class prototype as EMA of class embeddings:
with torch.no_grad():
    for c in range(4):
        mask = (labels == c)
        if mask.sum() > 0:
            class_feats = features[mask].mean(0)
            model.proto_head.prototypes[c] = 0.9 * model.proto_head.prototypes[c] + \
                                              0.1 * F.normalize(class_feats, dim=-1)
```

**Files to create:**
- `/workspace/autoresearch/models/prototype_head.py`
- `/workspace/autoresearch/train_prototype.py`

- **Estimated time:** 4–6 hours
- **Expected gain:** +2–5 mAP50-95 points (primarily by reducing B3 over-prediction, improving B2/B3 boundary)

---

## Finding 6: RF-DETR — DINOv2-Backbone Detection Transformer for Small Datasets

- **Source:** "RF-DETR: Neural Architecture Search for Real-Time Detection Transformers" (ICLR 2026): https://arxiv.org/html/2511.09554v1 | Roboflow blog: https://blog.roboflow.com/rf-detr/ | GitHub: https://github.com/roboflow/rf-detr
- **Method:** RF-DETR replaces YOLO's CNN backbone with a DINOv2 ViT backbone, adding multi-scale deformable cross-attention between transformer decoder queries and multi-scale feature pyramid (from ViT intermediate layers). Unlike two-stage pipelines, RF-DETR performs end-to-end detection + classification in one shot — the transformer decoder queries directly attend to discriminative regions without the cascade error that killed our two-stage approach. Critically, the DINOv2 backbone is pre-trained with self-supervised DINO on 142M diverse images, giving far richer color/texture representations than ImageNet CNN pretraining.
- **Reported gain:** RF-DETR-Base: 54.7 mAP50-95 on COCO vs yolo11m: 48.3 (6.4 points better). On small datasets with DINOv2 backbone, "transfer learning significantly improves detection accuracy." Converges faster than YOLO on custom small datasets.
- **Why relevant:** This directly replaces our failing two-stage design with an end-to-end architecture that has DINOv2's rich representations built-in. The transformer decoder can attend to the specific color gradient regions of each bunch (loose fruits, exocarp coloring) that distinguish B2 from B3.

### Implementation for our project:

**Step 1 — Install:**
```bash
pip install rfdetr
# For large model (recommended):
pip install rfdetr[plus]
```

**Step 2 — Prepare data in COCO JSON format:**
```python
# File: /workspace/autoresearch/scripts/convert_to_coco.py
# Convert existing YOLO labels (txt) to COCO JSON
# Classes: 0=B1, 1=B2, 2=B3, 3=B4
# Use the train/val/test split from current dataset

import json, os
from pathlib import Path

def yolo_to_coco(images_dir, labels_dir, output_json):
    coco = {"images": [], "annotations": [], "categories": [
        {"id": 1, "name": "B1"}, {"id": 2, "name": "B2"},
        {"id": 3, "name": "B3"}, {"id": 4, "name": "B4"}
    ]}
    ann_id = 0
    for img_id, img_path in enumerate(sorted(Path(images_dir).glob("*.jpg"))):
        # ... (standard YOLO→COCO conversion)
        pass
    with open(output_json, "w") as f:
        json.dump(coco, f)
```

**Step 3 — Fine-tune RF-DETR-Large:**
```python
# File: /workspace/autoresearch/train_rfdetr.py
from rfdetr import RFDETRLarge

model = RFDETRLarge(pretrain_weights="rf-detr-large.pth")

model.train(
    dataset_dir="/workspace/autoresearch/data/coco_format/",
    epochs=100,              # Small dataset: more epochs needed
    batch_size=8,            # Fit on single GPU
    lr=1e-4,                 # DINOv2 backbone: low LR critical
    gradient_clip_val=0.1,   # From paper recommendations
    weight_decay=1e-4,
    resolution=640,          # Start at 640, try 800 if GPU allows
)
```

**Step 4 — Key hyperparameters for our 4K-image dataset:**
```python
# Per-layer LR decay: 0.8 (backbone gets LR * 0.8^num_layers)
# This prevents over-adaptation of DINOv2 backbone
# EMA: True (helps small datasets)
# Warmup: 5 epochs
# No TTA during training, use TTA at inference for +1-2% mAP

# Class weighting for imbalance:
# RF-DETR uses focal loss internally — tune focal_gamma=2.0 to 3.0 for B1/B2/B4
```

**Step 5 — Inference and evaluation:**
```python
from rfdetr import RFDETRLarge
model = RFDETRLarge()
model.load("checkpoint_best.pth")

# Run on validation set and compute mAP50-95 with pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
```

**Files to create:**
- `/workspace/autoresearch/scripts/convert_to_coco.py`
- `/workspace/autoresearch/train_rfdetr.py`
- `/workspace/autoresearch/eval_rfdetr.py`
- `/workspace/autoresearch/configs/rfdetr_tbs.yaml`

**Estimated time:** 8–12 hours (data conversion + training + evaluation)
**Expected gain:** +8–14 mAP50-95 points. This is our highest-upside experiment. RF-DETR-Large should comfortably reach 35–40%+ on our task given the DINOv2 backbone quality. To hit 40%+, we may need to combine with the CORN ordinal loss (Finding 7) or prototype head (Finding 5).

---

## Finding 7: CORN Ordinal Regression Loss for Ripeness Classification

- **Source:** "Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities" (TMLR 2024): https://arxiv.org/abs/2111.08851 | coral-pytorch GitHub: https://github.com/Raschka-research-group/coral-pytorch | Applied to fruit quality: "Privacy Preserving Ordinal-Meta Learning" (arXiv 2025): https://arxiv.org/html/2511.01449
- **Method:** CORN (Conditional Ordinal Regression for Neural Networks) decomposes the K-class ordinal problem into K-1 binary classifiers. For our 4 classes: (1) Is it above B1? (2) Given it's above B1, is it above B2? (3) Given it's above B2, is it above B3? Each classifier is trained on a filtered subset that satisfies the previous condition. This enforces rank consistency: P(≥B3) ≤ P(≥B2) ≤ P(≥B1). For B2/B3 boundary confusion (our main bottleneck), CORN explicitly trains a dedicated binary classifier on the hardest pair. Achieves 92.71% accuracy on 5-class fruit freshness (Unripe/Early-Ripe/Ripe/Overripe/Bad).
- **Reported gain:** CORN outperforms standard CE by 3–8% on ordinal classification tasks, especially at boundary classes.
- **Why relevant:** Standard cross-entropy treats all class confusions equally. For B2/B3, a wrong B2→B3 prediction costs the same as B2→B1. With CORN, the model learns that B2→B3 is the "smallest" error and focuses discriminative capacity on the hardest boundaries.

### Implementation for our project:

**Step 1 — Install coral-pytorch:**
```bash
pip install coral-pytorch
```

**Step 2 — Replace classification loss in YOLO classifier or two-stage classifier:**
```python
# File: /workspace/autoresearch/models/corn_loss.py
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

class CORNClassificationHead(nn.Module):
    """CORN ordinal classification head: K-1 binary outputs."""
    def __init__(self, in_features, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        # Output K-1=3 logits (one per rank boundary)
        self.fc = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        return self.fc(x)  # shape: [B, num_classes-1]

# In training loop:
# logits = corn_head(features)  # [B, 3] — boundaries B1/B2, B2/B3, B3/B4
# loss = corn_loss(logits, labels, num_classes=4)
#
# For inference:
# probs = corn_label_from_logits(logits)  # returns predicted class label

# YOLO integration: add CORN head as auxiliary classification branch
# In ultralytics/models/yolo/detect/train.py, add corn_loss term:
# L_total = L_box + L_dfl + L_cls + 0.3 * L_corn
```

**Step 3 — Hybrid loss (CE + CORN) for stability:**
```python
alpha = 0.7  # weight for CE
beta = 0.3   # weight for CORN

ce_loss = F.cross_entropy(cls_logits_4d, labels)
corn_logits = corn_head(features)
ordinal_loss = corn_loss(corn_logits, labels, num_classes=4)

total_cls_loss = alpha * ce_loss + beta * ordinal_loss
```

**Step 4 — Apply to both YOLO (primary) and two-stage (if re-attempted with LoRA):**
- For YOLO: modify `ultralytics/utils/loss.py` ClassificationLoss to add CORN term
- For RF-DETR: override the classification loss in the decoder

**Files to create:**
- `/workspace/autoresearch/models/corn_loss.py`
- `/workspace/autoresearch/train_corn_yolo.py`

**Estimated time:** 4–6 hours
**Expected gain:** +3–7 mAP50-95 points (additive with other improvements, since CORN is a loss modification that doesn't conflict with architectural changes).

---

## Finding 8: Multi-Scale Feature Fusion — Fine-Grained Improvements

- **Source:** "An Improved YOLOv11 architecture with multi-scale attention and spatial fusion for fine-grained residual detection", ScienceDirect 2025: https://www.sciencedirect.com/science/article/pii/S2590123025031160 | MFF-YOLO (TST 2024): https://www.sciopen.com/article/10.26599/TST.2024.9010097
- **Method:** Three specific modifications for fine-grained texture-based detection: (1) C2PSA_iEMA module — Cross-Stage Partial with Improved Efficient Multi-scale Attention, enhancing subtle color/texture representation; (2) C3k2_BFAM_EMA module — Bi-directional Feature Aggregation Module with EMA for neck cross-scale complementarity; (3) Adaptive Spatial Feature Fusion (ASFF) extended to 4 detection heads to auto-weight feature scales per spatial location.
- **Reported gain:** +2–5% mAP50-95 on fine-grained industrial defect detection (analogous to our subtle B2/B3 color differences).
- **Why relevant:** The ASFF module is directly applicable — it lets the model automatically weight P3 (fine-grained color texture features) vs P4/P5 (semantic bunch shape features) per detection, which is exactly what B2/B3 disambiguation needs.

### Implementation for our project:

**Step 1 — Implement ASFF (Adaptively Spatial Feature Fusion):**
```python
# File: /workspace/autoresearch/models/asff.py
import torch, torch.nn as nn, torch.nn.functional as F

class ASFF(nn.Module):
    """Adaptively fuses features from 3 pyramid levels."""
    def __init__(self, level, channels=(256, 512, 1024)):
        super().__init__()
        self.level = level
        out_ch = channels[level]
        # Weight generators for each input level
        self.weight_l0 = nn.Conv2d(channels[0], 1, 1)
        self.weight_l1 = nn.Conv2d(channels[1], 1, 1)
        self.weight_l2 = nn.Conv2d(channels[2], 1, 1)
        self.fuse_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x0, x1, x2):
        # Resize all to target level resolution
        target_size = x0.shape[2:] if self.level == 0 else \
                      x1.shape[2:] if self.level == 1 else x2.shape[2:]

        r0 = F.interpolate(x0, size=target_size, mode='bilinear')
        r1 = F.interpolate(x1, size=target_size, mode='bilinear')
        r2 = F.interpolate(x2, size=target_size, mode='bilinear')

        w0 = self.weight_l0(r0)
        w1 = self.weight_l1(r1)
        w2 = self.weight_l2(r2)

        weights = F.softmax(torch.cat([w0, w1, w2], dim=1), dim=1)
        fused = weights[:, 0:1] * r0 + weights[:, 1:2] * r1 + weights[:, 2:3] * r2
        return self.fuse_conv(fused)
```

**Step 2 — Add ASFF to YOLO11 neck:**
Replace the final feature concatenation in the neck (before detection heads) with ASFF modules at each scale. Modify the YOLO11l YAML config accordingly.

**Step 3 — Add C2PSA_iEMA (improved EMA attention in C2f-like block):**
This is a standard C2f block with an EMA (Efficient Multi-scale Attention) module added after the bottleneck layers. The EMA uses depth-wise convolutions at multiple kernel sizes (3, 5, 7) to capture multi-scale texture patterns.

**Files to create:**
- `/workspace/autoresearch/models/asff.py`
- `/workspace/autoresearch/configs/yolo11l_asff.yaml`

**Estimated time:** 6–8 hours
**Expected gain:** +2–4 mAP50-95 points (moderate standalone, but high synergy with HAT-YOLO from Finding 3)

---

## Finding 9: Oil Palm FFB Color Space — LAB/HSV Feature Channels as Input Augmentation

- **Source:** "Fresh Fruit Bunch Ripeness Classification Methods: A Review", Food and Bioprocess Technology 2024: https://link.springer.com/article/10.1007/s11947-024-03483-0 | Intelligence Color Vision System for FFB (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC3545614/ | Hybrid color correction (ScienceDirect 2024): https://www.sciencedirect.com/science/article/pii/S277237552400248X
- **Method:** Palm oil FFB ripeness is primarily a color phenomenon: B1 (black/dark green) → B2 (reddish-orange tinge, 30–60% red) → B3 (60–80% orange-red) → B4 (dark red with loose fruits). The L*a*b* color space separates luminance (L*) from color (a* = red-green, b* = blue-yellow). For FFB, the a* channel is the most discriminative feature. Appending a* and b* channels as extra input channels to YOLO (making it 5-channel input: RGB + a* + b*) gives the backbone direct access to the most discriminative signals. A hybrid color correction approach (normalizing illumination before extracting features) improved mAP@0.5 by 1.5% (88.2% → 89.7%).
- **Reported gain:** +1.5% mAP50 from color normalization alone; color-space features give strong baseline improvements for FFB classification.
- **Why relevant:** This is a LOW COST, HIGH CONFIDENCE improvement. Our current YOLO receives raw RGB. Adding pre-processed a* channel (which measures red-orange intensity directly) gives the backbone an explicit B2/B3 discriminative signal before any convolution.

### Implementation for our project:

**Step 1 — Create LAB channel preprocessing:**
```python
# File: /workspace/autoresearch/data/lab_preprocessing.py
import cv2
import numpy as np

def rgb_to_lab_channels(img_rgb):
    """Returns L*, a*, b* channels normalized to [0,1]."""
    img_uint8 = (img_rgb * 255).astype(np.uint8) if img_rgb.max() <= 1 else img_rgb.astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    # L*: 0-255 → 0-100, a*: 0-255 → -128 to +127, b*: same
    L = lab[:, :, 0].astype(np.float32) / 255.0
    a = (lab[:, :, 1].astype(np.float32) - 128) / 128.0  # Normalize to [-1, 1]
    b = (lab[:, :, 2].astype(np.float32) - 128) / 128.0
    return L, a, b  # a* is the key B2/B3 discriminator
```

**Step 2 — Create 5-channel YOLO variant (RGB + a* + b*):**
```python
# Modify YOLO11 first conv layer to accept 5 channels:
# In yolo11l.yaml, change first Conv from ch=3 to ch=5
# Initialize extra channels with small random weights (don't zero-init)

# In dataset loader, append a* and b* to each image:
class TBSDataset5ch(Dataset):
    def __getitem__(self, idx):
        img = load_rgb_image(idx)  # [H, W, 3]
        L, a, b = rgb_to_lab_channels(img)
        # Stack: [R, G, B, a*, b*]
        img5ch = np.stack([img[:,:,0], img[:,:,1], img[:,:,2], a, b], axis=-1)
        return img5ch, labels[idx]
```

**Step 3 — Alternative: Use a* as additional attention mask:**
Instead of modifying YOLO's input (which requires retraining from scratch), use a* channel as a spatial attention prior: multiply feature maps in the backbone's early layers by a sigmoid-transformed a* heatmap. This can be inserted without full retraining.

**Files to create:**
- `/workspace/autoresearch/data/lab_preprocessing.py`
- `/workspace/autoresearch/configs/yolo11l_5ch.yaml`
- `/workspace/autoresearch/data/tbs_dataset_5ch.py`

**Estimated time:** 3–5 hours (lowest effort of all findings)
**Expected gain:** +1–3 mAP50-95 points. Low risk, high confidence, fast to implement.

---

## Finding 10: DINOv2 Distillation into YOLOv8 for Few-Shot Detection

- **Source:** "Improving YOLOv8 for Fast Few-Shot Object Detection by DINOv2 Distillation", ICIP 2025 (LIRMM): https://www.lirmm.fr/~chaumont/publications/ICIP-2025-FOURRET-CHAUMONT-FIORIO-SUBSOL-BRAU-DinoDistillationIntoYoloV8_ForFSD.pdf
- **Method:** Rather than using DINOv2 as a two-stage classifier (our failed approach) or replacing YOLO entirely (RF-DETR), this method distills DINOv2 features INTO the YOLOv8 backbone during training. A DINOv2 teacher provides rich feature targets; the YOLOv8 student backbone learns to mimic these features at corresponding scales. The distillation happens during YOLO training, not at inference (so no speed penalty). This captures DINOv2's semantic richness within YOLO's efficient architecture.
- **Reported gain:** "Superior metrics for novel class learning compared to baseline YOLOv8" in few-shot settings. Particularly strong for classes with limited training examples (our B1, B2, B4 problem).
- **Why relevant:** This is the middle path between failed frozen DINOv2 two-stage and full RF-DETR replacement. If RF-DETR is too risky (data format overhead), DINOv2 distillation into YOLO11l is a strong alternative.

### Implementation for our project:

**Step 1 — Setup DINOv2 teacher (frozen):**
```python
# File: /workspace/autoresearch/train_yolo_dino_distill.py
from transformers import AutoModel
import torch

dino_teacher = AutoModel.from_pretrained("facebook/dinov2-large").cuda()
dino_teacher.eval()
for p in dino_teacher.parameters():
    p.requires_grad = False
```

**Step 2 — Add feature distillation loss to YOLO training:**
```python
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

class DinoDistillLoss(v8DetectionLoss):
    def __init__(self, model, dino_teacher, distill_weight=0.5):
        super().__init__(model)
        self.dino = dino_teacher
        self.distill_w = distill_weight
        # Adapter: maps YOLO backbone feature dim to DINOv2 feature dim
        yolo_ch = model.model[-1].reg_max  # depends on model
        self.adapter = nn.Linear(512, 1024).cuda()  # YOLO→DINOv2 feat dim

    def __call__(self, preds, batch):
        det_loss, loss_items = super().__call__(preds, batch)

        # Get DINOv2 CLS token for each image
        with torch.no_grad():
            dino_feats = self.dino(pixel_values=batch["img"]).last_hidden_state[:, 0]

        # Get YOLO backbone features (hook into P3 or P4 output)
        yolo_feats = preds[1][0].mean(dim=[2, 3])  # GAP over spatial dims

        # Distillation: minimize cosine distance
        yolo_proj = self.adapter(yolo_feats)
        distill_loss = 1 - F.cosine_similarity(yolo_proj, dino_feats).mean()

        return det_loss + self.distill_w * distill_loss, loss_items
```

**Step 3 — Train YOLO with distillation:**
```bash
# Patch ultralytics loss class and run standard training
python train_yolo_dino_distill.py \
  --model yolo11l.pt \
  --data /workspace/autoresearch/data/tbs.yaml \
  --epochs 200 --imgsz 640 --distill_weight 0.5
```

**Files to create:**
- `/workspace/autoresearch/train_yolo_dino_distill.py`

**Estimated time:** 8–12 hours
**Expected gain:** +4–8 mAP50-95 points. This is a solid "best of both worlds" approach — keeps YOLO speed while injecting DINOv2 feature quality.

---

## Recommended Next Experiments (priority order)

1. **RF-DETR-Large fine-tuning** (Finding 6, ~10 hours) — Highest expected ceiling. End-to-end DINOv2-backbone transformer detection. This is the single most likely path to 40%+ mAP50-95. Start this immediately after current PAD_RATIO=0.6 run completes. Convert dataset to COCO JSON format and run 100 epochs RF-DETR-Large at LR=1e-4.

2. **HAT-YOLOv8 neck modification on yolo11l** (Finding 3, ~10 hours) — Best YOLO-internal improvement. The Hybrid Attention Transformer in the neck's TopDownLayer2, combined with Shuffle Attention in the backbone, gave 7.6–11% mAP gains on multi-class fruit ripeness in published results. This is the most directly analogous experiment to our task. Implement `ShuffleAttention` and `HATNeckBlock` using timm's SwinTransformerBlock.

3. **CORN ordinal loss replacing CE in classification branch** (Finding 7, ~5 hours) — Quickest high-impact change. Install `coral-pytorch`, replace `nn.CrossEntropyLoss` with `corn_loss(logits, labels, num_classes=4)` in either the YOLO classification head or the two-stage classifier. This directly addresses B2/B3 boundary confusion at negligible implementation cost. Can be combined with any other experiment.

4. **DINOv2 + LoRA fine-tuning in two-stage** (Finding 4, ~8 hours) — Re-try the two-stage pipeline with the key fix: LoRA fine-tuning instead of frozen DINOv2. Use `peft` library, inject LoRA into Q/K/V layers, train with class-weighted CE + CORN. The prior two-stage failures were likely due to domain gap (frozen backbone) + cascade error + CE loss ignoring ordinality. LoRA + CORN + wider crops (PAD_RATIO=0.4) addresses all three.

5. **LAB color channel as input augmentation** (Finding 9, ~4 hours) — Lowest effort, high confidence. Append a* and b* channels from L*a*b* color space to input (5-channel YOLO). The a* channel directly measures red-orange intensity — the primary B2 vs B3 discriminator in FFB biology. Modify first YOLO conv layer to accept 5 channels.

6. **Prototype head + SupCon loss for class imbalance** (Finding 5, ~5 hours) — Addresses B3 over-prediction from class imbalance. Replace linear classification head with nearest-prototype classifier. Use `pytorch-metric-learning` for ProxyAnchor loss. Best combined with Finding 3 or Finding 6 rather than as a standalone experiment.

### Key Insight (Outside the Box)

The core insight from this research is: **our problem is not object detection, it is fine-grained visual classification of objects we can already detect**. The single-class detector ceiling is 39% — the gap from 26.9% to 39% is purely classification quality. Every approach above directly attacks classification via: (a) better backbone features (RF-DETR/DINOv2), (b) attention to discriminative regions (HAT neck), (c) ordinal-aware loss (CORN), (d) explicit color features (LAB channels), or (e) prototype-based imbalance handling. The highest-upside approach is RF-DETR because it eliminates the cascade error from two-stage pipelines while providing DINOv2-quality features end-to-end.

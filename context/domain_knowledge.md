# Domain Knowledge

## Oil Palm Fruit Bunch (FFB) Maturity Classes

| Class | Maturity | Visual Characteristics |
|-------|----------|----------------------|
| B1 | Unripe | Dark purple/black, compact fruitlets |
| B2 | Underripe | Some orange fruitlets visible, mostly dark |
| B3 | Ripe | Predominantly orange/red, loose fruitlets |
| B4 | Overripe | Deep orange/brown, very loose, some fallen fruitlets |

## Key Detection Challenges

- **B2/B3 overlap**: B2 and B3 overlap ~84% in bounding box size. The transition from underripe to ripe is gradual — color and texture are the primary distinguishing features, not shape or size.
- **B4 rarity**: B4 (overripe) is underrepresented in the dataset. Recall matters more than precision for harvest timing.
- **Occlusion**: Bunches are frequently partially occluded by fronds and other vegetation.
- **Lighting variation**: Plantation canopy creates highly variable lighting conditions.

## What Matters for Improvement

- B2/B3 confusion rate is the key bottleneck — reducing it directly improves overall mAP
- B4 recall is operationally important (missed overripe bunches = lost yield)
- Per-class AP is more informative than headline mAP for diagnosing progress
- Confusion patterns show where the model fails, not why — hypotheses about causes must be tested

## Dataset

- Varieties: Damimas (Blok A21B), Lonsum (Blok A21A)
- 4 classes, YOLO format annotations
- Canonical split is frozen in prepare.py

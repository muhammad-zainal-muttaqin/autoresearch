# Color Feature Analysis Report

Generated: 2026-03-15

## Summary

Run color_classifier.py on Dataset-Crops/val (2786 crop images).

**Verdict**: HSV color alone is NOT sufficient for ripeness classification.
Color classifier val accuracy = **31.6%** vs EfficientNet's **62.7%**.
Using color as primary classifier would be a significant regression.

---

## HSV Features Per Class (Training Set, 100 samples each)

| Class | mean_H | Green | Yellow | Orange | Red | Saturation |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 26.6° | 1.6% | 31.3% | **52.1%** | 13.7% | 49.0% |
| B2 | 46.5° | 7.0% | **29.0%** | 35.3% | 5.2% | 31.6% |
| B3 | 74.5° | **6.6%** | 16.6% | 21.1% | 0.7% | 30.7% |
| B4 | 55.3° | 4.1% | **26.7%** | 26.4% | 4.8% | 29.7% |

### Critical Observations

1. **B1 is NOT green** (mean_H=26.6° = orange range). Intuitive assumption (unripe=green) is WRONG.
   - B1 dominant color: orange (52.1%). This is counter-intuitive.
   - Hypothesis: B1 may represent "freshly harvested/ripe" stage, not "unripe".
   - OR: The B1 fruits in this dataset happen to be photographed against orange soil/background.

2. **B2 vs B3 color overlap is severe**:
   - B2 mean_H = 46.5° (yellow-green range)
   - B3 mean_H = 74.5° (green range)
   - Both have similar saturation (~30-31%)
   - The rule "B3=orange/red" is backward — B3 appears greener than B2

3. **B4 is between B2 and B3** in color space (mean_H=55.3°), making color-based B4 detection ambiguous.

4. **Counter-intuitive ordering**: By hue, ordering is B1(26°) < B2(46°) < B4(55°) < B3(75°)
   This does NOT match the expected ripeness progression B1→B2→B3→B4.

---

## Validation Set Results

| Class | GT Count | Color Correct | Color Acc | EfficientNet Acc |
|---|---:|---:|---:|---:|
| B1 | 301 | 3 | **1.0%** | ~85% |
| B2 | 612 | 252 | **41.2%** | 46.6% |
| B3 | 1319 | 571 | **43.3%** | ~75% |
| B4 | 554 | 53 | **9.6%** | ~55% |
| **Overall** | **2786** | **879** | **31.6%** | **62.7%** |

### Confusion Matrix (Val Set)

```
           B1    B2    B3    B4
B1 (301):   3   131   121    46
B2 (612):  15   252   292    53
B3 (1319): 98   559   571    91
B4 (554):  52   263   186    53
```

### Key Confusions:
- B1 is almost entirely predicted as B2 or B3 (color rule predicts orange=B2, but B1 IS orange!)
- B2 split between B2 (41%) and B3 (48%) — nearly random
- B3 split between B2 (42%) and B3 (43%) — worse than B2!
- B4 entirely mis-predicted (9.6% acc)

---

## Conclusion: Color is NOT Discriminative

**The B2/B3 confusion is NOT a color problem.**

If B2/B3 were distinguishable by color, the HSV rule-based classifier should achieve at least
50-60% B2 accuracy. Instead, B2=41.2% (barely above random for 4-class problem=25%).

This means:
- B2/B3 distinction requires **morphological/textural features** (shape, surface texture, spikelet density)
- NOT color features
- This is consistent with DINOv2 being a better approach (rich texture/shape representations)

**Recommendation**: Do NOT use color features as classifier input.
They add noise rather than discriminative signal.

However, color COULD be useful as a **negative filter**:
- If mean_H > 70° (very green): likely B3, unlikely B1/B4
- This might reduce false positives for B1 classification

---

## Integration Recommendation

Color features should NOT be integrated into the main pipeline.
Color classifier accuracy (31.6%) is far below EfficientNet (62.7%) and expected DINOv2 (>70%).

**One potential use**: Color as a hard reject for B1 detection
- If detection predicted as B1 but crop is clearly green (H > 70°, green_ratio > 30%): reject
- This could reduce false B1 detections without complex retraining

This is a minor optimization and not a priority given the larger gains expected from DINOv2.

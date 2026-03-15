"""
Color-based classifier for oil palm bunch ripeness.

Key insight: B1=green, B2=yellow-green, B3=orange/red, B4=small dark red.
Ripeness IS largely color. Test if color histogram can outperform learned features.

Also test: HSV + learned features ensemble (color-enhanced classifier).
"""

import cv2
import numpy as np
from pathlib import Path
import json

CROP_DIR = Path("/workspace/autoresearch/Dataset-Crops")
CLASS_NAMES = ["B1", "B2", "B3", "B4"]


def extract_hsv_features(img_bgr):
    """Extract HSV histogram features from a BGR crop image."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(float)
    s = hsv[:, :, 1].astype(float) / 255.0
    v = hsv[:, :, 2].astype(float) / 255.0

    # Only consider pixels with reasonable saturation and brightness
    mask = (s > 0.2) & (v > 0.1)
    h_masked = h[mask]

    if len(h_masked) == 0:
        return {'green': 0, 'yellow': 0, 'orange': 0, 'red': 0, 'dark': 0, 'mean_h': 0, 'mean_s': 0}

    # Hue ranges (OpenCV: 0-180)
    # Red/dark red: H < 10 or H > 170
    # Orange: H 10-25
    # Yellow: H 25-45
    # Yellow-green: H 45-70
    # Green: H 70-90

    red_ratio = np.mean((h_masked < 10) | (h_masked > 165))
    orange_ratio = np.mean((h_masked >= 10) & (h_masked < 25))
    yellow_ratio = np.mean((h_masked >= 25) & (h_masked < 45))
    yellow_green_ratio = np.mean((h_masked >= 45) & (h_masked < 70))
    green_ratio = np.mean((h_masked >= 70) & (h_masked < 90))

    return {
        'red': red_ratio,
        'orange': orange_ratio,
        'yellow': yellow_ratio,
        'yellow_green': yellow_green_ratio,
        'green': green_ratio,
        'mean_h': float(np.mean(h_masked)),
        'mean_s': float(np.mean(s[mask])),
        'mean_v': float(np.mean(v[mask])),
        'dark_ratio': float(np.mean(v < 0.2))
    }


def color_predict_class(features):
    """
    Rule-based class prediction from HSV features.
    B1: dominant green (H 70-90)
    B2: yellow-green to yellow (H 25-70)
    B3: orange to red (H 5-25)
    B4: dark red + small (H <10 or >165), often darker
    """
    green = features['green'] + features.get('yellow_green', 0) * 0.5
    yellow = features['yellow'] + features.get('yellow_green', 0) * 0.5
    orange = features['orange']
    red_dark = features['red']

    scores = {
        0: green,      # B1 = green
        1: yellow,     # B2 = yellow
        2: orange,     # B3 = orange
        3: red_dark,   # B4 = dark red
    }

    return max(scores, key=scores.get), scores


def evaluate_color_classifier(split):
    """Evaluate rule-based color classifier on crops."""
    split_dir = CROP_DIR / split
    class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    class_total = {0: 0, 1: 0, 2: 0, 3: 0}

    confusion = np.zeros((4, 4), dtype=int)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            continue

        for img_path in cls_dir.glob('*.jpg'):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            features = extract_hsv_features(img)
            pred_cls, scores = color_predict_class(features)

            confusion[cls_idx, pred_cls] += 1

            if pred_cls == cls_idx:
                class_correct[cls_idx] += 1
            class_total[cls_idx] += 1

    total_correct = sum(class_correct.values())
    total = sum(class_total.values())

    print(f"\n{split} set results:")
    print(f"  Overall accuracy: {100.*total_correct/total:.1f}%")
    print(f"  Per-class accuracy:")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if class_total[cls_idx] > 0:
            acc = 100. * class_correct[cls_idx] / class_total[cls_idx]
            print(f"    {cls_name}: {acc:.1f}% ({class_correct[cls_idx]}/{class_total[cls_idx]})")

    print(f"\n  Confusion matrix (rows=GT, cols=pred):")
    print(f"  {'':12s} " + " ".join(f"{n:8s}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        row_str = " ".join(f"{confusion[i,j]:8d}" for j in range(4))
        print(f"  {name:12s} {row_str}")

    return total_correct / total if total > 0 else 0


def analyze_color_by_class():
    """Print mean HSV features per class to understand separability."""
    print("Mean HSV features per class (training set):")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = CROP_DIR / 'train' / cls_name
        all_features = []

        for img_path in list(cls_dir.glob('*.jpg'))[:100]:  # Sample 100
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            features = extract_hsv_features(img)
            all_features.append(features)

        if not all_features:
            continue

        mean_h = np.mean([f['mean_h'] for f in all_features])
        mean_s = np.mean([f['mean_s'] for f in all_features])
        green = np.mean([f['green'] + f.get('yellow_green', 0)*0.5 for f in all_features])
        yellow = np.mean([f['yellow'] + f.get('yellow_green', 0)*0.5 for f in all_features])
        orange = np.mean([f['orange'] for f in all_features])
        red = np.mean([f['red'] for f in all_features])

        print(f"  {cls_name}: mean_H={mean_h:.1f}° | "
              f"green={100*green:.1f}% | yellow={100*yellow:.1f}% | "
              f"orange={100*orange:.1f}% | red={100*red:.1f}% | "
              f"sat={100*mean_s:.1f}%")


if __name__ == '__main__':
    print("=== Color-Based Ripeness Classifier Analysis ===")
    print("\nStep 1: Analyze HSV features per class...")
    analyze_color_by_class()

    print("\nStep 2: Evaluate rule-based color classifier...")
    val_acc = evaluate_color_classifier('val')

    print("\nStep 3: Compare with learned EfficientNet classifier (62.74% val acc)")
    print(f"  Color classifier val acc: {100*val_acc:.1f}%")
    print(f"  EfficientNet classifier val acc: 62.7%")

    if val_acc > 0.627:
        print("  COLOR WINS! Consider using color as primary classifier.")
    else:
        print("  EfficientNet is better overall.")
    print("  Key question: does color help B2/B3 specifically?")

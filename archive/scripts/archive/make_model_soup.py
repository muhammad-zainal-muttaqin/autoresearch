"""
Model Soup: average weights of multiple independently trained models.

Greedy soup algorithm:
1. Start with best individual model
2. Add models one by one, averaging weights
3. Keep the average only if it improves val performance

Reference: "Model soups: averaging weights of multiple fine-tuned models improves accuracy and robustness" (Wortsman et al., 2022)
"""

import torch
from pathlib import Path
from ultralytics import YOLO
from prepare import DATA_YAML

SOUP_DIR = Path("/workspace/autoresearch")
MODEL_PATHS = [
    SOUP_DIR / "soup_seed0.pt",
    SOUP_DIR / "soup_seed42.pt",
    SOUP_DIR / "soup_seed123.pt",
]
OUTPUT_PATH = SOUP_DIR / "model_soup.pt"


def evaluate_model_soup(model_path, data_yaml, imgsz=1024):
    """Evaluate a model on val set."""
    model = YOLO(str(model_path))
    results = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        conf=0.001,
        iou=0.6,
        split="val",
        verbose=False,
    )
    return float(results.box.map)  # mAP50-95


def load_state_dict(model_path):
    """Load model state dict from pt file."""
    ckpt = torch.load(str(model_path), map_location="cpu")
    if isinstance(ckpt, dict):
        # Ultralytics saves with 'model' key
        if 'model' in ckpt:
            model_obj = ckpt['model']
            if hasattr(model_obj, 'state_dict'):
                return model_obj.state_dict()
            else:
                return model_obj
        return ckpt
    elif hasattr(ckpt, 'state_dict'):
        return ckpt.state_dict()
    return ckpt


def average_state_dicts(state_dicts):
    """Average multiple state dicts."""
    avg = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key] for sd in state_dicts]
        # Only average float tensors
        if tensors[0].dtype in (torch.float32, torch.float16, torch.bfloat16):
            avg[key] = torch.stack(tensors, dim=0).float().mean(dim=0).to(tensors[0].dtype)
        else:
            avg[key] = tensors[0]  # Use first model for non-float tensors (e.g., indices)
    return avg


def greedy_soup(model_paths, data_yaml, imgsz=1024):
    """
    Greedy soup: add models one by one, keeping the running average
    only if it improves val performance.
    """
    print("=== Model Soup (Greedy Algorithm) ===")

    # Evaluate all individual models first
    print("\n1. Evaluating individual models:")
    individual_scores = []
    for path in model_paths:
        if not path.exists():
            print(f"  SKIP {path.name} — file not found")
            continue
        score = evaluate_model_soup(path, data_yaml, imgsz)
        print(f"  {path.name}: mAP50-95 = {score:.6f}")
        individual_scores.append((score, path))

    if not individual_scores:
        print("No models found!")
        return None

    # Sort by descending score
    individual_scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = individual_scores[0]

    print(f"\n2. Starting greedy soup with best model: {best_path.name} ({best_score:.6f})")

    # Load models as YOLO objects for weight averaging
    # Load all individual checkpoint files
    loaded_ckpts = []
    for _, path in individual_scores:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        loaded_ckpts.append((path.name, ckpt))

    # Get state dict from best model
    best_ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
    best_model_sd = best_ckpt['model'].state_dict()

    current_soup_sds = [best_model_sd]
    current_best = best_score
    soup_members = [best_path.name]

    print("\n3. Greedily adding models:")
    for score, path in individual_scores[1:]:
        # Load this model's state dict
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        candidate_sd = ckpt['model'].state_dict()

        # Try averaging
        trial_sds = current_soup_sds + [candidate_sd]
        trial_avg_sd = average_state_dicts(trial_sds)

        # Save trial soup to temp file
        trial_ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
        trial_ckpt['model'].load_state_dict(trial_avg_sd)
        trial_path = SOUP_DIR / "model_soup_trial.pt"
        torch.save(trial_ckpt, str(trial_path))

        # Evaluate trial soup
        trial_score = evaluate_model_soup(trial_path, data_yaml, imgsz)

        if trial_score > current_best:
            print(f"  + {path.name}: {score:.6f} → soup {trial_score:.6f} (+{trial_score-current_best:.6f}) ADDED")
            current_soup_sds = trial_sds
            current_best = trial_score
            soup_members.append(path.name)
        else:
            print(f"  - {path.name}: {score:.6f} → soup {trial_score:.6f} ({trial_score-current_best:.6f}) REJECTED")

    # Save final soup
    if len(soup_members) > 1:
        print(f"\n4. Final soup members: {soup_members}")
        final_avg_sd = average_state_dicts(current_soup_sds)
        final_ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
        final_ckpt['model'].load_state_dict(final_avg_sd)
        torch.save(final_ckpt, str(OUTPUT_PATH))
        print(f"   Saved to {OUTPUT_PATH}")
        print(f"   Final soup mAP50-95: {current_best:.6f}")
    else:
        print("\n4. No improvement from soup — keeping best individual model")
        import shutil
        shutil.copy2(str(best_path), str(OUTPUT_PATH))

    # Cleanup trial file
    if (SOUP_DIR / "model_soup_trial.pt").exists():
        (SOUP_DIR / "model_soup_trial.pt").unlink()

    return current_best, soup_members


if __name__ == '__main__':
    import os
    os.chdir(DATA_YAML.parent)
    result = greedy_soup(MODEL_PATHS, DATA_YAML, imgsz=1024)
    if result:
        score, members = result
        print(f"\nModel soup complete! mAP50-95 = {score:.6f}")
        print(f"Soup members: {members}")

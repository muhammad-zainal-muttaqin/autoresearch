"""Create RF-DETR compatible dataset from Dataset-TrainTest (train+test split merged)."""

import shutil
from pathlib import Path

SRC_DIR = Path("/workspace/autoresearch/Dataset-TrainTest")
DST_DIR = Path("/workspace/autoresearch/Dataset-RFDETR-TT")


def create_rfdetr_dataset():
    print(f"Creating RF-DETR dataset at {DST_DIR}...")
    for split in ["train", "valid"]:
        (DST_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DST_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # train → train
    src_split = "train"
    n = 0
    for img_path in sorted((SRC_DIR / "images" / src_split).glob("*")):
        if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            dst = DST_DIR / "train" / "images" / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)
            n += 1
    for lbl_path in sorted((SRC_DIR / "labels" / src_split).glob("*.txt")):
        dst = DST_DIR / "train" / "labels" / lbl_path.name
        if not dst.exists():
            shutil.copy2(lbl_path, dst)
    print(f"  train: {n} images")

    # val → valid
    src_split = "val"
    n = 0
    for img_path in sorted((SRC_DIR / "images" / src_split).glob("*")):
        if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            dst = DST_DIR / "valid" / "images" / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)
            n += 1
    for lbl_path in sorted((SRC_DIR / "labels" / src_split).glob("*.txt")):
        dst = DST_DIR / "valid" / "labels" / lbl_path.name
        if not dst.exists():
            shutil.copy2(lbl_path, dst)
    print(f"  valid: {n} images")

    yaml_content = """path: /workspace/autoresearch/Dataset-RFDETR-TT
train: train/images
val: valid/images

nc: 4
names:
  - B1
  - B2
  - B3
  - B4
"""
    with open(DST_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)
    print(f"  data.yaml created")
    print(f"Done: {DST_DIR}")


if __name__ == "__main__":
    create_rfdetr_dataset()

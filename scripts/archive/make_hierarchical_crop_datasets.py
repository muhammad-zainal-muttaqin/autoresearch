"""Create hierarchical crop datasets from Dataset-Crops using symlinks."""

from __future__ import annotations

import shutil
from pathlib import Path


SRC = Path("/workspace/autoresearch/Dataset-Crops")
COARSE_DST = Path("/workspace/autoresearch/Dataset-Crops-Coarse3")
B23_DST = Path("/workspace/autoresearch/Dataset-Crops-B23")
SPLITS = ("train", "val")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_class(src_dir: Path, dst_dir: Path, prefix: str = "") -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in sorted(src_dir.glob("*")):
        if not img_path.is_file():
            continue
        dst_path = dst_dir / f"{prefix}{img_path.name}"
        dst_path.symlink_to(img_path.resolve())
        count += 1
    return count


def build_coarse3() -> None:
    print(f"Creating {COARSE_DST}...")
    reset_dir(COARSE_DST)
    for split in SPLITS:
        b1 = symlink_class(SRC / split / "B1", COARSE_DST / split / "B1")
        b2 = symlink_class(SRC / split / "B2", COARSE_DST / split / "B23", prefix="B2__")
        b3 = symlink_class(SRC / split / "B3", COARSE_DST / split / "B23", prefix="B3__")
        b4 = symlink_class(SRC / split / "B4", COARSE_DST / split / "B4")
        print(f"  {split}: B1={b1}, B23={b2 + b3} (B2={b2}, B3={b3}), B4={b4}")


def build_b23() -> None:
    print(f"\nCreating {B23_DST}...")
    reset_dir(B23_DST)
    for split in SPLITS:
        b2 = symlink_class(SRC / split / "B2", B23_DST / split / "B2")
        b3 = symlink_class(SRC / split / "B3", B23_DST / split / "B3")
        print(f"  {split}: B2={b2}, B3={b3}")


def main():
    build_coarse3()
    build_b23()
    print("\nDone.")


if __name__ == "__main__":
    main()

"""
prepare_dataset.py
------------------
Run this ONCE before training.

What it does:
  1. Reads all images from Color_Images\ and Segmentation\
  2. Matches them by filename stem
  3. Splits into 80% train / 20% val  (reproducible, seed=42)
  4. Copies into the correct folder structure:
       dataset\train\images\
       dataset\train\masks\
       dataset\val\images\
       dataset\val\masks\

Usage (from your project root):
  python prepare_dataset.py

Edit the two paths below if your folders are in a different location.
"""

import os
import shutil
import random
from pathlib import Path

# ── EDIT THESE TWO PATHS TO MATCH WHERE YOU PUT YOUR DOWNLOADED DATA ─────────
COLOR_IMAGES_DIR  = r"C:\Users\ommit\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Color_Images"
SEGMENTATION_DIR  = r"C:\Users\ommit\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Segmentation"
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_ROOT   = "dataset"
TRAIN_SPLIT   = 0.80
RANDOM_SEED   = 42


def main():
    color_dir = Path(COLOR_IMAGES_DIR)
    seg_dir   = Path(SEGMENTATION_DIR)

    if not color_dir.exists():
        raise FileNotFoundError(f"Color_Images folder not found: {color_dir}\nPlease edit COLOR_IMAGES_DIR in this script.")
    if not seg_dir.exists():
        raise FileNotFoundError(f"Segmentation folder not found: {seg_dir}\nPlease edit SEGMENTATION_DIR in this script.")

    color_stems = {p.stem for p in color_dir.glob("*.png")}
    seg_stems   = {p.stem for p in seg_dir.glob("*.png")}
    matched     = sorted(color_stems & seg_stems)

    print(f"Color images   : {len(color_stems)}")
    print(f"Mask images    : {len(seg_stems)}")
    print(f"Matched pairs  : {len(matched)}")

    if len(matched) == 0:
        raise RuntimeError("No matching filenames found between the two folders!")

    random.seed(RANDOM_SEED)
    random.shuffle(matched)
    n_train = int(len(matched) * TRAIN_SPLIT)
    train_stems = matched[:n_train]
    val_stems   = matched[n_train:]
    print(f"Train samples  : {len(train_stems)}")
    print(f"Val samples    : {len(val_stems)}")

    dirs = [
        f"{OUTPUT_ROOT}/train/images",
        f"{OUTPUT_ROOT}/train/masks",
        f"{OUTPUT_ROOT}/val/images",
        f"{OUTPUT_ROOT}/val/masks",
        f"{OUTPUT_ROOT}/testImages",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    def copy_pair(stem, split):
        src_img  = color_dir / f"{stem}.png"
        src_mask = seg_dir   / f"{stem}.png"
        dst_img  = Path(OUTPUT_ROOT) / split / "images" / f"{stem}.png"
        dst_mask = Path(OUTPUT_ROOT) / split / "masks"  / f"{stem}.png"
        shutil.copy2(src_img,  dst_img)
        shutil.copy2(src_mask, dst_mask)

    print("\nCopying train files...")
    for i, stem in enumerate(train_stems):
        copy_pair(stem, "train")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(train_stems)}")

    print("Copying val files...")
    for i, stem in enumerate(val_stems):
        copy_pair(stem, "val")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(val_stems)}")

    print("""
============================================================
  Dataset prepared successfully!
  You can now run:
    conda activate EDU
    python scripts/train.py
============================================================
""")


if __name__ == "__main__":
    main()

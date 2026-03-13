"""
dataset.py – PyTorch Dataset for Offroad Semantic Segmentation
Handles raw label IDs → contiguous class indices remapping.
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random


# ── Label remapping ──────────────────────────────────────────────────────────
RAW_ID_TO_INDEX = {
    100:   0,   # Trees
    200:   1,   # Lush Bushes
    300:   2,   # Dry Grass
    500:   3,   # Dry Bushes
    550:   4,   # Ground Clutter
    600:   5,   # Flowers
    700:   6,   # Logs
    800:   7,   # Rocks
    7100:  8,   # Landscape
    10000: 9,   # Sky
}
NUM_CLASSES = 10
IGNORE_INDEX = 255   # pixels with unknown IDs are ignored in loss


def remap_mask(mask_array: np.ndarray) -> np.ndarray:
    """Convert raw label IDs to 0-based class indices."""
    out = np.full(mask_array.shape, IGNORE_INDEX, dtype=np.uint8)
    for raw_id, idx in RAW_ID_TO_INDEX.items():
        out[mask_array == raw_id] = idx
    return out


# ── Dataset ──────────────────────────────────────────────────────────────────
class OffRoadDataset(Dataset):
    """
    Expects:
        images_dir/  *.png  (RGB)
        masks_dir/   *.png  (single-channel, raw label IDs)
    Image and mask filenames must match (stem).
    """

    def __init__(self, images_dir: str, masks_dir: str,
                 image_size: int = 512, augment: bool = False):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.image_size = image_size
        self.augment    = augment

        # Collect paired files
        img_names = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.samples = []
        for name in img_names:
            stem = os.path.splitext(name)[0]
            # Try .png then .jpg for mask
            for ext in ('.png', '.jpg', '.jpeg'):
                mask_path = os.path.join(masks_dir, stem + ext)
                if os.path.exists(mask_path):
                    self.samples.append((
                        os.path.join(images_dir, name),
                        mask_path
                    ))
                    break

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No paired images/masks found in {images_dir} / {masks_dir}"
            )

        # ImageNet normalisation
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path)

        # ── Resize ────────────────────────────────────────────────────────
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask  = mask.resize ((self.image_size, self.image_size), Image.NEAREST)

        # ── Augmentation ──────────────────────────────────────────────────
        if self.augment:
            image, mask = self._augment(image, mask)

        # ── To tensor ─────────────────────────────────────────────────────
        image_t = TF.to_tensor(image)          # [3,H,W] float32 in [0,1]
        image_t = self.normalize(image_t)

        mask_np  = np.array(mask, dtype=np.int32)
        # Handle RGB masks (take red channel which stores the ID)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        mask_np  = remap_mask(mask_np)
        mask_t   = torch.from_numpy(mask_np).long()

        return image_t, mask_t

    # ── Private helpers ───────────────────────────────────────────────────
    def _augment(self, image: Image.Image, mask: Image.Image):
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # Random crop & resize
        if random.random() > 0.5:
            i, j, h, w = T.RandomCrop.get_params(
                image, output_size=(int(self.image_size * 0.8),
                                    int(self.image_size * 0.8))
            )
            image = TF.crop(image, i, j, h, w)
            mask  = TF.crop(mask,  i, j, h, w)
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            mask  = mask.resize ((self.image_size, self.image_size), Image.NEAREST)

        # Random rotation ±15°
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
            mask  = TF.rotate(mask,  angle, interpolation=Image.NEAREST)

        # Color jitter (image only)
        if random.random() > 0.5:
            jitter = T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            )
            image = jitter(image)

        return image, mask


# ── Test-time dataset (no masks) ─────────────────────────────────────────────
class OffRoadTestDataset(Dataset):
    """Loads only images for inference (no ground-truth masks)."""

    def __init__(self, images_dir: str, image_size: int = 512):
        self.images_dir = images_dir
        self.image_size = image_size
        self.image_paths = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        orig_size = image.size          # (W, H) – saved for later
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_t = TF.to_tensor(image)
        image_t = self.normalize(image_t)
        return image_t, img_path, orig_size

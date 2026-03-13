"""
visualize.py – Visualisation Utilities
Usage: python scripts/visualize.py [--config configs/config.yaml] [--weights checkpoints/model_best.pth]
Produces:
  • Side-by-side comparison panels (image | ground truth | prediction)
  • Class legend
  • TensorBoard loss / IoU plots are already logged during training
"""

import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import torch
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.dataset import OffRoadDataset, NUM_CLASSES, IGNORE_INDEX, remap_mask
from models.segmentation_model import build_model


DEFAULT_PALETTE = [
    [34,  139, 34 ],
    [0,   200, 0  ],
    [210, 180, 140],
    [139, 90,  43 ],
    [160, 82,  45 ],
    [255, 20,  147],
    [101, 67,  33 ],
    [128, 128, 128],
    [194, 178, 128],
    [135, 206, 235],
]


def mask_to_color(mask_np, palette):
    h, w   = mask_np.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, rgb in enumerate(palette):
        colour[mask_np == cls_idx] = rgb
    return colour


def make_legend(class_names, palette):
    patches = [
        mpatches.Patch(
            facecolor=np.array(palette[i]) / 255.0,
            label=class_names[i]
        )
        for i in range(len(class_names))
    ]
    return patches


def visualize_samples(model, dataset, device, use_amp, palette,
                      class_names, out_dir, num_samples=8):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)),
                               replace=False)

    for k, idx in enumerate(indices):
        image_t, mask_t = dataset[idx]
        image_t_b = image_t.unsqueeze(0).to(device)

        with torch.no_grad(), autocast(enabled=use_amp):
            logits = model(image_t_b)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Denormalize image for display
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = image_t.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * std + mean).clip(0, 1)

        gt_np   = mask_t.numpy()
        gt_col  = mask_to_color(gt_np,  palette)
        pred_col = mask_to_color(pred,  palette)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_np);     axes[0].set_title("RGB Image",     fontsize=14)
        axes[1].imshow(gt_col);     axes[1].set_title("Ground Truth",  fontsize=14)
        axes[2].imshow(pred_col);   axes[2].set_title("Prediction",    fontsize=14)
        for ax in axes: ax.axis("off")

        patches = make_legend(class_names, palette)
        fig.legend(handles=patches, loc="lower center",
                   ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"comparison_{k:02d}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"Saved: {save_path}")


def plot_failure_cases(model, dataset, device, use_amp, palette,
                       class_names, out_dir, num_cases=4):
    """
    Find samples where prediction differs most from GT (low per-image IoU)
    and save annotated panels for the failure-case section of the report.
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    scored = []
    for idx in range(len(dataset)):
        image_t, mask_t = dataset[idx]
        with torch.no_grad(), autocast(enabled=use_amp):
            logits = model(image_t.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        gt   = mask_t.numpy().flatten()
        pr   = pred.flatten()
        valid = gt != IGNORE_INDEX
        if valid.sum() == 0:
            continue
        acc = (gt[valid] == pr[valid]).mean()
        scored.append((acc, idx))

    scored.sort(key=lambda x: x[0])          # worst first
    for rank, (acc, idx) in enumerate(scored[:num_cases]):
        image_t, mask_t = dataset[idx]
        with torch.no_grad(), autocast(enabled=use_amp):
            logits = model(image_t.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = (image_t.cpu().numpy().transpose(1, 2, 0) * std + mean).clip(0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_np)
        axes[0].set_title("RGB Image", fontsize=13)
        axes[1].imshow(mask_to_color(mask_t.numpy(), palette))
        axes[1].set_title("Ground Truth", fontsize=13)
        axes[2].imshow(mask_to_color(pred, palette))
        axes[2].set_title(f"Prediction  (acc={acc:.2f})", fontsize=13)
        for ax in axes: ax.axis("off")

        patches = make_legend(class_names, palette)
        fig.legend(handles=patches, loc="lower center",
                   ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Failure Case #{rank+1}", fontsize=15, color="red",
                     fontweight="bold")
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"failure_{rank+1:02d}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"Saved failure case: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--weights", default="checkpoints/model_best.pth")
    parser.add_argument("--num_samples",  type=int, default=8)
    parser.add_argument("--num_failures", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_amp = cfg["mixed_precision"] and device.type == "cuda"

    model = build_model(
        architecture=cfg["model"]["architecture"],
        num_classes=cfg["classes"]["num_classes"],
        backbone=cfg["model"]["backbone"],
        pretrained=False,
    ).to(device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    palette     = cfg["classes"].get("palette", DEFAULT_PALETTE)
    class_names = cfg["classes"]["names"]
    img_size    = cfg["dataset"]["image_size"]

    val_ds = OffRoadDataset(
        cfg["dataset"]["val_images"],
        cfg["dataset"]["val_masks"],
        image_size=img_size,
        augment=False,
    )

    vis_dir  = os.path.join("outputs", "visualizations")
    fail_dir = os.path.join("outputs", "failure_cases")

    print("Generating comparison panels...")
    visualize_samples(model, val_ds, device, use_amp,
                      palette, class_names, vis_dir, args.num_samples)

    print("\nGenerating failure case panels...")
    plot_failure_cases(model, val_ds, device, use_amp,
                       palette, class_names, fail_dir, args.num_failures)

    print("\nAll visualisations saved.")


if __name__ == "__main__":
    main()

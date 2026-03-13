"""
test.py – Inference + Evaluation Pipeline
Usage: python scripts/test.py [--config configs/config.yaml] [--weights checkpoints/model_best.pth]
"""

import os
import sys
import argparse
import json
import time
import yaml
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.dataset import OffRoadTestDataset, OffRoadDataset, NUM_CLASSES, IGNORE_INDEX
from models.segmentation_model import build_model


# ── Colour palette for visualisation ─────────────────────────────────────────
DEFAULT_PALETTE = [
    [34,  139, 34 ],   # Trees
    [0,   200, 0  ],   # Lush Bushes
    [210, 180, 140],   # Dry Grass
    [139, 90,  43 ],   # Dry Bushes
    [160, 82,  45 ],   # Ground Clutter
    [255, 20,  147],   # Flowers
    [101, 67,  33 ],   # Logs
    [128, 128, 128],   # Rocks
    [194, 178, 128],   # Landscape
    [135, 206, 235],   # Sky
]


def mask_to_color(mask_np: np.ndarray, palette) -> np.ndarray:
    """Convert class-index mask [H,W] to RGB image [H,W,3]."""
    h, w   = mask_np.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, rgb in enumerate(palette):
        colour[mask_np == cls_idx] = rgb
    return colour


def compute_iou(preds: np.ndarray, targets: np.ndarray,
                num_classes: int, ignore_index: int = 255):
    iou_per_class = []
    preds   = preds.flatten()
    targets = targets.flatten()
    valid   = targets != ignore_index
    preds, targets = preds[valid], targets[valid]

    for cls in range(num_classes):
        pred_c   = preds   == cls
        target_c = targets == cls
        inter    = (pred_c & target_c).sum()
        union    = (pred_c | target_c).sum()
        iou_per_class.append(inter / union if union > 0 else float('nan'))

    valid_ious = [v for v in iou_per_class if not np.isnan(v)]
    mean_iou   = float(np.mean(valid_ious)) if valid_ious else 0.0
    return iou_per_class, mean_iou


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--weights", default="checkpoints/model_best.pth")
    parser.add_argument("--mode",    default="test",
                        choices=["test", "val"],
                        help="'test'=unseen images (no GT), 'val'=compute IoU")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device  = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_amp = cfg["mixed_precision"] and device.type == "cuda"

    os.makedirs(cfg["paths"]["outputs"],  exist_ok=True)
    os.makedirs(cfg["paths"]["metrics"],  exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    num_classes = cfg["classes"]["num_classes"]
    model = build_model(
        architecture=cfg["model"]["architecture"],
        num_classes=num_classes,
        backbone=cfg["model"]["backbone"],
        pretrained=False,          # weights loaded below
    ).to(device)

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded weights from {args.weights}")
    print(f"Checkpoint epoch: {ckpt.get('epoch','?')} | "
          f"Val mIoU at save: {ckpt.get('val_miou', '?'):.4f}")

    palette     = cfg["classes"].get("palette", DEFAULT_PALETTE)
    class_names = cfg["classes"]["names"]
    img_size    = cfg["dataset"]["image_size"]

    # ── TEST mode – no ground truth ───────────────────────────────────────
    if args.mode == "test":
        dataset = OffRoadTestDataset(
            cfg["dataset"]["test_images"], image_size=img_size
        )
        loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        print(f"\nRunning inference on {len(dataset)} test images...")

        times = []
        for images, img_paths, orig_sizes in loader:
            images = images.to(device)
            t0 = time.time()
            with torch.no_grad(), autocast(enabled=use_amp):
                logits = model(images)
            times.append(time.time() - t0)

            preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()

            # Resize back to original size
            orig_w, orig_h = orig_sizes[0].item(), orig_sizes[1].item()
            pred_img = Image.fromarray(preds.astype(np.uint8))
            pred_img = pred_img.resize((orig_w, orig_h), Image.NEAREST)
            preds_resized = np.array(pred_img)

            # Save colour overlay
            colour = mask_to_color(preds_resized, palette)
            stem   = os.path.splitext(os.path.basename(img_paths[0]))[0]
            out_path = os.path.join(cfg["paths"]["outputs"], stem + "_pred.png")
            Image.fromarray(colour).save(out_path)

        avg_ms = np.mean(times) * 1000
        print(f"Done. Avg inference time: {avg_ms:.1f} ms/image")
        print(f"Predictions saved to: {cfg['paths']['outputs']}")

    # ── VAL mode – compute IoU ────────────────────────────────────────────
    else:
        dataset = OffRoadDataset(
            cfg["dataset"]["val_images"],
            cfg["dataset"]["val_masks"],
            image_size=img_size,
            augment=False,
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
        print(f"\nEvaluating on {len(dataset)} val images...")

        all_preds, all_targets = [], []
        times = []

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(device)
                t0 = time.time()
                with autocast(enabled=use_amp):
                    logits = model(images)
                times.append(time.time() - t0)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(masks.cpu().numpy())

                # Save coloured predictions
                for i in range(images.size(0)):
                    colour = mask_to_color(preds[i].cpu().numpy(), palette)
                    idx    = len(all_preds) * images.size(0) + i
                    out_path = os.path.join(
                        cfg["paths"]["outputs"], f"val_{idx:04d}_pred.png"
                    )
                    Image.fromarray(colour).save(out_path)

        all_preds   = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        iou_per_class, mean_iou = compute_iou(
            all_preds, all_targets, NUM_CLASSES, IGNORE_INDEX
        )
        avg_ms = np.mean(times) * 1000 / 4   # approximate per-image

        print(f"\n{'='*50}")
        print(f"  Mean IoU : {mean_iou:.4f}")
        print(f"  Avg time : {avg_ms:.1f} ms/image")
        print(f"{'='*50}")
        print("\nPer-class IoU:")
        for i, name in enumerate(class_names):
            v = iou_per_class[i]
            bar = "█" * int((v if not np.isnan(v) else 0) * 30)
            if not np.isnan(v):
                print(f"  {name:20s}: {v:.4f}  {bar}")
            else:
                print(f"  {name:20s}: N/A")

        # Save metrics JSON
        metrics = {
            "mean_iou":      round(mean_iou, 4),
            "avg_inference_ms": round(avg_ms, 2),
            "per_class_iou": {
                class_names[i]: round(iou_per_class[i], 4)
                if not np.isnan(iou_per_class[i]) else None
                for i in range(num_classes)
            }
        }
        metrics_path = os.path.join(cfg["paths"]["metrics"], "eval_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

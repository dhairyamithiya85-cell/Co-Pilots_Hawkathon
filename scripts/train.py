"""
train.py – Full Training Pipeline
Usage: python scripts/train.py [--config configs/config.yaml]
"""

import os
import sys
import argparse
import random
import time
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.dataset import OffRoadDataset, NUM_CLASSES, IGNORE_INDEX
from models.segmentation_model import build_model, CombinedLoss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_iou(preds: torch.Tensor, targets: torch.Tensor,
                num_classes: int, ignore_index: int = 255):
    """Returns per-class IoU and mean IoU (ignores ignore_index)."""
    iou_per_class = []
    preds   = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()

    valid = targets != ignore_index
    preds, targets = preds[valid], targets[valid]

    for cls in range(num_classes):
        pred_cls   = preds   == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum()
        union        = (pred_cls | target_cls).sum()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append(intersection / union)

    valid_ious = [v for v in iou_per_class if not np.isnan(v)]
    mean_iou   = float(np.mean(valid_ious)) if valid_ious else 0.0
    return iou_per_class, mean_iou


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    device, use_amp, epoch, writer):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for step, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach())
        all_targets.append(masks.detach())

        if step % 20 == 0:
            print(f"  [Train] Epoch {epoch} | Step {step}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    all_preds   = torch.cat([p.cpu() for p in all_preds])
    all_targets = torch.cat([t.cpu() for t in all_targets])
    _, mean_iou = compute_iou(all_preds, all_targets, NUM_CLASSES, IGNORE_INDEX)

    writer.add_scalar("Loss/train",    avg_loss, epoch)
    writer.add_scalar("mIoU/train",    mean_iou, epoch)
    return avg_loss, mean_iou


# ─────────────────────────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────────────────────────

def validate(model, loader, criterion, device, use_amp, epoch, writer,
             class_names):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                logits = model(images)
                loss   = criterion(logits, masks)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    avg_loss = total_loss / len(loader)
    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    iou_per_class, mean_iou = compute_iou(
        all_preds, all_targets, NUM_CLASSES, IGNORE_INDEX
    )

    writer.add_scalar("Loss/val",  avg_loss, epoch)
    writer.add_scalar("mIoU/val",  mean_iou, epoch)

    # Per-class IoU to TensorBoard
    for cls_idx, cls_iou in enumerate(iou_per_class):
        if not np.isnan(cls_iou):
            writer.add_scalar(
                f"IoU_per_class/{class_names[cls_idx]}", cls_iou, epoch
            )

    print(f"\n  [Val] Epoch {epoch} | Loss: {avg_loss:.4f} | mIoU: {mean_iou:.4f}")
    for i, name in enumerate(class_names):
        v = iou_per_class[i]
        print(f"    {name:20s}: {v:.4f}" if not np.isnan(v) else
              f"    {name:20s}: N/A")
    print()

    return avg_loss, mean_iou, iou_per_class


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ── Seed & device ─────────────────────────────────────────────────────
    set_seed(cfg["seed"])
    device  = torch.device(
        cfg["device"] if torch.cuda.is_available() else "cpu"
    )
    use_amp = cfg["mixed_precision"] and device.type == "cuda"
    print(f"Using device: {device}  |  Mixed precision: {use_amp}")

    # ── Dirs ──────────────────────────────────────────────────────────────
    os.makedirs(cfg["paths"]["checkpoints"], exist_ok=True)
    os.makedirs(cfg["paths"]["logs"],        exist_ok=True)
    os.makedirs(cfg["paths"]["metrics"],     exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────
    img_size = cfg["dataset"]["image_size"]
    aug_cfg  = cfg["training"]["augmentation"]

    train_ds = OffRoadDataset(
        cfg["dataset"]["train_images"],
        cfg["dataset"]["train_masks"],
        image_size=img_size,
        augment=True,
    )
    val_ds = OffRoadDataset(
        cfg["dataset"]["val_images"],
        cfg["dataset"]["val_masks"],
        image_size=img_size,
        augment=False,
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    num_classes = cfg["classes"]["num_classes"]
    model = build_model(
        architecture=cfg["model"]["architecture"],
        num_classes=num_classes,
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    # ── Loss, optimiser, scheduler ────────────────────────────────────────
    criterion = CombinedLoss(
        num_classes=num_classes,
        ce_weight=cfg["training"]["ce_weight"],
        dice_weight=cfg["training"]["dice_weight"],
        ignore_index=IGNORE_INDEX,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    epochs       = cfg["training"]["epochs"]
    warmup_ep    = cfg["training"]["warmup_epochs"]
    scheduler_t  = cfg["training"]["scheduler"]

    def lr_lambda(ep):
        if ep < warmup_ep:
            return (ep + 1) / warmup_ep
        if scheduler_t == "cosine":
            progress = (ep - warmup_ep) / max(1, epochs - warmup_ep)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler(enabled=use_amp)

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=cfg["paths"]["logs"])

    # ── Training loop ─────────────────────────────────────────────────────
    best_miou   = 0.0
    class_names = cfg["classes"]["names"]

    print("\n" + "="*60)
    print("  Starting Training")
    print("="*60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_miou = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, use_amp, epoch, writer
        )
        val_loss, val_miou, _ = validate(
            model, val_loader, criterion,
            device, use_amp, epoch, writer, class_names
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Save checkpoints
        ckpt = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "val_miou":   val_miou,
            "config":     cfg,
        }
        torch.save(ckpt, os.path.join(cfg["paths"]["checkpoints"], "model_last.pth"))

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(ckpt, os.path.join(cfg["paths"]["checkpoints"], "model_best.pth"))
            print(f"  ✓ New best model saved! mIoU = {best_miou:.4f}")

    writer.close()
    print(f"\nTraining complete. Best Val mIoU: {best_miou:.4f}")
    print(f"Best weights saved to: {cfg['paths']['checkpoints']}model_best.pth")


if __name__ == "__main__":
    main()

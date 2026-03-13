# Offroad Semantic Scene Segmentation
**Duality AI Hackathon – Segmentation Track**

---

## Project Structure
```
project_root/
├── configs/
│   └── config.yaml          ← all hyperparameters live here
├── dataset/
│   ├── train/images/
│   ├── train/masks/
│   ├── val/images/
│   ├── val/masks/
│   └── testImages/          ← unseen test images
├── models/
│   └── segmentation_model.py
├── scripts/
│   ├── dataset.py
│   ├── train.py
│   ├── test.py
│   └── visualize.py
├── ENV_SETUP/
│   └── setup_env.bat        ← Windows setup
├── setup_env.sh             ← Mac/Linux setup
├── requirements.txt
├── checkpoints/             ← saved model weights (auto-created)
├── outputs/                 ← predictions & metrics (auto-created)
└── logs/                    ← TensorBoard logs (auto-created)
```

---

## Quick Start

### 1. Create Environment

**Windows (Anaconda Prompt):**
```bat
cd ENV_SETUP
setup_env.bat
```

**Mac / Linux:**
```bash
bash setup_env.sh
```

### 2. Place Dataset
Download the dataset from Falcon and arrange as:
```
dataset/train/images/   ← RGB images
dataset/train/masks/    ← segmentation masks (raw label IDs)
dataset/val/images/
dataset/val/masks/
dataset/testImages/     ← test RGB images (no masks)
```

### 3. Train
```bash
conda activate EDU
python scripts/train.py --config configs/config.yaml
```

Logs stream to TensorBoard:
```bash
tensorboard --logdir logs/
```

### 4. Evaluate on Val Set
```bash
python scripts/test.py --mode val --weights checkpoints/model_best.pth
```
Prints per-class IoU + saves `outputs/metrics/eval_metrics.json`.

### 5. Generate Test Predictions
```bash
python scripts/test.py --mode test --weights checkpoints/model_best.pth
```
Coloured prediction masks are saved to `outputs/predictions/`.

### 6. Visualise
```bash
python scripts/visualize.py --weights checkpoints/model_best.pth
```
Saves comparison panels to `outputs/visualizations/` and failure cases to
`outputs/failure_cases/`.

---

## Classes

| ID    | Class         | Index |
|-------|---------------|-------|
| 100   | Trees         | 0     |
| 200   | Lush Bushes   | 1     |
| 300   | Dry Grass     | 2     |
| 500   | Dry Bushes    | 3     |
| 550   | Ground Clutter| 4     |
| 600   | Flowers       | 5     |
| 700   | Logs          | 6     |
| 800   | Rocks         | 7     |
| 7100  | Landscape     | 8     |
| 10000 | Sky           | 9     |

---

## Model Architecture

**Default:** SegFormer-B0 (pretrained on ADE20K, fine-tuned here)

Alternative architectures configurable in `config.yaml`:
- `deeplabv3plus` – ResNet-101 backbone
- `unet` – lightweight pure-PyTorch UNet

**Loss:** 0.5 × CrossEntropy + 0.5 × Dice  
**Optimizer:** AdamW (lr=1e-4, weight_decay=0.01)  
**Scheduler:** Cosine annealing with 3-epoch warmup  
**Input size:** 512×512  
**Batch size:** 8  
**Epochs:** 50  

---

## Expected Performance

| Metric        | Target   |
|---------------|----------|
| Mean IoU      | ≥ 0.50   |
| Inference time| < 1 s    |
| VRAM usage    | < 8 GB   |

---

## Reproducing Results

1. Clone repository
2. Run `setup_env.sh` / `setup_env.bat`
3. Place dataset under `dataset/`
4. Run `python scripts/train.py`
5. Run `python scripts/test.py --mode val`

All random seeds are fixed (`seed: 42` in config.yaml).

---

## Dependencies

See `requirements.txt`. Key packages:
- `torch >= 2.0`
- `transformers >= 4.35` (SegFormer)
- `torchvision`, `timm`, `opencv-python`, `matplotlib`

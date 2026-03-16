# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HaMeR (Hand Mesh Recovery) is a CVPR 2024 system for 3D hand mesh reconstruction from images using Transformers. It estimates MANO hand model parameters (pose, shape, camera) from cropped hand images via a ViT backbone + transformer decoder architecture.

## Setup & Installation

```bash
# Python 3.10 required
conda create --name hamer python=3.10 && conda activate hamer

# Install PyTorch (customize CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# Install main package
pip install -e .[all]

# Install ViTPose (required for demo/inference)
pip install -v -e third-party/ViTPose

# Download pretrained checkpoints and demo data
bash fetch_demo_data.sh
# MANO model requires registration at https://mano.is.tue.mpg.de
# Place MANO_RIGHT.pkl in _DATA/data/mano/
```

## Key Commands

**Demo / Inference:**
```bash
python demo.py --img_folder example_data --out_folder demo_out --side_view --save_mesh
python video_demo.py --video example_data/video.mp4 --out_folder demo_out
```

**Training:**
```bash
python train.py exp_name=hamer data=mix_all experiment=hamer_vit_transformer trainer=gpu launcher=local
```

**Evaluation:**
```bash
python eval.py --dataset FREIHAND-VAL
python eval.py --dataset 'FREIHAND-VAL,HO3D-VAL,NEWDAYS-TEST-ALL,EPICK-TEST-ALL,EGO4D-TEST-ALL'
```

There is no dedicated test suite; correctness is validated via the evaluation scripts above.

## Architecture

### Inference Pipeline (`demo.py`)
1. Person detector (ViTDet / RegNetY via Detectron2) finds people in the image
2. ViTPose keypoint detector localizes hands
3. Hand regions are cropped and batched
4. HAMER model predicts MANO parameters per hand
5. Results rendered to PNG; optionally exported as OBJ meshes

### Model Architecture (`hamer/models/`)
- **`hamer.py`** — `HAMER(pl.LightningModule)`: top-level model; owns the backbone, MANO head, discriminator, and losses. Key methods: `forward_step()`, `compute_loss()`, `training_step()`, `validation_step()`.
- **`backbones/vit.py`** — ViT backbone (256×192 input, patch=16, dim=1280, depth=32, heads=16). Outputs spatial token grid for cross-attention.
- **`backbones/dinov3.py`** — Alternative DINOv3 backbone.
- **`heads/mano_head.py`** — `MANOTransformerDecoderHead`: transformer decoder with cross-attention over backbone tokens. Predicts pose (48-dim rotation matrices for 16 joints), shape (10-dim betas), and camera (3-dim). Supports iterative error feedback (IEF).
- **`components/pose_transformer.py`** — `TransformerDecoder`, `CrossAttention`, `Attention` primitives used by the MANO head.
- **`mano_wrapper.py`** — Wraps the SMPL-X MANO layer; converts predicted parameters to 3D vertices and keypoints.
- **`discriminator.py`** — Adversarial discriminator for shape regularization during training.

### Configuration System (dual)
- **Hydra** (`hamer/configs_hydra/`) — training experiments, dataset mixes, trainer settings, launcher (local/SLURM). Entry via `train.py`.
- **YACS** (`hamer/configs/`) — model/dataset defaults loaded at runtime for inference and evaluation.

Key experiment config: `hamer/configs_hydra/experiment/hamer_vit_transformer.yaml` — sets LR (`1e-5`), batch size (`8`), 1M total steps, loss weights.

### Datasets (`hamer/datasets/`)
Training uses a weighted mix of ~10 datasets defined in `hamer/configs_hydra/data/mix_all.yaml` (FreiHAND, InterHand2.6M, HO3D, COCO, etc.) plus a MoCap dataset for adversarial training. The `HAMERDataModule` class assembles these.

For inference, `ViTDetDataset` wraps detected bounding boxes into batches for the model.

### Data Directories
- `_DATA/data/mano/` — MANO model files (`MANO_RIGHT.pkl`, `MANO_MEAN_PARAMS.npz`)
- `_DATA/hamer_ckpts/` — downloaded pretrained checkpoints
- `hamer_training_data/` — training archives (tar.gz)
- `hamer_evaluation_data/` — evaluation datasets
- `logs/` — training checkpoints and TensorBoard logs
- `results/` — evaluation CSVs and JSON predictions for benchmark submission

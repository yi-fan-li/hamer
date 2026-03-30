# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HaMeR (Hand Mesh Recovery) is a CVPR 2024 system for 3D hand mesh reconstruction from images using Transformers. It estimates MANO hand model parameters (pose, shape, camera) from cropped hand images via a ViT backbone + transformer decoder architecture.

## Setup & Installation

```bash
# Python 3.10 required
# current environment on this dev computer  
conda activate hamer
conda activate hamer_dino_train 

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

**Training (DINOv3 backbone):**
```bash
python train.py exp_name=hamer data=mix_all experiment=hamer_dinov3_base trainer=gpu launcher=local
```

**Training (original ViT backbone):**
```bash
python train.py exp_name=hamer data=mix_all experiment=hamer_vit_transformer trainer=gpu launcher=local
```

**Training (GCN refinement head on top of frozen DINOv3 HaMeR):**
```bash
python train_gcn.py exp_name=gcn_dinov3b experiment=hamer_gcn trainer=gpu launcher=local \
    GCN.HAMER_CHECKPOINT=_DATA/hamer_ckpts/dinov3/checkpoints/dinov3_base.ckpt
```
Config: `hamer/configs_hydra/experiment/hamer_gcn.yaml`. Only `GCNRefinementHead` parameters are trained; the full HaMeR backbone+head is frozen. If the GCN architecture changes (layer count, hidden dim, input channels), any existing checkpoint becomes incompatible — delete `last.ckpt` and start fresh.

**Evaluation:**
```bash
python eval.py --dataset FREIHAND-VAL
python eval.py --dataset 'FREIHAND-VAL,HO3D-VAL,NEWDAYS-TEST-ALL,EPICK-TEST-ALL,EGO4D-TEST-ALL'
# GCN checkpoint:
python eval.py --dataset FREIHAND-VAL --model_type gcn --checkpoint <gcn_ckpt>
```

**Inference time benchmarking** (separate from eval, uses `--benchmark` flag):
```bash
python eval.py --dataset FREIHAND-VAL --benchmark
```
Reports ms/frame and FPS after a warmup period (first 5 batches discarded). Uses `cuda.synchronize()` for accurate GPU timing.

**Visualization on eval data** (bypasses ViTPose/mmpose — uses pre-computed bboxes from eval dataset):
```bash
python visualize_eval.py --checkpoint logs/train/runs/hamer/checkpoints/dinov3_base.ckpt \
                          --dataset NEWDAYS-TEST-ALL --num_samples 50 --out_folder vis_out_dinov3
```
Saves a side-by-side image per sample: **original | mesh overlay | mesh only**. Add `--side_view` for a fourth panel. Use this instead of `demo.py` when ViTPose is unavailable due to PyTorch version conflicts.

`visualize_eval.py` auto-detects GCN checkpoints by checking for a `GCN` section in `model_config.yaml` and calls `load_hamer_gcn` automatically — no flag needed.

There is no dedicated test suite; correctness is validated via the evaluation scripts above.

**Evaluation dataset paths** must be configured in `hamer/configs/datasets_eval.yaml` before running eval. The file contains hardcoded paths from the original authors (`/home/user/datasets/...`) that need to be updated to match the local machine.

**HOT3D evaluation** (hand-object interaction + occlusion dataset):

Disk usage: only sampled frames are written (not all 440 GB). With `stride=30`, expect ~1-2 GB of images; delete them after eval and keep only the ~50 MB NPZ.

```bash
# Step 0 — set up MANO models (one-time):
mkdir -p /tmp/mano_models
cp /home/yifanli/github/hamer/_DATA/data/mano/MANO_RIGHT.pkl /tmp/mano_models/
cp /home/yifanli/github/HaWoR/_DATA/data/mano/mano_left/MANO_LEFT.pkl /tmp/mano_models/

# Step 1 — preprocess (must use hot3d pixi env: has projectaria_tools + smplx):
/home/yifanli/hot3d/hot3d/.pixi/envs/default/bin/python prepare_hot3d_eval.py \
    --hot3d_root /home/yifanli/hot3d/hot3d/dataset \
    --out_dir /tmp/hot3d_eval \
    --mano_dir /tmp/mano_models \
    --split train --stride 30

# Quick sanity test (one sequence, ~50 MB, no MANO GT):
/home/yifanli/hot3d/hot3d/.pixi/envs/default/bin/python prepare_hot3d_eval.py \
    --hot3d_root /home/yifanli/hot3d/hot3d/dataset \
    --out_dir /tmp/hot3d_eval --sequences P0001_10a27bf7 --stride 30

# Step 2 — update paths in hamer/configs/datasets_eval.yaml HOT3D-VAL entry
#          (script prints exact paths to copy)

# Step 3 — run eval (HaMeR env):
python eval.py --dataset HOT3D-VAL --checkpoint <ckpt>

# Step 4 — reclaim disk space (keep NPZ, delete images):
rm -rf /tmp/hot3d_eval/images/
```
- `prepare_hot3d_eval.py` cannot run in HaMeR's Python 3.13 env; use the hot3d pixi env (Python 3.10).
- GT 2D keypoints require MANO_LEFT.pkl + MANO_RIGHT.pkl in the same directory (see Step 0).
- HOT3D participant splits: train = P0001–P0015 (has GT), test = P0004–P0020 (no GT).
- Metrics: PCK@[0.05, 0.1, 0.15] and mode_kpl2 (same as NEWDAYS/EPICK/EGO4D).

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
- **`backbones/dinov3.py`** — DINOv3 backbone. Loaded via `torch.hub.load` from `third-party/dinov3/`. The `dinov3_vitb16` variant outputs `embed_dim=768` tokens and is configured in `hamer_dinov3_base.yaml`.
- **`heads/mano_head.py`** — `MANOTransformerDecoderHead`: transformer decoder with cross-attention over backbone tokens. Predicts pose (48-dim rotation matrices for 16 joints), shape (10-dim betas), and camera (3-dim). Supports iterative error feedback (IEF).
- **`components/pose_transformer.py`** — `TransformerDecoder`, `CrossAttention`, `Attention` primitives used by the MANO head.
- **`mano_wrapper.py`** — Wraps the SMPL-X MANO layer; converts predicted parameters to 3D vertices and keypoints.
- **`discriminator.py`** — Adversarial discriminator for shape regularization during training.
- **`gcn_refinement.py`** — `GCNRefinementHead`: GCN that refines the 16 MANO joint rotations in **6D representation**. Takes `joints_6d [B,16,6]` + backbone `feat_map [B,C,H,W]` + `joints_2d [B,16,2]` (kinematic joints in `[-0.5,0.5]` space); samples one feature vector per joint via `grid_sample`, concatenates with 6D rotation, projects to hidden dim, runs N GCN layers, predicts a 6D residual. Config: `FEAT_CHANNELS=768`, `HIDDEN_DIM=256`, `NUM_LAYERS=6`. Zero-init on the output head so training starts as identity.
- **`hamer_gcn.py`** — `HAMERWithGCN(pl.LightningModule)`: loads a frozen `HAMER` checkpoint, runs its backbone+MANO head under `torch.no_grad()`, builds 6D pose inputs for the GCN, runs `GCNRefinementHead`, converts refined 6D back to rotation matrices via Gram-Schmidt (`six_d_to_rot_mat`), and re-runs the MANO forward pass to get refined vertices and joints. Both `pred_vertices` and `pred_keypoints_3d` in the output are from the refined MANO pass. Uses `_MANO_KIN_TO_OPENPOSE = [0,5,6,7,9,10,11,17,18,19,13,14,15,1,2,3]` to reorder MANO's OpenPose-ordered joint output into the kinematic order expected by the GCN.

### Configuration System (dual)
- **Hydra** (`hamer/configs_hydra/`) — training experiments, dataset mixes, trainer settings, launcher (local/SLURM). Entry via `train.py`.
- **YACS** (`hamer/configs/`) — model/dataset defaults loaded at runtime for inference and evaluation.

Key experiment configs:
- `hamer/configs_hydra/experiment/hamer_dinov3_base.yaml` — DINOv3-B backbone, `context_dim=768`, LR `1e-5`, batch size `8`, 1M steps.
- `hamer/configs_hydra/experiment/hamer_vit_transformer.yaml` — original ViT backbone config.
- `hamer/configs_hydra/experiment/hamer_gcn.yaml` — GCN refinement head config; `NUM_LAYERS=6`, `HIDDEN_DIM=256`, `FEAT_CHANNELS=768`, 200k steps. `GCN.HAMER_CHECKPOINT` must be set at launch.

### Camera Parametrization
The model predicts a 3-element weak-perspective camera `[s, tx, ty]` via `self.deccam = nn.Linear(dim, 3)` in `mano_head.py`. The focal length (`EXTRA.FOCAL_LENGTH: 5000`) is a **fixed constant**, not learned. Depth is recovered as `tz = 2 * focal / (box_size * s)` in `cam_crop_to_full`. Inaccurate `s` predictions cause the mesh to appear at the wrong scale in the full image.

The ViT backbone uses a non-square crop (`BBOX_SHAPE = [192, 256]`, injected by `load_hamer`). DINOv3 uses square crops (`BBOX_SHAPE` not set). Each model's camera head is calibrated to its own crop convention.

### Evaluation Metrics
- **MPJPE / PA-MPJPE (mm)**: 3D joint errors after wrist alignment only, and after wrist alignment + Procrustes (similarity transform), respectively. Reported as `mode_mpjpe` / `mode_re`.
- **PCK@threshold**: fraction of 2D keypoints within `threshold × box_size` of GT, in image space.
- **mode_kpl2**: mean MSE of 2D projected keypoints, confidence-weighted.
- Training uses **confidence-weighted L1** for 2D keypoints; eval uses MSE — these can diverge when error distributions differ (e.g. a model with fewer outliers but more uniformly mediocre predictions can have lower kpl2 but worse PCK).
- The `'right'` key in dataset npz files indicates handedness; falls back to all-ones (all right hands) if missing.

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

### Known Issues & Fixes

**DataLoader worker crashes (segfault / ConnectionResetError):** Caused by `breakpoint()` calls in dataset processing code firing inside worker processes (no TTY). All `breakpoint()` calls in `hamer/datasets/image_dataset.py` and `hamer/datasets/utils.py` have been replaced with `raise ValueError(...)`. The `process_webdataset_tar_item` method catches these, logs a warning with the sample key, and returns `None`; a `.select(lambda x: x is not None)` filter in the webdataset pipeline skips bad samples.

**`model_config.yaml` MANO path interpolation:** Hydra saves `${MANO.DATA_DIR}/mano_mean_params.npz` style OmegaConf references into `model_config.yaml`, but eval loads this with YACS which does not resolve them. Fix: set `MANO.MODEL_PATH` and `MANO.MEAN_PARAMS` to plain relative paths (`data/mano`, `data/mano_mean_params.npz`) in the saved config, matching the pretrained checkpoint's config format. The `get_config(..., update_cachedir=True)` call then correctly prefixes them with `./_DATA/`.

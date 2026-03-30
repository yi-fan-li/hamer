# GCN Residual Plan

> **Status (2026-03-30):** v2 is fully implemented and is the current architecture.
> v1 (21 joint xyz refinement) has been replaced. The files `gcn_refinement.py` and `hamer_gcn.py`
> now implement v2. See v2 sections below for the current dataflow and architecture details.

---

## Motivation

Add a GCN at the end of the HaMeR pipeline to correct the 21 joint positions with a residual.
Targets the loss of implicit positional bias when training with a smaller backbone (DINOv3-B).
HaMeR weights are fully frozen; only the GCN is trained.

---

## Dataflow (v2 — current)

```
frozen HaMeR → pred_mano_params (global_orient [B,1,3,3], hand_pose [B,15,3,3], betas)
                         ↓
              coarse MANO forward → 21 joint positions [B,21,3] (OpenPose order)
                         ↓
              reorder via _MANO_KIN_TO_OPENPOSE → 16 kinematic joints [B,16,3]
                         ↓
              weak-perspective project → 16 (u,v) locations [B,16,2]
                         ↓
              index into backbone feature map via grid_sample → 16 × C vectors
                         ↓
              stack global_orient + hand_pose → [B,16,3,3] → rot_mat_to_6d → [B,16,6]
                         ↓
              concat with sampled features → 16 × (C + 6)
                         ↓
              shared MLP (Linear → LayerNorm → GELU) → 16 × d
                         ↓
              6 × GCN layers (MANO kinematic adjacency) → 16 × d
                         ↓
              linear head (zero-init) → 16 × 6 delta
                         ↓
              add to coarse 6D → refined 16 × 6
                         ↓
              six_d_to_rot_mat (Gram-Schmidt) → [B,16,3,3]
                         ↓
              refined MANO forward (refined pose + original betas)
                         ↓
              refined vertices [B,778,3]  +  refined joints [B,21,3]
```

---

## Architecture Details

### GCN layer
- Standard spectral graph conv: `H' = σ(A_hat @ H @ W)`
- `A_hat = D^{-1/2} (A + I) D^{-1/2}` — fixed buffer, no gradient

### Hand skeleton adjacency (16 joints, MANO kinematic ordering)
```
Joint order:  0=wrist, 1-3=index, 4-6=middle, 7-9=little, 10-12=ring, 13-15=thumb
Edges:  0-1, 1-2, 2-3          # index
        0-4, 4-5, 5-6          # middle
        0-7, 7-8, 8-9          # little/pinky
        0-10, 10-11, 11-12     # ring
        0-13, 13-14, 14-15     # thumb
```

MANO's `.joints` output is in OpenPose order; `_MANO_KIN_TO_OPENPOSE = [0,5,6,7,9,10,11,17,18,19,13,14,15,1,2,3]` maps kinematic index → OpenPose index to recover joints in the correct GCN order.

### Hyperparameters
- `C = 768` (DINOv3-B feature channels)
- `d = 256` (GCN hidden dim)
- **6 GCN layers**
- Input MLP: `(C + 6) → d` in one step (6D rotation replaces xyz)

---

## Feature Map Indexing — Coordinate System (verified)

Backbone receives `x[:,:,:,32:-32]`:
- Input image: **256 × 256**
- After crop: **256 × 192** (32px removed from each side of width)
- DINOv3-B output (patch_size=16): feature map **[B, 768, 16, 12]**

`pred_keypoints_2d` are in the same space as `gt_keypoints_2d`:
- Normalized as `pixel / 256 - 0.5` → range **[-0.5, 0.5]** centered at 0

The feature map covers:
- width: pixels [32, 224] → normalized u ∈ [-0.375, 0.375]
- height: pixels [0, 256]  → normalized v ∈ [-0.5, 0.5]

Conversion from `pred_2d` ([-0.5, 0.5]) to `grid_sample` coords ([-1, 1]):
```python
gs_u = pred_u / 0.375   # = (8/3) * pred_u
gs_v = pred_v / 0.5     # = 2 * pred_v
```

Derivation for u:
```
pred_u_pixel = (pred_u + 0.5) * 256
crop_u       = pred_u_pixel - 32          # relative to cropped image
gs_u         = crop_u / 96 - 1           # normalize to [-1,1]
             = ((pred_u + 0.5)*256 - 32) / 96 - 1
             = (256*pred_u + 96) / 96 - 1
             = (8/3)*pred_u              ✓
```

Out-of-bounds handling: `padding_mode='zeros'` (default), consistent with HaMeR's approach.

---

## Training

### What is frozen vs trained
- **Frozen**: entire HaMeR model (backbone + MANO head + MANO layer), regardless of which backbone variant is used
- **Trainable**: `GCNRefinementHead` only

### Loss functions
- `Keypoint3DLoss` (L1) on refined joints — same as HaMeR training
- `Keypoint2DLoss` (L1) on reprojected refined joints — same as HaMeR training
- No MANO parameter loss (GCN doesn't output MANO params)
- Optional: small L2 regularisation on the delta to keep residuals small (TBD)

### Optimizer
- AdamW, `lr=1e-5`, `weight_decay=1e-4` — same as HaMeR

---

## Output Dict Changes

The GCN produces refined outputs for all three:
- `pred_keypoints_3d` — from refined MANO forward pass
- `pred_keypoints_2d` — reprojection of refined joints
- `pred_vertices` — from refined MANO forward pass (unlike v1, vertices are also refined)
- `coarse_keypoints_3d` / `coarse_vertices` — coarse MANO outputs kept for debugging

---

## Files to Create / Modify

### New files
| File | Purpose |
|------|---------|
| `hamer/models/gcn_refinement.py` | `GCNLayer`, `GCNRefinementHead` |
| `hamer/models/hamer_gcn.py` | `HAMERWithGCN` Lightning module |
| `hamer/configs_hydra/experiment/hamer_gcn.yaml` | Training config |
| `train_gcn.py` | Training entry point |

### Modified files
| File | Change |
|------|--------|
| `hamer/models/__init__.py` | Export `HAMERWithGCN`; add `load_hamer_gcn()` loader |
| `visualize_eval.py` | Auto-detects GCN checkpoint (checks for `GCN` section in `model_config.yaml`), calls `load_hamer_gcn` automatically |

---

## Eval Notes

- `eval.py` supports GCN via `--model_type gcn` flag (calls `load_hamer_gcn` instead of `load_hamer`)
- `visualize_eval.py` auto-detects GCN checkpoints (no flag needed)
- Both `pred_vertices` and `pred_keypoints_3d` are refined in the GCN output

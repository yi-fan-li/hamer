# GCN Residual Plan

> **Note (2026-03-27):** The original design (v1, implemented in `gcn_refinement.py` / `hamer_gcn.py`)
> refines the **21 3D keypoints** in Cartesian space. The plan is being revised (v2) to instead
> refine the **16 MANO joint rotations** (6D representation) directly, after which the refined
> pose + original shape betas are passed through the MANO forward pass to recover the full mesh.
> This is more architecturally principled: corrections propagate through the kinematic chain and
> the model stays within the MANO parameter space. The v1 files remain in the repo for reference.

---

## Motivation

Add a GCN at the end of the HaMeR pipeline to correct the 21 joint positions with a residual.
Targets the loss of implicit positional bias when training with a smaller backbone (DINOv3-B).
HaMeR weights are fully frozen; only the GCN is trained.

---

## Dataflow

```
MANO params → FK → 21 joint positions in 3D
                         ↓
              weak-perspective project → 21 (u,v) locations
                         ↓
              index into backbone feature map → 21 × C vectors  (grid_sample)
                         ↓
              concat with 3D coords → 21 × (C + 3)
                         ↓
              shared MLP (Linear → LayerNorm → GELU) → 21 × d
                         ↓
              3 × GCN layers (skeleton adjacency) → 21 × d
                         ↓
              linear head → 21 × 3 delta
                         ↓
              add to coarse joints → refined 21 × 3
```

---

## Architecture Details

### GCN layer
- Standard spectral graph conv: `H' = σ(A_hat @ H @ W)`
- `A_hat = D^{-1/2} (A + I) D^{-1/2}` — fixed buffer, no gradient

### Hand skeleton adjacency (21 joints, OpenPose ordering)
```
Edges:  0-1, 1-2, 2-3, 3-4        # thumb
        0-5, 5-6, 6-7, 7-8        # index
        0-9, 9-10, 10-11, 11-12   # middle
        0-13, 13-14, 14-15, 15-16 # ring
        0-17, 17-18, 18-19, 19-20 # pinky
```

### Hyperparameters
- `C = 768` (DINOv3-B feature channels)
- `d = 256` (GCN hidden dim, to decide)
- 3 GCN layers
- Input MLP: `(C + 3) → d` in one step (no intermediate projection)

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

The GCN replaces `pred_keypoints_3d` and `pred_keypoints_2d` with refined versions.
`pred_vertices` remains the coarse MANO mesh (no vertex refinement).

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
| `hamer/models/__init__.py` | Export `HAMERWithGCN` |
| `eval.py` | Add `--model_type` flag to optionally load `HAMERWithGCN` |

---

## Eval Notes

- `eval.py` currently calls `load_hamer()` which returns a `HAMER` instance
- For GCN eval, add a `--model_type gcn` flag and a `load_hamer_gcn(ckpt, hamer_ckpt)` loader
- `pred_vertices` in output will be coarse (from frozen MANO); `pred_keypoints_3d` will be refined
- FreiHAND/HO3D submit vertices for online eval — these will be unrefined; joint metrics will reflect GCN improvement

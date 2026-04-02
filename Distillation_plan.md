# Plans to implement teacher-student distillation with VIT-Huge and VIT-base

## * Backbone distillation

---

## Experiment 2: Distill ViT-H attention maps → DINOv3-B student (planned)

### Motivation

The current running experiment (Exp 1) distills ViT-H → ViT-B (vitpose_base). This experiment instead uses the DINOv3-B backbone as the student. DINOv3-B was already trained for HaMeR independently (`dinov3_base.ckpt`) and is the stronger backbone. Attention distillation from the ViT-H teacher may close the remaining gap further, particularly on out-of-distribution hands.

### What changes vs Exp 1

| | Exp 1 (running) | Exp 2 (this plan) |
|---|---|---|
| Student backbone | `vitpose_base` | `dinov3_vitb16` |
| Student embed_dim | 768 | 768 |
| Student layers | 12 | 12 |
| Student heads | 12 | 12 |
| Crop convention | non-square `[192, 256]` | non-square `[192, 256]` (same — see below) |
| Student init | random MANO head | `dinov3_base.ckpt` (strongly recommended) |
| Config | `hamer_distill.yaml` | new `hamer_distill_dino.yaml` |

Layer pairing is identical (12 student layers → 12 paired teacher layers out of 32), and head assignment via Hungarian is identical.

### Crop convention — no change needed

DINOv3 uses **RoPE (Rotary Position Embedding)** instead of fixed sinusoidal positional embeddings. RoPE is computed from per-patch 2D coordinates at runtime, so it handles arbitrary aspect ratios natively — no modification to the model is needed. With patch_size=16 and a 192×256 input, DINOv3 produces exactly `12×16 = 192` spatial tokens, identical to the teacher. This eliminates the token-count mismatch entirely.

Keep `x_cropped = x[:, :, :, 32:-32]` (same crop used for the teacher) and pass it directly to both backbones.

### DINOv3 prefix tokens — mask before distillation

DINOv3-B prepends **5 prefix tokens** to the patch sequence in every attention layer:
- index 0: CLS token
- indices 1–4: 4 storage tokens (`n_storage_tokens=4` in `dinov3_vitb16`)

These tokens absorb global/noise signal and have no spatial correspondence with the teacher's purely spatial attention maps. Including them in the KL loss would add meaningless noise.

**Fix:** after extracting DINOv3 attention maps `[B, H, N_total, N_total]` where `N_total = 5 + 192 = 197`, slice out the spatial submatrix:
```python
NUM_PREFIX = 5  # 1 cls + 4 storage
student_spatial_attn = attn[:, :, NUM_PREFIX:, NUM_PREFIX:]  # [B, H, 192, 192]
```
Row-renormalise after slicing so each query distribution sums to 1:
```python
student_spatial_attn = student_spatial_attn / (student_spatial_attn.sum(dim=-1, keepdim=True) + 1e-8)
```
The teacher attention maps are already purely spatial (ViT-H has no CLS/register tokens in its sequence — pos embed is added as `pos_embed[:, 1:] + pos_embed[:, :1]` without prepending a CLS token to the sequence), so teacher maps require no slicing.

### Adding `set_store_attn` / `get_attn_maps` to DINOv3Backbone

The current `DINOv3Backbone` wrapper (`hamer/models/backbones/dinov3.py`) does not expose these methods — they only exist on the ViT backbone. Need to add them:
- `set_store_attn(layer_indices)`: iterate over `self.model.blocks`, set `blk.attn.store_attn = True/False`.
- `get_attn_maps(layer_indices)`: return `[blk.attn._attn_weights for i, blk in ... if i in layer_indices]`.
- In the attention module inside DINOv3's blocks, hook in the same `store_attn / _attn_weights` pattern used in `hamer/models/backbones/vit.py:140`. Check whether DINOv3's `SelfAttentionBlock` exposes post-softmax weights; if not, register a forward hook on `blk.attn` to capture them.

### Implementation steps

1. **New experiment config** `hamer/configs_hydra/experiment/hamer_distill_dino.yaml`:
   - Set `MODEL.BACKBONE.TYPE: dinov3_vitb16`
   - Keep `MODEL.BBOX_SHAPE: [192, 256]` (same non-square crop as teacher)
   - Set `DISTILL.STUDENT_CHECKPOINT: _DATA/hamer_ckpts/dinov3/checkpoints/dinov3_base.ckpt`
   - Set `MODEL.MANO_HEAD.TRANSFORMER_DECODER.context_dim: 768`

2. **Add attention storage to `DINOv3Backbone`** (`hamer/models/backbones/dinov3.py`):
   - Mirror the `set_store_attn` / `get_attn_maps` interface from `vit.py`.
   - If DINOv3's internal attention module does not expose a `store_attn` flag, use a `register_forward_hook` to capture the post-softmax weight tensor.

3. **`distilled_hamer.py` — no changes needed.** Prefix-token slicing and renormalization are handled inside `DINOv3Backbone.get_attn_maps()`, which returns purely spatial maps `[B, H, 192, 192]` — the same shape the teacher returns. The crop logic (`x_cropped = x[:, :, :, 32:-32]`) and all distillation code are identical.

4. **Training command:**
   ```bash
   python train_distill.py exp_name=distill_dino experiment=hamer_distill_dino trainer=gpu launcher=local \
       DISTILL.TEACHER_CHECKPOINT=_DATA/hamer_ckpts/checkpoints/hamer.ckpt \
       DISTILL.STUDENT_CHECKPOINT=_DATA/hamer_ckpts/dinov3/checkpoints/dinov3_base.ckpt
   ```

5. **λ tuning:** same procedure as Exp 1 — after ~100 steps check `train/loss_task` vs `train/loss_distill` in TensorBoard and adjust `DISTILL.LOSS_WEIGHT` so `λ * loss_distill ≈ loss_task`.

### Expected outcome

Since DINOv3-B is already a strong student (trained independently), the distillation signal should be a fine-tuning signal rather than training-from-scratch. Expect `loss_task` to start low and `loss_distill` to drive subtle attention re-alignment. Monitor PA-MPJPE on FreiHAND-VAL and compare to the standalone `dinov3_base.ckpt` baseline.

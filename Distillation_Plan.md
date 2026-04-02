# Plans to implement teacher-student distillation with VIT-Huge and VIT-base
## Backbone distillation
The goal is to distill the attention map of the teacher backbone (ViT-Huge, which will be frozen) to student (ViT-Base). The idea would be to make the student attention head learn the location where it should pay attention to on an image. To do so, we will need to access the result of the q, k, v values of the teacher model, and learn it using a KL divergence loss on the student network. Call the student network distilled_hamer.

## Design of distillation model of backbone:
The teacher model backbone is a ViT-H, with 32 layers and 16 attention head. They will be mapped to the student model backbone, a ViT-B, with 12 layers and 12 attention heads. The model will be trained with uniform spacing for the layers: round(l × 32/12). Possible future change is to have the model learn more from the final few layers (map last 12 teacher layers to last 6 student layer, to capture more detail)

For the mismatch in attention heads, go for a nearest neighbor assignment for now. So for any two pair of heads, compare their similarity (average cosine similarity over the 192×192 maps over a batch?), and assign each student head to track the teacher head most similar to them (but only 1 to 1 match, no 1 to many or many to 1 match.)

## Implementation details:

### Attention Map Extraction

The `Attention` class in `hamer/models/backbones/vit.py` currently discards the post-softmax attention weights after computing the output. We need to expose them. The cleanest approach is to add a `store_attn: bool` flag to `Attention.__init__` (default `False`) and store `self._attn_weights` in `forward()` after the softmax and before dropout:

```python
attn = attn.softmax(dim=-1)
if self.store_attn:
    self._attn_weights = attn  # [B, num_heads, N, N]
```

Add a helper method `ViT.set_store_attn(enabled: bool)` that toggles this flag on all blocks, and `ViT.get_attn_maps() -> list[Tensor]` that returns `block.attn._attn_weights` for all blocks (length 32 for ViT-H, 12 for ViT-B). The teacher only needs to store maps at the 12 paired layers to avoid unnecessary memory overhead.

Attention maps have shape `[B, num_heads, 128, 128]` during training. Both backbones receive the same `x[:,:,:,32:-32]` crop (256×192 → 256×128), which with patch_size=16 and padding=4 yields 16×8 = 128 tokens. The plan originally said 192 tokens (from the full 256×192 input), but the crop reduces width from 12 to 8 patches.

### Layer Pairing

Uniform spacing formula but forcing the first and last layer to be the same for student and teacher:

| Student layer | Teacher layer |
|:---:|:---:|
| 0 | 0 |
| 1 | 3 |
| 2 | 5 |
| 3 | 8 |
| 4 | 11 |
| 5 | 14 |
| 6 | 17 |
| 7 | 20 |
| 8 | 23 |
| 9 | 26 |
| 10 | 29 |
| 11 | 31 |

Stored as the module-level constant `LAYER_PAIRS: list[tuple[int, int]]` in `hamer/models/distilled_hamer.py`.

### Head Assignment

The assignment is a bipartite matching problem: 12 student heads must be matched 1-to-1 to 12 of the 16 teacher heads (4 teacher heads will be unmatched and ignored). Use the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) on the cost matrix, where the cost of matching student head `s` to teacher head `t` at a given layer pair is the **negative average cosine similarity** of their `[B, 192, 192]` attention maps, flattened to `[B, 192*192]` and averaged over the batch.

The assignment is computed **lazily on the first training batch**. It is stored as a registered buffer `head_assignments: Tensor[12, 12]` (layer pairs × student heads) alongside a bool buffer `assignment_computed`. Because it is a registered buffer, it is saved in the checkpoint and restored automatically — no re-computation on resume. A companion `assignment_computed` flag prevents re-running the Hungarian step after a resume.

Re-computing the assignment periodically (e.g., every 10k steps) is a potential future improvement but starts with a fixed assignment.

### KL Divergence Loss

For each layer pair `(l_s, l_t)` and each matched head pair `(s, t)`:
- Let `A_T ∈ R^{B×192×192}` be the teacher attention (each row is a probability distribution over 192 key tokens for one query token).
- Let `A_S` be the corresponding student attention.
- Loss per pair: `KL(A_T ∥ A_S) = Σ_{query} Σ_{key} A_T * (log A_T - log A_S)`, summed over query tokens, averaged over batch.

Total distillation loss: average over all 12 layer pairs and their matched heads.

The stored post-softmax maps are used directly for the KL loss (no temperature scaling for now — adding temperature would require storing the pre-softmax `q@k.T` logits instead, which is a straightforward future extension if needed).

The implementation vectorises the inner loop: for each layer pair the matched teacher heads are gathered in one index operation (`t_maps[:, assignment, :, :]`), then both teacher and student tensors are reshaped to `[B×S_H×N, N]` so a single `F.kl_div(..., reduction='batchmean')` covers all heads and query positions at once.

A small epsilon (`+1e-8`) is added to the student maps before `log()` to avoid numerical instability at zero.

The combined training loss is:
```
L_total = L_task + λ * L_distill
```
where `L_task` is the standard HaMeR supervised loss (2D/3D keypoint + shape + camera + adversarial losses) and `λ` is `DISTILL.LOSS_WEIGHT`. The task loss keeps the student useful for hand reconstruction while the distillation loss shapes its attention.

Both `L_task` and `L_distill` are logged separately to TensorBoard as `train/loss_task` and `train/loss_distill` at every `LOG_STEPS` interval. On the first run, set `DISTILL.LOSS_WEIGHT: 1e-3`, inspect the two logged values after ~100 steps, and adjust `λ` so that `λ * L_distill` is in the same order of magnitude as `L_task`. Only then does tuning `λ` have a meaningful effect.

### DistilledHAMER Class

`DistilledHAMER(pl.LightningModule)` in `hamer/models/distilled_hamer.py`:

```
Teacher (HAMER, ViT-H, frozen):
  - Loaded via HAMER.load_from_checkpoint(..., init_renderer=False)
  - All parameters set requires_grad=False, model kept in eval() mode
  - set_store_attn() called with the 12 paired teacher layer indices only

Student:
  - ViT-B backbone via create_backbone(cfg) — loads ViTPose-Base pretrained weights
  - MANOTransformerDecoderHead with context_dim=768, randomly initialised
  - set_store_attn() called with all 12 student layer indices

forward_step(batch, train=False):
  - When train=True:
    1. Run teacher backbone under torch.no_grad() to populate stored attn maps
    2. Run student backbone (with grad), collect student attn maps
    3. On first batch: compute Hungarian head assignment and store in buffers
    4. Compute L_distill via KL divergence
  - Always: run student MANO head, build standard output dict

training_step:
  1. Call forward_step(batch, train=True)
  2. Compute L_task (keypoints + MANO params)
  3. Compute adversarial loss against discriminator
  4. L_total = L_task + λ * L_distill
  5. Manual backward (automatic_optimization=False, same as HAMER)
  6. Run discriminator training step

validation_step:
  - Calls forward_step(batch, train=False) — teacher is NOT run, no distillation loss
```

The MANO model instance is constructed fresh from the student cfg (same model, same weights, avoids sharing state with the frozen teacher). The discriminator and adversarial loss are kept exactly as in the base `HAMER` class.

### Training Config

`hamer/configs_hydra/experiment/hamer_distill.yaml` (inherits `default.yaml`):
- `MODEL.BACKBONE.TYPE: vitpose_base`
- `MODEL.MANO_HEAD.TRANSFORMER_DECODER.context_dim: 768`
- `DISTILL.TEACHER_CHECKPOINT: ???`  (must be supplied on the command line)
- `DISTILL.STUDENT_CHECKPOINT: ???`  (optional but strongly recommended — initialises student backbone + MANO head from a trained ViT-B HaMeR checkpoint instead of random MANO head init)
- `DISTILL.LOSS_WEIGHT: 1e-3`
- `DISTILL.HEAD_ASSIGN_BATCHES: 8`  (8 batches × batch_size images averaged before Hungarian)
- `TRAIN.LR: 1e-5`
- `GENERAL.TOTAL_STEPS: 500_000`

Usage:
```bash
python train_distill.py exp_name=distill_vitb experiment=hamer_distill trainer=gpu launcher=local \
    DISTILL.TEACHER_CHECKPOINT=_DATA/hamer_ckpts/checkpoints/hamer.ckpt \
    DISTILL.STUDENT_CHECKPOINT=_DATA/hamer_ckpts/hamer_base/checkpoints/hamer_base.ckpt
```

### Training Script

`train_distill.py` mirrors `train_gcn.py` exactly, replacing `HAMERWithGCN` with `DistilledHAMER`. The teacher checkpoint path is read from `cfg.DISTILL.TEACHER_CHECKPOINT` inside `DistilledHAMER.__init__`; the training script itself does not need to know about it.

## Files added/modified

| Action | File | What changed |
|--------|------|--------------|
| **Modified** | `hamer/models/backbones/vit.py` | `Attention.__init__`: added `self.store_attn = False` and `self._attn_weights = None`. `Attention.forward()`: stores post-softmax attn when flag is set. `ViT`: added `set_store_attn(layer_indices)` and `get_attn_maps(layer_indices)`. |
| **Added** | `hamer/models/distilled_hamer.py` | `DistilledHAMER` LightningModule; `LAYER_PAIRS` constant; `_compute_sim_matrices()`, `_assign_from_sim_matrices()` (head assignment accumulated over `HEAD_ASSIGN_BATCHES` batches); `_compute_distill_loss()`; `compute_task_loss()`; optional student init from `DISTILL.STUDENT_CHECKPOINT`; full training/validation loop. |
| **Added** | `hamer/configs_hydra/experiment/hamer_distill.yaml` | Experiment config with `DISTILL` section. |
| **Added** | `train_distill.py` | Training entry point (mirrors `train_gcn.py`). |
| **Modified** | `hamer/models/__init__.py` | Added `from .distilled_hamer import DistilledHAMER`. |
| **Modified** | `CLAUDE.md` | Added distillation training command and notes. |
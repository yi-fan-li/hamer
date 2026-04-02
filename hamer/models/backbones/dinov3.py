import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# DINOv3-B prefix-token count: 1 CLS token + n_storage_tokens (4 for dinov3_vitb16).
# Attention maps have shape [B, H, N_total, N_total] where the first NUM_PREFIX_TOKENS
# rows/cols are CLS + storage tokens.  get_attn_maps() slices these out automatically
# so callers always receive purely spatial attention maps.
_DEFAULT_NUM_PREFIX_TOKENS = 5  # 1 CLS + 4 storage (n_storage_tokens=4 in dinov3_vitb16)


def _patch_attn_module(attn_module) -> None:
    """Monkey-patch a DINOv3 SelfAttention module to support optional attention storage.

    Adds two attributes:
      - store_attn (bool): when True, the next forward pass stores post-softmax
        attention weights in _attn_weights.
      - _attn_weights (Tensor | None): cached [B, H, N, N] after the last forward
        with store_attn=True.

    When store_attn=True, compute_attention uses an explicit softmax instead of
    SDPA so the weights can be captured.  When False, the original SDPA path is
    used unchanged.
    """
    attn_module.store_attn = False
    attn_module._attn_weights = None

    def _compute_attention_with_storage(qkv, attn_bias=None, rope=None):
        B, N, _ = qkv.shape
        C = attn_module.qkv.in_features
        H = attn_module.num_heads
        D = C // H

        qkv_r = qkv.reshape(B, N, 3, H, D)
        q, k, v = torch.unbind(qkv_r, 2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if rope is not None:
            q, k = attn_module.apply_rope(q, k, rope)

        if attn_module.store_attn:
            # Explicit softmax path: materialize attention weights for storage.
            scale = D ** -0.5
            attn_w = torch.softmax(
                q.float() @ k.float().transpose(-2, -1) * scale, dim=-1
            )
            attn_module._attn_weights = attn_w.detach()
            x = attn_w.to(v.dtype) @ v
        else:
            x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2)
        return x.reshape(B, N, C)

    attn_module.compute_attention = _compute_attention_with_storage


class DINOv3Backbone(nn.Module):
    def __init__(self, model, patch_size=16, num_unfrozen_layers=0):
        super().__init__()
        self.model = model
        self.patch_size = patch_size

        # Number of non-spatial prefix tokens prepended by DINOv3 (CLS + storage tokens).
        # Derived from the model so it stays correct if n_storage_tokens changes.
        self._num_prefix_tokens: int = 1 + getattr(model, 'n_storage_tokens', _DEFAULT_NUM_PREFIX_TOKENS - 1)

        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last num_unfrozen_layers transformer blocks
        if num_unfrozen_layers > 0:
            blocks = list(self.model.blocks)
            for block in blocks[-num_unfrozen_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
            # Also unfreeze the final norm layer
            if hasattr(self.model, 'norm'):
                for param in self.model.norm.parameters():
                    param.requires_grad = True

        n_frozen = sum(1 for p in self.model.parameters() if not p.requires_grad)
        n_unfrozen = sum(1 for p in self.model.parameters() if p.requires_grad)
        print(f"DINOv3 backbone: {n_unfrozen} unfrozen params, {n_frozen} frozen params "
              f"(num_unfrozen_layers={num_unfrozen_layers})")

        # Patch every SelfAttention module so set_store_attn / get_attn_maps work.
        for blk in self.model.blocks:
            _patch_attn_module(blk.attn)

    def forward(self, x):
        B, C, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size
        features = self.model.forward_features(x)
        # patch tokens: [B, h*w, embed_dim]
        patch_tokens = features["x_norm_patchtokens"]
        embed_dim = patch_tokens.shape[-1]
        # reshape to [B, embed_dim, h, w] to match ViT backbone output format
        return patch_tokens.reshape(B, h, w, embed_dim).permute(0, 3, 1, 2)

    # ---------------------------------------------------------------------- #
    # Attention storage interface (mirrors hamer/models/backbones/vit.py)     #
    # ---------------------------------------------------------------------- #

    def set_store_attn(self, layer_indices: List[int]) -> None:
        """Enable attention-map storage for the given layer indices, disable for all others.

        After a forward pass, stored maps are accessible via get_attn_maps().
        Only enable on layers you need — the explicit softmax path has higher
        memory cost than SDPA.

        Args:
            layer_indices: list of block indices (0-based) to enable storage on.
        """
        active = set(layer_indices)
        for i, blk in enumerate(self.model.blocks):
            blk.attn.store_attn = (i in active)

    def get_attn_maps(self, layer_indices: List[int]) -> List[torch.Tensor]:
        """Return post-softmax spatial attention maps cached by the last forward pass.

        Prefix tokens (CLS + storage) are sliced off automatically so the
        returned tensors contain only patch-to-patch attention:
            [B, num_heads, num_patches, num_patches]

        Must be called after set_store_attn() and a forward pass.

        Args:
            layer_indices: list of block indices whose maps to return, in order.

        Returns:
            List of tensors, one per requested layer, each [B, H, N_spatial, N_spatial].
        """
        p = self._num_prefix_tokens
        maps = []
        for i in layer_indices:
            attn = self.model.blocks[i].attn._attn_weights  # [B, H, N_total, N_total]
            if attn is None:
                raise RuntimeError(
                    f'No attention map cached for layer {i}. '
                    'Call set_store_attn() before the forward pass.'
                )
            # Slice to spatial tokens only and renormalise rows to sum to 1.
            spatial = attn[:, :, p:, p:]  # [B, H, N_spatial, N_spatial]
            spatial = spatial / (spatial.sum(dim=-1, keepdim=True) + 1e-8)
            maps.append(spatial)
        return maps


def dinov3(cfg):
    DINOV3_LOCATION = "/home/yifanli/github/hamer/third-party/dinov3"

    print(f"DINOv3 location set to {DINOV3_LOCATION}")

    MODEL_NAME = "dinov3_vitb16"

    dino_v3_encoder = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=MODEL_NAME,
        source="local",
        pretrained=False,
    )

    num_unfrozen_layers = cfg.MODEL.BACKBONE.get('NUM_UNFROZEN_LAYERS', 0)

    return DINOv3Backbone(dino_v3_encoder, patch_size=16, num_unfrozen_layers=num_unfrozen_layers)

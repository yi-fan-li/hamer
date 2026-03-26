import torch
import torch.nn as nn


class DINOv3Backbone(nn.Module):
    def __init__(self, model, patch_size=16, num_unfrozen_layers=0):
        super().__init__()
        self.model = model
        self.patch_size = patch_size

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


def dinov3(cfg):
    DINOV3_LOCATION = "third-party/dinov3"

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

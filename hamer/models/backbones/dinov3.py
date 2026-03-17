import torch
import torch.nn as nn


class DINOv3Backbone(nn.Module):
    def __init__(self, model, patch_size=16):
        super().__init__()
        self.model = model
        self.patch_size = patch_size

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
    DINOV3_LOCATION = "./third_party/dinov3"

    print(f"DINOv3 location set to {DINOV3_LOCATION}")

    MODEL_NAME = "dinov3_vitb16"

    dino_v3_encoder = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=MODEL_NAME,
        source="local",
        pretrained=False,
    )

    return DINOv3Backbone(dino_v3_encoder, patch_size=16)

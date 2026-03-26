import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUM_JOINTS = 21

# Hand skeleton edges in OpenPose 21-joint format
_HAND_EDGES = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),          # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),           # index
    (0, 9),  (9, 10), (10, 11),(11, 12),          # middle
    (0, 13), (13, 14),(14, 15),(15, 16),          # ring
    (0, 17), (17, 18),(18, 19),(19, 20),          # pinky
]


def _build_normalized_adjacency(num_joints: int, edges) -> torch.Tensor:
    """Build D^{-1/2} (A + I) D^{-1/2} for spectral GCN."""
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + np.eye(num_joints, dtype=np.float32)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
    return torch.from_numpy(D_inv_sqrt @ A @ D_inv_sqrt)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # x: [B, N, in_dim],  A_hat: [N, N]
        x = torch.bmm(A_hat.unsqueeze(0).expand(x.size(0), -1, -1), x)
        x = self.linear(x)
        x = F.gelu(self.norm(x))
        return x


class GCNRefinementHead(nn.Module):
    """
    Refines coarse MANO 3D joint positions using a GCN over sampled backbone
    features.

    Dataflow:
        joints_3d [B,21,3] + feat_map [B,C,H,W] + pred_keypoints_2d [B,21,2]
        → sample one feature vector per joint via grid_sample
        → concat with 3D coords  → shared MLP  → GCN layers  → delta [B,21,3]
        → joints_3d + delta

    Coordinate system for grid_sample:
        pred_keypoints_2d is in [-0.5, 0.5] (full 256×256 image, centred at 0).
        The backbone receives x[:,:,:,32:-32] so the feature map covers
        x-pixels [32, 224], i.e. u ∈ [-0.375, 0.375] in the normalised space.

            gs_u = pred_u / 0.375   (maps feat coverage → [-1, 1])
            gs_v = pred_v / 0.5     (full height, no crop)
    """

    def __init__(self, feat_channels: int, hidden_dim: int, num_layers: int):
        super().__init__()

        A_hat = _build_normalized_adjacency(NUM_JOINTS, _HAND_EDGES)
        self.register_buffer('A_hat', A_hat)

        # (C + 3) → hidden_dim in one step
        self.input_proj = nn.Sequential(
            nn.Linear(feat_channels + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.delta_head = nn.Linear(hidden_dim, 3)
        # Zero-init so the model starts as identity
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        joints_3d: torch.Tensor,          # [B, 21, 3]
        feat_map: torch.Tensor,           # [B, C, H, W]
        pred_keypoints_2d: torch.Tensor,  # [B, 21, 2]  in [-0.5, 0.5] space
    ) -> torch.Tensor:
        """Returns refined joints [B, 21, 3]."""

        # --- Sample one feature vector per joint ---
        gs_u = pred_keypoints_2d[..., 0] / 0.375   # [B, 21]
        gs_v = pred_keypoints_2d[..., 1] / 0.5     # [B, 21]
        grid = torch.stack([gs_u, gs_v], dim=-1).unsqueeze(2)  # [B, 21, 1, 2]

        # [B, C, H, W] × [B, 21, 1, 2] → [B, C, 21, 1]
        sampled = F.grid_sample(
            feat_map, grid,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B, 21, C]

        # --- Build node features and project ---
        node_feats = torch.cat([sampled, joints_3d], dim=-1)  # [B, 21, C+3]
        h = self.input_proj(node_feats)                        # [B, 21, d]

        # --- GCN ---
        for layer in self.gcn_layers:
            h = layer(h, self.A_hat)   # [B, 21, d]

        # --- Residual delta ---
        delta = self.delta_head(h)     # [B, 21, 3]
        return joints_3d + delta

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUM_JOINTS = 16  # 1 global orient (wrist) + 15 hand joints

# MANO kinematic tree edges (parent → child), using MANO joint ordering:
#   0=wrist, 1-3=index, 4-6=middle, 7-9=little, 10-12=ring, 13-15=thumb
_MANO_EDGES = [
    (0, 1),  (1, 2),  (2, 3),          # index
    (0, 4),  (4, 5),  (5, 6),          # middle
    (0, 7),  (7, 8),  (8, 9),          # little/pinky
    (0, 10), (10, 11), (11, 12),       # ring
    (0, 13), (13, 14), (14, 15),       # thumb
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


def rot_mat_to_6d(rot_mat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to 6D representation (first two columns).

    Args:
        rot_mat: [..., 3, 3]
    Returns:
        [..., 6]  (col0 || col1 of each rotation matrix)
    """
    return torch.cat([rot_mat[..., 0], rot_mat[..., 1]], dim=-1)


def six_d_to_rot_mat(x6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation back to rotation matrices via
    Gram-Schmidt orthonormalisation.

    Args:
        x6d: [..., 6]
    Returns:
        [..., 3, 3]
    """
    a1 = x6d[..., :3]
    a2 = x6d[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # [..., 3, 3]


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
    Refines MANO pose parameters (16 joint rotations in 6D representation)
    using a GCN over sampled backbone features.

    Dataflow:
        joints_6d [B,16,6] + feat_map [B,C,H,W] + joints_2d [B,16,2]
        → sample one feature vector per joint via grid_sample
        → concat with 6D rotation → shared MLP → GCN layers → delta [B,16,6]
        → joints_6d + delta  (refined 6D rotations)

    The 6D representation (first two columns of the rotation matrix) is
    continuous and suitable for residual prediction. Caller converts back to
    rotation matrices via six_d_to_rot_mat (Gram-Schmidt).

    joints_2d ordering must match joints_6d: MANO kinematic order
    (0=wrist, 1-3=index, 4-6=middle, 7-9=little, 10-12=ring, 13-15=thumb).

    Coordinate system for grid_sample:
        joints_2d is in [-0.5, 0.5] (full 256×256 image, centred at 0).
        The backbone receives x[:,:,:,32:-32] so the feature map covers
        x-pixels [32, 224], i.e. u ∈ [-0.375, 0.375] in normalised space.

            gs_u = pred_u / 0.375   (maps feat coverage → [-1, 1])
            gs_v = pred_v / 0.5     (full height, no crop)
    """

    def __init__(self, feat_channels: int, hidden_dim: int, num_layers: int):
        super().__init__()

        A_hat = _build_normalized_adjacency(NUM_JOINTS, _MANO_EDGES)
        self.register_buffer('A_hat', A_hat)

        # (C + 6) → hidden_dim in one step
        self.input_proj = nn.Sequential(
            nn.Linear(feat_channels + 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.delta_head = nn.Linear(hidden_dim, 6)
        # Zero-init so the model starts as identity
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        joints_6d: torch.Tensor,   # [B, 16, 6]  6D rotations in MANO kin. order
        feat_map: torch.Tensor,    # [B, C, H, W]
        joints_2d: torch.Tensor,   # [B, 16, 2]  in [-0.5, 0.5] space, MANO kin. order
    ) -> torch.Tensor:
        """Returns refined 6D rotations [B, 16, 6]."""

        # --- Sample one feature vector per joint ---
        gs_u = joints_2d[..., 0] / 0.375   # [B, 16]
        gs_v = joints_2d[..., 1] / 0.5     # [B, 16]
        grid = torch.stack([gs_u, gs_v], dim=-1).unsqueeze(2)  # [B, 16, 1, 2]

        # [B, C, H, W] × [B, 16, 1, 2] → [B, C, 16, 1]
        sampled = F.grid_sample(
            feat_map, grid,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B, 16, C]

        # --- Build node features and project ---
        node_feats = torch.cat([sampled, joints_6d], dim=-1)  # [B, 16, C+6]
        h = self.input_proj(node_feats)                        # [B, 16, d]

        # --- GCN ---
        for layer in self.gcn_layers:
            h = layer(h, self.A_hat)   # [B, 16, d]

        # --- Residual delta in 6D space ---
        delta = self.delta_head(h)     # [B, 16, 6]
        return joints_6d + delta

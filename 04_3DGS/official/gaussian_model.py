import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


def knn_points_torch(
    points: torch.Tensor,
    queries: torch.Tensor,
    K: int,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch KNN, matching the return signature of pytorch3d.ops.knn_points.
    Returns squared distances.
    """
    B, N, D = points.shape
    M = queries.shape[1]
    K = min(K, N)

    all_dists = []
    all_idx = []
    for b in range(B):
        p = points[b]    # (N, D)
        q = queries[b]   # (M, D)

        dists_b = torch.empty((M, K), device=p.device, dtype=p.dtype)
        idx_b = torch.empty((M, K), device=p.device, dtype=torch.long)

        # Chunk over queries to avoid building a full (M, N) matrix at once
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            q_chunk = q[start:end]                              # (m, D)
            d = torch.cdist(q_chunk, p, p=2.0) ** 2             # (m, N), squared L2
            d_topk, i_topk = torch.topk(d, k=K, dim=-1, largest=False, sorted=True)
            dists_b[start:end] = d_topk
            idx_b[start:end] = i_topk

        all_dists.append(dists_b)
        all_idx.append(idx_b)

    dists = torch.stack(all_dists, dim=0)     # (B, M, K)
    indices = torch.stack(all_idx, dim=0)     # (B, M, K)
    return dists, indices, None


@dataclass
class GaussianParameters:
    positions: torch.Tensor   # (N, 3) World space positions
    colors: torch.Tensor      # (N, 3) RGB colors in [0,1]
    opacities: torch.Tensor   # (N, 1) Opacity values in [0,1]
    covariance: torch.Tensor  # (N, 3, 3) Covariance matrices
    rotations: torch.Tensor   # (N, 4) Quaternions
    scales: torch.Tensor      # (N, 3) Log-space scales

class GaussianModel(nn.Module):
    def __init__(self, points3D_xyz: torch.Tensor, points3D_rgb: torch.Tensor):
        """
        Initialize 3D Gaussian Splatting model
        
        Args:
            points3D_xyz: (N, 3) tensor of point positions
            points3D_rgb: (N, 3) tensor of RGB colors in [0, 255]
        """
        super().__init__()
        self.n_points = len(points3D_xyz)
        
        # Initialize learnable parameters
        self._init_positions(points3D_xyz)
        self._init_rotations()
        self._init_scales(points3D_xyz)
        self._init_colors(points3D_rgb)
        self._init_opacities()

    def _init_positions(self, points3D_xyz: torch.Tensor) -> None:
        """Initialize 3D positions from input points"""
        self.positions = nn.Parameter(
            torch.as_tensor(points3D_xyz, dtype=torch.float32)
        )

    def _init_rotations(self) -> None:
        """Initialize rotations as identity quaternions [w,x,y,z]"""
        initial_rotations = torch.zeros((self.n_points, 4))
        initial_rotations[:, 0] = 1.0  # w=1, x=y=z=0 for identity
        self.rotations = nn.Parameter(initial_rotations)

    def _init_scales(self, points3D_xyz: torch.Tensor) -> None:
        """Initialize scales based on local point density"""
        # Compute mean distance to K nearest neighbors
        K = min(50, self.n_points - 1)
        points = torch.as_tensor(points3D_xyz, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        dists, _, _ = knn_points_torch(points, points, K=K)
        
        # Use log space for unconstrained optimization
        mean_dists = torch.mean(torch.sqrt(dists[0].clamp_min(1e-12)), dim=1, keepdim=True) * 2.
        mean_dists = mean_dists.clamp(0.2*torch.median(mean_dists), 3.0*torch.median(mean_dists))  # Prevent infinite scales
        print('init_scales', torch.min(mean_dists), torch.max(mean_dists))
        
        log_scales = torch.log(mean_dists)
        self.scales = nn.Parameter(log_scales.repeat(1, 3))

    def _init_colors(self, points3D_rgb: torch.Tensor) -> None:
        """Initialize colors in logit space for sigmoid activation"""
        # Convert to [0,1] and apply logit for unconstrained optimization
        colors = torch.as_tensor(points3D_rgb, dtype=torch.float32) / 255.0
        colors = colors.clamp(0.001, 0.999)  # Prevent infinite logits
        self.colors = nn.Parameter(torch.logit(colors))

    def _init_opacities(self) -> None:
        """Initialize opacities in logit space for sigmoid activation"""
        # Initialize to high opacity (sigmoid(8.0) ≈ 0.9997)
        self.opacities = nn.Parameter(
            8.0 * torch.ones((self.n_points, 1), dtype=torch.float32)
        )

    def _compute_rotation_matrices(self) -> torch.Tensor:
        """Convert quaternions to 3x3 rotation matrices"""
        # Normalize quaternions to unit length
        q = self.rotations / self.rotations.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        w, x, y, z = q.unbind(-1)
        
        # Build rotation matrix elements
        R00 = 1 - 2*y*y - 2*z*z
        R01 = 2*x*y - 2*w*z
        R02 = 2*x*z + 2*w*y
        R10 = 2*x*y + 2*w*z
        R11 = 1 - 2*x*x - 2*z*z
        R12 = 2*y*z - 2*w*x
        R20 = 2*x*z - 2*w*y
        R21 = 2*y*z + 2*w*x
        R22 = 1 - 2*x*x - 2*y*y
        
        return torch.stack([
            R00, R01, R02,
            R10, R11, R12,
            R20, R21, R22
        ], dim=-1).reshape(-1, 3, 3)

    def compute_covariance(self) -> torch.Tensor:
        """Compute covariance matrices for all gaussians"""
        # Get rotation matrices
        R = self._compute_rotation_matrices()
        
        # Convert scales from log space and create diagonal matrices
        scales = torch.exp(self.scales.clamp(-6, 1.5))
        S = torch.diag_embed(scales)
        
        # Compute covariance: Sigma = R S S^T R^T = (R S) (R S)^T
        M = torch.bmm(R, S)
        Covs3d = torch.bmm(M, M.transpose(1, 2))
        
        return Covs3d

    def get_gaussian_params(self) -> GaussianParameters:
        """Get all gaussian parameters in world space"""
        return GaussianParameters(
            positions=self.positions,
            colors=torch.sigmoid(self.colors),
            opacities=torch.sigmoid(self.opacities),
            covariance=self.compute_covariance(),
            rotations=self.rotations / self.rotations.norm(dim=-1, keepdim=True).clamp_min(1e-8),
            scales=torch.exp(self.scales.clamp(-10, 6))
        )

    def forward(self) -> Dict[str, torch.Tensor]:
        """Forward pass returns dictionary of parameters.

        For the official diff-gaussian-rasterization path we hand the rasterizer
        the activated scales and normalized quaternions directly and let it build
        the 3D covariance on the GPU. This is how the original 3DGS does it and it
        produces the correct screen-space gradients used by densification.
        """
        params = self.get_gaussian_params()
        return {
            'positions': params.positions,    # (N, 3)
            'scales': params.scales,          # (N, 3) activated (exp of log-scales)
            'rotations': params.rotations,    # (N, 4) normalized [w, x, y, z]
            'colors': params.colors,          # (N, 3) RGB in [0, 1]
            'opacities': params.opacities,    # (N, 1) in [0, 1]
            # kept for backward-compat / debugging; the rasterizer does not use it
            'covariance': params.covariance,
        }
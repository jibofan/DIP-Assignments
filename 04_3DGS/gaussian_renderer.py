import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))


    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. Transform points to camera space
        cam_points = means3D @ R.T + t.unsqueeze(0) # (N, 3)
        
        # 2. Get depths before projection for proper sorting and clipping
        raw_z = cam_points[:, 2]
        safe_z = raw_z.clamp(min=0.01)
        depths = raw_z.clamp(min=1.)  # (N, )
        
        # 3. Project to screen space using camera intrinsics
        screen_points = cam_points @ K.T  # (N, 3)
        means2D = screen_points[..., :2] / safe_z.unsqueeze(-1) # (N, 2)
        
        # 4. Transform covariance to camera space and then to 2D
        # Compute Jacobian of perspective projection
        # For projection (X, Y, Z) -> (fx*X/Z + cx, fy*Y/Z + cy):
        # J = [[fx/Z,    0, -fx*X/Z^2],
        #      [   0, fy/Z, -fy*Y/Z^2]]
        fx, fy = K[0, 0], K[1, 1]
        X, Y = cam_points[:, 0], cam_points[:, 1]
        inv_Z = 1.0 / safe_z
        inv_Z2 = inv_Z * inv_Z

        J_proj = torch.zeros((N, 2, 3), device=means3D.device, dtype=means3D.dtype)
        J_proj[:, 0, 0] = fx * inv_Z
        J_proj[:, 0, 2] = -fx * X * inv_Z2
        J_proj[:, 1, 1] = fy * inv_Z
        J_proj[:, 1, 2] = -fy * Y * inv_Z2
        
        # Transform covariance to camera space
        # Apply world-to-camera rotation: Sigma_cam = R Sigma_world R^T
        covs_cam = R @ covs3d @ R.T  # (N, 3, 3), broadcasts over the N dim
        
        # Project to 2D
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))  # (N, 2, 2)
        covs2D = covs2D + 0.3 * torch.eye(2, device=covs2D.device, dtype=covs2D.dtype).unsqueeze(0)

        invalid = (raw_z <= 0).unsqueeze(-1)
        means2D = torch.where(invalid, torch.zeros_like(means2D), means2D)

        return means2D, covs2D, depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # Compute offset from mean (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-2
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0)
        
        # Compute determinant for normalization
        # Closed-form 2x2 inverse: [[a,b],[c,d]]^-1 = 1/det * [[d,-b],[-c,a]]
        a = covs2D[:, 0, 0]
        b = covs2D[:, 0, 1]
        c = covs2D[:, 1, 0]
        d = covs2D[:, 1, 1]
        det = (a * d - b * c).clamp(min=1e-4)  # (N,)

        inv_a = ( d / det).view(N, 1, 1)
        inv_b = (-b / det).view(N, 1, 1)
        inv_c = (-c / det).view(N, 1, 1)
        inv_d = ( a / det).view(N, 1, 1)

        # Mahalanobis distance: dx^T Sigma^-1 dx
        dx0 = dx[..., 0]  # (N, H, W)
        dx1 = dx[..., 1]  # (N, H, W)
        mahal = (dx0 * dx0) * inv_a + (dx0 * dx1) * (inv_b + inv_c) + (dx1 * dx1) * inv_d
        mahal = mahal.clamp(min=0.0, max=80.0)

        # Unnormalized gaussian (standard choice in 3DGS: peak = 1)
        gaussian = torch.exp(-0.5 * mahal)  # (N, H, W)
    
        return gaussian

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        N = means3D.shape[0]
        
        # 1. Project to 2D, means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # 2. Depth mask
        valid_mask = (depths > 1.) & (depths < 50.0)  # (N,)
        
        # 3. Sort by depth (front-to-back for correct alpha compositing)
        indices = torch.argsort(depths, dim=0, descending=False)  # (N, )
        means2D = means2D[indices]      # (N, 2)
        covs2D = covs2D[indices]       # (N, 2, 2)
        colors = colors[ indices]       # (N, 3)
        opacities = opacities[indices] # (N, 1)
        valid_mask = valid_mask[indices] # (N,)
        
        # 4. Compute gaussian values
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)
        
        # 5. Apply valid mask
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)  # (N, H, W)
        
        # 6. Alpha composition setup
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        alphas = alphas.clamp(0.0, 0.99)                   # avoid log(0) in T computation
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        # 7. Compute weights
        # Standard front-to-back alpha compositing:
        #   w_i = alpha_i * T_i, where T_i = prod_{j<i} (1 - alpha_j)
        # Compute T via a cumulative product of (1 - alpha) shifted by one.
        ones = torch.ones_like(alphas[:1])                       # (1, H, W)
        trans = torch.cat([ones, 1.0 - alphas[:-1]], dim=0)      # (N, H, W)
        T = torch.cumprod(trans, dim=0)                          # (N, H, W)
        weights = alphas * T                                     # (N, H, W)
        
        # 8. Final rendering
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        
        return rendered
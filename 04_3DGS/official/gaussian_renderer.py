import math
from typing import Tuple

import torch
import torch.nn as nn

# Official Inria CUDA rasterizer.
# Install (on a CUDA machine, with a CUDA-enabled PyTorch build + matching nvcc):
#   pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
# or build the submodule that ships with https://github.com/graphdeco-inria/gaussian-splatting
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


class GaussianRenderer(nn.Module):
    """
    Thin wrapper around the official 3D Gaussian Splatting CUDA rasterizer
    (graphdeco-inria/diff-gaussian-rasterization).

    It keeps the same camera parameterization the rest of this codebase uses
    (COLMAP / OpenCV style world-to-camera ``R``, ``t`` and a pinhole intrinsic
    matrix ``K``) and internally builds the view / projection matrices and FoV
    that the rasterizer expects, so the training code only has to swap which
    Gaussian attributes it passes in.

    Camera-matrix construction below was numerically verified to reproduce the
    pinhole projection used by the previous pure-PyTorch renderer (including
    off-center principal points), reducing exactly to the official
    ``getProjectionMatrix`` when ``cx = W/2`` and ``cy = H/2``.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        znear: float = 0.01,
        zfar: float = 100.0,
        sh_degree: int = 0,
        bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale_modifier: float = 1.0,
    ):
        super().__init__()
        self.H = int(image_height)
        self.W = int(image_width)
        self.znear = float(znear)
        self.zfar = float(zfar)
        self.sh_degree = int(sh_degree)
        self.scale_modifier = float(scale_modifier)
        # Registered as a buffer so it follows `.to(device)` with the module.
        self.register_buffer("bg_color", torch.tensor(bg_color, dtype=torch.float32))

    # ------------------------------------------------------------------ #
    # Camera matrices                                                     #
    # ------------------------------------------------------------------ #
    def _view_matrix(self, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """World-to-view transform, transposed, as the rasterizer expects.

        This codebase uses  X_cam = R @ X_world + t  (R, t are world->camera),
        so the 4x4 world-to-camera matrix is W2C = [[R, t], [0, 1]] and the
        CUDA code (row-vector convention) wants W2C^T.
        """
        W2C = torch.eye(4, device=R.device, dtype=R.dtype)
        W2C[:3, :3] = R
        W2C[:3, 3] = t
        return W2C.transpose(0, 1).contiguous()

    def _proj_matrix(self, K: torch.Tensor) -> torch.Tensor:
        """OpenGL-style projection matrix (transposed), principal-point aware.

        Equivalent to the official ``getProjectionMatrix`` when the principal
        point is centered; the cx/cy terms add support for off-center COLMAP
        intrinsics. z_sign = +1 (OpenCV: camera looks down +Z).
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        W, H = self.W, self.H
        z_n, z_f = self.znear, self.zfar

        P = torch.zeros((4, 4), device=K.device, dtype=K.dtype)
        P[0, 0] = 2.0 * fx / W
        P[1, 1] = 2.0 * fy / H
        P[0, 2] = 2.0 * cx / W - 1.0
        P[1, 2] = 2.0 * cy / H - 1.0
        P[2, 2] = z_f / (z_f - z_n)
        P[2, 3] = -(z_f * z_n) / (z_f - z_n)
        P[3, 2] = 1.0
        return P.transpose(0, 1).contiguous()

    def _make_settings(self, K, R, t, device, dtype) -> GaussianRasterizationSettings:
        fx, fy = K[0, 0], K[1, 1]
        tanfovx = float(self.W / (2.0 * fx))  # => focal_x == fx inside the rasterizer
        tanfovy = float(self.H / (2.0 * fy))  # => focal_y == fy

        view_T = self._view_matrix(R, t)                                   # (4, 4)
        proj_T = self._proj_matrix(K)                                      # (4, 4)
        full_proj = (view_T.unsqueeze(0) @ proj_T.unsqueeze(0)).squeeze(0)  # world->clip
        cam_center = view_T.inverse()[3, :3]                               # (3,)

        kwargs = dict(
            image_height=self.H,
            image_width=self.W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color.to(device=device, dtype=dtype),
            scale_modifier=self.scale_modifier,
            viewmatrix=view_T,
            projmatrix=full_proj,
            sh_degree=self.sh_degree,
            campos=cam_center,
            prefiltered=False,
            debug=False,
        )
        # Newer rasterizer versions add an `antialiasing` field; include it only
        # if this installed version supports it (keeps us version-agnostic).
        if "antialiasing" in GaussianRasterizationSettings._fields:
            kwargs["antialiasing"] = False
        return GaussianRasterizationSettings(**kwargs)

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        means3D: torch.Tensor,     # (N, 3) world-space positions
        scales: torch.Tensor,      # (N, 3) activated (positive) scales
        rotations: torch.Tensor,   # (N, 4) normalized quaternions [w, x, y, z]
        opacities: torch.Tensor,   # (N, 1) in [0, 1]
        colors: torch.Tensor,      # (N, 3) RGB in [0, 1] (precomputed)
        K: torch.Tensor,           # (3, 3) pinhole intrinsics
        R: torch.Tensor,           # (3, 3) world-to-camera rotation
        t: torch.Tensor,           # (3,)   world-to-camera translation
    ) -> torch.Tensor:             # (H, W, 3)
        device = means3D.device
        dtype = means3D.dtype
        K = K.to(device=device, dtype=dtype)
        R = R.to(device=device, dtype=dtype)
        t = t.reshape(3).to(device=device, dtype=dtype)

        raster_settings = self._make_settings(K, R, t, device, dtype)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Screen-space means: the rasterizer accumulates per-Gaussian 2D
        # gradients into `.grad` of this tensor (used by adaptive density
        # control). Harmless to keep even without densification.
        screenspace_points = torch.zeros_like(means3D, requires_grad=True)
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        out = rasterizer(
            means3D=means3D.contiguous(),
            means2D=screenspace_points,
            shs=None,
            colors_precomp=colors.contiguous(),
            opacities=opacities.contiguous(),
            scales=scales.contiguous(),
            rotations=rotations.contiguous(),
            cov3D_precomp=None,
        )

        # Different versions return (image, radii), (image, radii, depth), etc.
        rendered_image = out[0] if isinstance(out, (tuple, list)) else out

        # Rasterizer returns (3, H, W); the rest of the codebase expects (H, W, 3).
        return rendered_image.permute(1, 2, 0).contiguous()

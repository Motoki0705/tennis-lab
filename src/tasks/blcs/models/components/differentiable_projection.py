"""Differentiable pinhole projection for reprojection-based training.

Projects predicted 3D positions (in normalised court coordinates) back into
per-camera UV space so that a reprojection loss can be computed against
observed 2D ball detections.

Camera parameters are treated as **fixed** constants — no gradient flows
through them.  Gradients propagate only through the predicted 3D positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.schema.court import COURT_COORD_SCALE_XYZ


class DifferentiableProjection(nn.Module):
    """Batch-aware differentiable pinhole projection.

    Maps normalised court-coordinate 3D predictions ``(B, T, 3)`` to
    normalised UV coordinates ``(B, N, T, 2)`` for each of *N* cameras.

    The projection follows the same convention as
    :func:`src.utils.projection.camera_projector.project_points`:

    .. math::

        X_{\\text{cam}} = (X_{\\text{world}} - C) \\, R^\\top

        u_{\\text{px}} = f \\, \\frac{X_{\\text{cam}, x}}{X_{\\text{cam}, z}} + c_x

        v_{\\text{px}} = f \\, \\frac{-X_{\\text{cam}, y}}{X_{\\text{cam}, z}} + c_y

        u_{\\text{norm}} = u_{\\text{px}} / w, \\quad v_{\\text{norm}} = v_{\\text{px}} / h
    """

    scale_xyz: Tensor

    def __init__(
        self,
        scale_xyz: tuple[float, float, float] = COURT_COORD_SCALE_XYZ,
        depth_eps: float = 1e-6,
    ) -> None:
        """Initialise the projection module.

        Args:
            scale_xyz: Per-axis scale used to convert normalised court
                coordinates back to world coordinates (metres).
            depth_eps: Minimum depth value to avoid division-by-zero.
        """
        super().__init__()
        self.register_buffer(
            "scale_xyz",
            torch.tensor(scale_xyz, dtype=torch.float32),
            persistent=False,
        )
        self.depth_eps = float(depth_eps)

    def forward(
        self,
        position_norm: Tensor,
        camera_R: Tensor,
        camera_C: Tensor,
        camera_f: Tensor,
        camera_cx: Tensor,
        camera_cy: Tensor,
        camera_w: Tensor,
        camera_h: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Project normalised 3D positions into per-camera UV space.

        Args:
            position_norm: Predicted 3D positions in normalised court
                coordinates, shape ``(B, T, 3)``.
            camera_R: Rotation matrices (world → camera), ``(B, N, 3, 3)``.
            camera_C: Camera centres in world coordinates, ``(B, N, 3)``.
            camera_f: Focal lengths in pixels, ``(B, N)``.
            camera_cx: Principal-point x, ``(B, N)``.
            camera_cy: Principal-point y, ``(B, N)``.
            camera_w: Image widths, ``(B, N)``.
            camera_h: Image heights, ``(B, N)``.

        Returns:
            A tuple ``(uv_norm, in_front)``:

            - **uv_norm** — normalised UV coordinates ``(B, N, T, 2)``,
              values in ``[0, 1]`` when the point is visible.
            - **in_front** — boolean mask ``(B, N, T)`` indicating whether
              each point is in front of the camera (depth > 0).
        """
        # -- Denormalise to world coordinates (metres) --------------------
        # scale_xyz is a buffer → no gradient.
        scale = self.scale_xyz.to(
            device=position_norm.device,
            dtype=position_norm.dtype,
        )
        xyz_world = position_norm * scale.view(1, 1, 3)  # (B, T, 3)

        # -- Prepare for broadcasting: (B, 1, T, 3) vs (B, N, 1, 3) -----
        B, T, _ = xyz_world.shape
        N = camera_R.shape[1]

        xyz_world = xyz_world.unsqueeze(1).expand(B, N, T, 3)  # (B, N, T, 3)

        # Detach camera parameters — no gradient through them.
        C = camera_C.detach().unsqueeze(2)  # (B, N, 1, 3)
        R = camera_R.detach()  # (B, N, 3, 3)

        # -- World → camera coordinates ----------------------------------
        X_rel = xyz_world - C  # (B, N, T, 3)
        # Batched matmul:  (B, N, T, 3) @ (B, N, 3, 3)^T → (B, N, T, 3)
        X_cam = torch.einsum("bntj,bnkj->bntk", X_rel, R)

        z = X_cam[..., 2]  # (B, N, T)
        in_front = z > self.depth_eps
        z_safe = torch.where(in_front, z, torch.ones_like(z))

        # -- Pinhole projection -------------------------------------------
        f = camera_f.detach().unsqueeze(2)  # (B, N, 1)
        cx = camera_cx.detach().unsqueeze(2)
        cy = camera_cy.detach().unsqueeze(2)
        w = camera_w.detach().unsqueeze(2).clamp(min=1.0)
        h = camera_h.detach().unsqueeze(2).clamp(min=1.0)

        # OpenCV convention (must mirror src.utils.projection.project_points):
        # camera y-axis points image-down, so no v-flip.
        u_px = f * (X_cam[..., 0] / z_safe) + cx
        v_px = f * (X_cam[..., 1] / z_safe) + cy

        u_norm = u_px / w
        v_norm = v_px / h

        uv_norm = torch.stack([u_norm, v_norm], dim=-1)  # (B, N, T, 2)

        return uv_norm, in_front

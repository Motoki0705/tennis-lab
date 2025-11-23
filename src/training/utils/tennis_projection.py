"""Projection helpers shared by training-time debug visualization utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from src.tennis.geometry.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    NET_HEIGHT_POST,
)


def norm_to_px(coords: Tensor, width: int, height: int) -> np.ndarray:
    """Convert normalized coordinates ([-1, 1]) into pixel space."""
    coords_arr = coords.detach().float().cpu().numpy().astype("float32")
    out = np.empty_like(coords_arr)
    w_span = max(width - 1, 1)
    h_span = max(height - 1, 1)
    out[..., 0] = (coords_arr[..., 0] + 1.0) * 0.5 * float(w_span)
    out[..., 1] = (coords_arr[..., 1] + 1.0) * 0.5 * float(h_span)
    return out


def denorm_pose3d(pose_norm: Tensor) -> Tensor:
    """Scale normalized 3D poses back to metric court dimensions."""
    scales = pose_norm.new_tensor(
        [HALF_DOUBLES_WIDTH, HALF_LENGTH, NET_HEIGHT_POST],
        dtype=pose_norm.dtype,
    )
    return pose_norm * scales


def project_world_points(
    cam_C: Tensor,
    cam_R: Tensor,
    cam_intr: Tensor,
    xyz_world: Tensor,
) -> tuple[Tensor, Tensor]:
    """Project 3D world points into image plane and return visibility mask."""
    rel = xyz_world - cam_C.view(1, 3)
    Xc = rel @ cam_R.t()
    z = Xc[:, 2]
    mask = z > 1e-6
    z_safe = torch.where(mask, z, torch.ones_like(z))
    f = cam_intr[0]
    cx = cam_intr[1]
    cy = cam_intr[2]
    u = f * (Xc[:, 0] / z_safe) + cx
    v = f * (-Xc[:, 1] / z_safe) + cy
    uv = torch.stack([u, v], dim=-1)
    return uv, mask

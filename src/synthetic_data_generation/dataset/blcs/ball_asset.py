"""Deterministic asset-local Gaussian surface for a regulation tennis ball."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from src.synthetic_data_generation.composition import (
    GaussianCoordinates,
    GaussianTensorSet,
    validate_asset_tensors,
)
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCSCompositionAssets,
)

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def build_ball_gaussian_asset(assets: BLCSCompositionAssets) -> GaussianTensorSet:
    """Build the configured metric Gaussian shell without mesh or pixel primitives."""
    if not isinstance(assets, BLCSCompositionAssets):
        raise TypeError("assets must be BLCSCompositionAssets.")
    count = assets.ball.gaussian_count
    settings = assets.settings
    dtype = torch.float32 if assets.ball.floating_dtype == "float32" else torch.float64

    indices = torch.arange(count, dtype=dtype)
    z = 1.0 - 2.0 * (indices + 0.5) / float(count)
    radial_xy = torch.sqrt(torch.clamp(1.0 - z.square(), min=0.0))
    longitude = indices * _GOLDEN_ANGLE
    normals = torch.stack(
        (
            radial_xy * torch.cos(longitude),
            radial_xy * torch.sin(longitude),
            z,
        ),
        dim=1,
    )
    means = settings.radius_m * normals
    quaternions = _quaternions_from_positive_z(normals)
    scales = torch.tensor(
        (
            settings.tangential_scale_m,
            settings.tangential_scale_m,
            settings.radial_scale_m,
        ),
        dtype=dtype,
    ).expand(count, 3)
    opacity = torch.full((count,), settings.opacity, dtype=dtype)

    latitude = torch.asin(torch.clamp(z, min=-1.0, max=1.0))
    seam_curve = 0.36 * torch.sin(2.0 * longitude)
    seam = torch.abs(latitude - seam_curve) <= settings.seam_width_radians
    base_color = torch.tensor(settings.base_color_linear_rgb, dtype=dtype)
    seam_color = torch.tensor(settings.seam_color_linear_rgb, dtype=dtype)
    features = torch.where(seam[:, None], seam_color, base_color)

    result = GaussianTensorSet(
        means=means.contiguous(),
        quaternions_wxyz=quaternions.contiguous(),
        log_scales=torch.log(scales).contiguous(),
        opacity_logits=torch.logit(opacity).contiguous(),
        features=features.contiguous(),
        instance_ids=torch.zeros(count, dtype=torch.int64),
        coordinates=GaussianCoordinates.asset_local_metres(),
        appearance_model=assets.ball.appearance_model,
        appearance_space=assets.ball.appearance_space,
    )
    validate_asset_tensors(assets.ball, result)
    if not bool(seam.any()) or bool(seam.all()):
        raise ValueError(
            "Configured ball Gaussian sampling must contain both felt and seam materials."
        )
    return result


def _quaternions_from_positive_z(normals: Tensor) -> Tensor:
    """Return normalized wxyz rotations mapping local +Z onto each normal."""
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must have shape [N,3].")
    positive_z = torch.zeros_like(normals)
    positive_z[:, 2] = 1.0
    cross = torch.linalg.cross(positive_z, normals, dim=1)
    scalar = 1.0 + normals[:, 2]
    quaternions = torch.cat((scalar[:, None], cross), dim=1)
    near_south = scalar <= torch.finfo(normals.dtype).eps
    if bool(near_south.any()):
        quaternions[near_south] = quaternions.new_tensor((0.0, 1.0, 0.0, 0.0))
    return torch.nn.functional.normalize(quaternions, dim=1)


__all__ = ["build_ball_gaussian_asset"]

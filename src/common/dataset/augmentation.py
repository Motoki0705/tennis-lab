"""Common data augmentation utilities for datasets.

This module provides shared augmentation functions used across PLCS, BLCS,
and other modules to avoid code duplication.
"""

from __future__ import annotations

import torch
from torch import Tensor


def scale_uv_with_visibility(
    uv: Tensor,
    visibility: Tensor,
    scale: float,
    center: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Scale normalized UV coordinates and update visibility by bounds.

    Args:
        uv: UV tensor of shape (..., 2) in normalized coordinates [0, 1].
        visibility: Visibility tensor matching uv prefix shape (...,).
        scale: Isotropic scaling factor.
        center: Scaling center in normalized UV space.

    Returns:
        Tuple of (scaled_uv, updated_visibility).

    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")
    if uv.shape[-1] != 2:
        raise ValueError(f"uv must have last dimension 2, got shape {tuple(uv.shape)}.")

    uv_scaled = (uv - center) * scale + center
    in_bounds = (
        (uv_scaled[..., 0] >= 0.0)
        & (uv_scaled[..., 0] <= 1.0)
        & (uv_scaled[..., 1] >= 0.0)
        & (uv_scaled[..., 1] <= 1.0)
    )

    if visibility.dtype == torch.bool:
        visibility_scaled = visibility & in_bounds
    else:
        visibility_scaled = visibility * in_bounds.to(visibility.dtype)

    return uv_scaled.clamp(0.0, 1.0), visibility_scaled


def add_gaussian_noise(
    tensor: Tensor,
    noise_std: float,
) -> Tensor:
    """Add Gaussian noise to a tensor.

    Args:
        tensor: Input tensor of any shape.
        noise_std: Standard deviation of Gaussian noise.

    Returns:
        Tensor with added noise (same shape as input).

    """
    if noise_std <= 0:
        return tensor

    noise = torch.randn_like(tensor) * noise_std
    return tensor + noise


def random_visibility_dropout(
    visibility: Tensor,
    drop_prob: float,
) -> Tensor:
    """Randomly drop visibility flags for data augmentation.

    Args:
        visibility: Boolean or float visibility tensor of any shape.
        drop_prob: Probability of dropping each visibility flag.

    Returns:
        Updated visibility tensor with some flags set to False/0.

    """
    if drop_prob <= 0:
        return visibility

    # Generate dropout mask
    drop_mask = torch.rand(visibility.shape) < drop_prob

    # Apply dropout (convert to bool if needed)
    if visibility.dtype == torch.bool:
        return visibility & ~drop_mask
    else:
        # Float visibility (0.0 or 1.0)
        drop_mask_float = drop_mask.float()
        return visibility * (1.0 - drop_mask_float)


def augment_keypoints(
    keypoints: Tensor,
    visibility: Tensor,
    noise_std: float = 0.0,
    visibility_drop_prob: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Apply common keypoint augmentations: noise and visibility dropout.

    This is a convenience function that combines Gaussian noise and visibility
    dropout, commonly used together in PLCS and BLCS datasets.

    Args:
        keypoints: Keypoint coordinates, shape (..., N, 2) or (..., 2).
        visibility: Visibility flags, shape (..., N) or (...,).
        noise_std: Standard deviation of Gaussian noise to add.
        visibility_drop_prob: Probability of dropping visibility flags.

    Returns:
        Tuple of (augmented_keypoints, augmented_visibility).

    """
    # Add Gaussian noise
    augmented_kp = add_gaussian_noise(keypoints, noise_std)

    # Random visibility dropout
    augmented_vis = random_visibility_dropout(visibility, visibility_drop_prob)

    return augmented_kp, augmented_vis

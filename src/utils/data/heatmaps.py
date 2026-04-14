"""Shared spatial heatmap utilities."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def generate_gaussian_heatmaps(
    size_hw: tuple[int, int],
    centers_xy: Tensor | tuple[float, float] | list[float],
    sigma_ratio: float,
    visibility: Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Generate 2-D Gaussian heatmaps from normalized centers.

    Args:
        size_hw: Heatmap size as ``(height, width)``.
        centers_xy: Normalized centers in ``(x, y)`` ordering with shape
            ``(..., 2)``.
        sigma_ratio: Gaussian sigma as a ratio of the heatmap diagonal.
        visibility: Optional visibility mask broadcastable to ``centers_xy[..., 0]``.
        dtype: Output dtype.
        device: Output device.

    Returns:
        Heatmaps with shape ``(..., H, W)``.
    """
    height, width = _validate_size(size_hw)
    if sigma_ratio <= 0:
        raise ValueError("sigma_ratio must be positive.")

    centers = torch.as_tensor(centers_xy, dtype=dtype, device=device)
    if centers.shape == (2,):
        centers = centers.unsqueeze(0)
        squeeze_result = True
    else:
        squeeze_result = False
    if centers.shape[-1] != 2:
        raise ValueError(f"centers_xy must have shape (..., 2), got {tuple(centers.shape)}.")

    yy = torch.linspace(0.0, 1.0, height, dtype=dtype, device=centers.device)
    xx = torch.linspace(0.0, 1.0, width, dtype=dtype, device=centers.device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    view_shape = centers.shape[:-1] + (1, 1)
    center_x = centers[..., 0].view(view_shape)
    center_y = centers[..., 1].view(view_shape)

    sigma_norm = float(sigma_ratio) * math.sqrt(float(height * height + width * width))
    sigma_x = sigma_norm / max(width - 1, 1)
    sigma_y = sigma_norm / max(height - 1, 1)
    denom_x = 2.0 * sigma_x * sigma_x
    denom_y = 2.0 * sigma_y * sigma_y

    heatmaps = torch.exp(
        -(((grid_x - center_x) ** 2) / denom_x + ((grid_y - center_y) ** 2) / denom_y)
    )

    in_bounds = (
        (centers[..., 0] >= 0.0)
        & (centers[..., 0] <= 1.0)
        & (centers[..., 1] >= 0.0)
        & (centers[..., 1] <= 1.0)
    )
    if visibility is None:
        valid = in_bounds
    else:
        visibility_tensor = torch.as_tensor(visibility, dtype=torch.bool, device=centers.device)
        valid = visibility_tensor & in_bounds
    heatmaps = heatmaps * valid.view(view_shape).to(dtype=dtype)

    if squeeze_result:
        return heatmaps.squeeze(0)
    return heatmaps


def generate_gaussian_heatmap(
    size_hw: tuple[int, int],
    center_xy: Tensor | tuple[float, float] | list[float],
    sigma_ratio: float,
    visible: bool = True,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Generate a single 2-D Gaussian heatmap from one normalized center."""
    return generate_gaussian_heatmaps(
        size_hw=size_hw,
        centers_xy=center_xy,
        sigma_ratio=sigma_ratio,
        visibility=visible,
        dtype=dtype,
        device=device,
    )


def heatmaps_to_argmax(heatmaps: Tensor) -> tuple[Tensor, Tensor]:
    """Convert heatmaps to sparse normalized coordinates via hard argmax.

    Args:
        heatmaps: Tensor with shape ``(..., H, W)``.

    Returns:
        Tuple of:
            - coords: Normalized ``(..., 2)`` coordinates in ``(x, y)`` ordering.
            - values: Peak values with shape ``(...)``.
    """
    if heatmaps.ndim < 2:
        raise ValueError(f"heatmaps must have shape (..., H, W), got {tuple(heatmaps.shape)}.")

    *leading_shape, height, width = heatmaps.shape
    flat = heatmaps.reshape(*leading_shape, height * width)
    values, indices = flat.max(dim=-1)

    x = indices % width
    y = torch.div(indices, width, rounding_mode="floor")

    if width > 1:
        x = x.to(heatmaps.dtype) / float(width - 1)
    else:
        x = torch.zeros_like(x, dtype=heatmaps.dtype)
    if height > 1:
        y = y.to(heatmaps.dtype) / float(height - 1)
    else:
        y = torch.zeros_like(y, dtype=heatmaps.dtype)

    coords = torch.stack([x, y], dim=-1)
    return coords, values


def _validate_size(size_hw: tuple[int, int]) -> tuple[int, int]:
    if len(size_hw) != 2:
        raise ValueError(f"size_hw must be (height, width), got {size_hw!r}.")
    height, width = int(size_hw[0]), int(size_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"size_hw must be positive, got {(height, width)}.")
    return height, width

"""Shared spatial heatmap utilities."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
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


def heatmaps_to_soft_argmax(
    heatmaps: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Convert heatmaps to dense normalized coordinates via soft-argmax.

    Unlike :func:`heatmaps_to_argmax`, this conversion is differentiable and
    can propagate gradients from coordinate-space losses back to the heatmaps.

    Args:
        heatmaps: Tensor with shape ``(..., H, W)``. Values are treated as
            unnormalized scores (e.g. logits) for the spatial softmax.
        temperature: Softmax temperature. Lower values sharpen the spatial
            distribution towards the hard argmax.

    Returns:
        Normalized ``(..., 2)`` coordinates in ``(x, y)`` ordering.
    """
    if heatmaps.ndim < 2:
        raise ValueError(f"heatmaps must have shape (..., H, W), got {tuple(heatmaps.shape)}.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    *leading_shape, height, width = heatmaps.shape
    flat = heatmaps.reshape(*leading_shape, height * width)
    probs = torch.softmax(flat / float(temperature), dim=-1)
    probs = probs.reshape(*leading_shape, height, width)

    xs = (
        torch.linspace(0.0, 1.0, width, dtype=heatmaps.dtype, device=heatmaps.device)
        if width > 1
        else torch.zeros(1, dtype=heatmaps.dtype, device=heatmaps.device)
    )
    ys = (
        torch.linspace(0.0, 1.0, height, dtype=heatmaps.dtype, device=heatmaps.device)
        if height > 1
        else torch.zeros(1, dtype=heatmaps.dtype, device=heatmaps.device)
    )
    x = (probs.sum(dim=-2) * xs).sum(dim=-1)
    y = (probs.sum(dim=-1) * ys).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


def heatmaps_to_peaks(
    heatmaps: Tensor,
    *,
    threshold: float,
    nms_kernel: int,
    max_peaks: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Extract thresholded local peaks from dense heatmaps.

    Args:
        heatmaps: Tensor with shape ``(..., H, W)``.
        threshold: Minimum accepted heatmap value.
        nms_kernel: Odd max-pooling kernel used for local-maximum suppression.
        max_peaks: Maximum number of peaks retained per heatmap.

    Returns:
        Tuple of:
            - coords: Normalized coordinates with shape ``(..., K, 2)``.
            - values: Peak values with shape ``(..., K)``.
            - valid: Boolean mask with shape ``(..., K)``.
    """
    if heatmaps.ndim < 2:
        raise ValueError(f"heatmaps must have shape (..., H, W), got {tuple(heatmaps.shape)}.")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    if nms_kernel <= 0 or nms_kernel % 2 == 0:
        raise ValueError("nms_kernel must be a positive odd integer.")
    if max_peaks <= 0:
        raise ValueError("max_peaks must be positive.")

    *leading_shape, height, width = heatmaps.shape
    flattened_leading = math.prod(leading_shape) if leading_shape else 1
    maps = heatmaps.reshape(flattened_leading, 1, height, width)
    pooled = F.max_pool2d(
        maps,
        kernel_size=nms_kernel,
        stride=1,
        padding=nms_kernel // 2,
    )
    local_maxima = (maps >= pooled) & (maps >= float(threshold))
    candidate_values = maps.masked_fill(~local_maxima, float("-inf")).flatten(1)
    k = min(max_peaks, height * width)
    values, indices = candidate_values.topk(k, dim=1)
    valid = torch.isfinite(values)
    values = torch.where(valid, values, torch.zeros_like(values))

    x = indices % width
    y = torch.div(indices, width, rounding_mode="floor")
    x_normalized = (
        x.to(heatmaps.dtype) / float(width - 1)
        if width > 1
        else torch.zeros_like(x, dtype=heatmaps.dtype)
    )
    y_normalized = (
        y.to(heatmaps.dtype) / float(height - 1)
        if height > 1
        else torch.zeros_like(y, dtype=heatmaps.dtype)
    )
    coords = torch.stack([x_normalized, y_normalized], dim=-1)

    output_shape = tuple(leading_shape) + (k,)
    return (
        coords.reshape(*output_shape, 2),
        values.reshape(output_shape),
        valid.reshape(output_shape),
    )


def _validate_size(size_hw: tuple[int, int]) -> tuple[int, int]:
    if len(size_hw) != 2:
        raise ValueError(f"size_hw must be (height, width), got {size_hw!r}.")
    height, width = int(size_hw[0]), int(size_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"size_hw must be positive, got {(height, width)}.")
    return height, width

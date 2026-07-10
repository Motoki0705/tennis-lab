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


def heatmaps_to_pixel_coords(
    heatmaps: Tensor,
    *,
    height: int | None = None,
    width: int | None = None,
) -> Tensor:
    """Convert heatmaps to pixel coordinates via hard argmax.

    Decodes ``(..., H, W)`` heatmaps to ``(..., 2)`` ``(x, y)`` pixel coordinates
    by taking the argmax (see :func:`heatmaps_to_argmax`) and scaling the
    normalized result by ``(width - 1, height - 1)``.

    Args:
        heatmaps: Tensor with shape ``(..., H, W)``.
        height: Target pixel height. Defaults to the heatmap height.
        width: Target pixel width. Defaults to the heatmap width.

    Returns:
        Pixel coordinates with shape ``(..., 2)`` in ``(x, y)`` ordering.
    """
    *_, heatmap_h, heatmap_w = heatmaps.shape
    out_height = heatmap_h if height is None else int(height)
    out_width = heatmap_w if width is None else int(width)
    coords_normalized, _ = heatmaps_to_argmax(heatmaps)
    scale = coords_normalized.new_tensor(
        [
            float(out_width - 1) if out_width > 1 else 0.0,
            float(out_height - 1) if out_height > 1 else 0.0,
        ]
    )
    return coords_normalized * scale


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


def refine_peaks_log_parabolic(
    heatmaps: Tensor,
    coords: Tensor,
    *,
    eps: float = 1.0e-12,
) -> Tensor:
    """Refine lattice peak coordinates to sub-cell precision.

    Fits a 1-D parabola per axis to the log of the three heatmap values
    around each peak cell. For heatmaps that sample a 2-D axis-aligned
    Gaussian (the supervised ball-detection target) the recovered offset is
    exact, so this removes the lattice quantization of
    :func:`heatmaps_to_argmax` / :func:`heatmaps_to_peaks`.

    Peaks on the heatmap border or with a degenerate local curvature keep
    their input coordinate on that axis. Offsets are clamped to half a cell,
    matching the argmax guarantee that the true peak lies within the cell.

    Args:
        heatmaps: Tensor with shape ``(..., H, W)`` of non-negative values.
        coords: Normalized ``(x, y)`` peak coordinates produced by an argmax
            over *heatmaps*, with shape ``(..., 2)`` or ``(..., K, 2)`` whose
            leading dimensions match *heatmaps*.
        eps: Floor applied to heatmap values before the log, and minimum
            curvature magnitude accepted by the fit.

    Returns:
        Refined normalized coordinates with the same shape as *coords*.
    """
    if heatmaps.ndim < 2:
        raise ValueError(f"heatmaps must have shape (..., H, W), got {tuple(heatmaps.shape)}.")
    if coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape (..., 2), got {tuple(coords.shape)}.")
    *leading_shape, height, width = heatmaps.shape
    coords_shape = tuple(coords.shape)
    if coords_shape[: len(leading_shape)] != tuple(leading_shape) or len(coords_shape) not in (
        len(leading_shape) + 1,
        len(leading_shape) + 2,
    ):
        raise ValueError(
            "coords leading dimensions must match heatmaps leading dimensions "
            f"with an optional peak axis, got coords {coords_shape} for "
            f"heatmaps {tuple(heatmaps.shape)}."
        )
    if eps <= 0:
        raise ValueError("eps must be positive.")

    flattened_leading = math.prod(leading_shape) if leading_shape else 1
    maps = heatmaps.reshape(flattened_leading, height, width)
    flat_coords = coords.reshape(flattened_leading, -1, 2)

    x_index = (
        (flat_coords[..., 0] * float(width - 1)).round().long().clamp(0, width - 1)
    )
    y_index = (
        (flat_coords[..., 1] * float(height - 1)).round().long().clamp(0, height - 1)
    )
    batch_index = torch.arange(flattened_leading, device=heatmaps.device)[:, None]

    def _log_value(dy: Tensor, dx: Tensor) -> Tensor:
        y_neighbor = (y_index + dy).clamp(0, height - 1)
        x_neighbor = (x_index + dx).clamp(0, width - 1)
        return maps[batch_index, y_neighbor, x_neighbor].clamp_min(eps).log()

    zero = torch.zeros_like(x_index)
    one = torch.ones_like(x_index)

    def _axis_offset(
        log_minus: Tensor,
        log_center: Tensor,
        log_plus: Tensor,
        index: Tensor,
        last_index: int,
    ) -> Tensor:
        curvature = 2.0 * log_center - log_minus - log_plus
        offset = 0.5 * (log_plus - log_minus) / curvature.clamp_min(eps)
        offset = offset.clamp(-0.5, 0.5)
        interior = (index > 0) & (index < last_index) & (curvature > eps)
        return torch.where(interior, offset, torch.zeros_like(offset))

    log_center = _log_value(zero, zero)
    offset_x = _axis_offset(
        _log_value(zero, -one), log_center, _log_value(zero, one), x_index, width - 1
    )
    offset_y = _axis_offset(
        _log_value(-one, zero), log_center, _log_value(one, zero), y_index, height - 1
    )

    refined_x = (x_index.to(coords.dtype) + offset_x) / max(width - 1, 1)
    refined_y = (y_index.to(coords.dtype) + offset_y) / max(height - 1, 1)
    refined = torch.stack([refined_x, refined_y], dim=-1).clamp(0.0, 1.0)
    return refined.reshape(coords_shape)


def resize_heatmap_sequence(heatmaps: Tensor, target_size_hw: tuple[int, int]) -> Tensor:
    """Bilinearly resize ``(B, T, H, W)`` heatmaps/logits to ``target_size_hw``.

    Returns the input unchanged when the spatial size already matches.
    Reproduces the flatten → ``F.interpolate(mode="bilinear",
    align_corners=False)`` → restore blocks previously duplicated across
    ball-detection training and evaluation.
    """
    if heatmaps.ndim != 4:
        raise ValueError(
            f"Expected heatmaps with shape (B, T, H, W), got {tuple(heatmaps.shape)}."
        )
    height, width = _validate_size(target_size_hw)
    if tuple(heatmaps.shape[-2:]) == (height, width):
        return heatmaps
    batch_size, num_frames = heatmaps.shape[:2]
    flat = heatmaps.reshape(batch_size * num_frames, 1, *heatmaps.shape[-2:])
    flat = F.interpolate(
        flat,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return flat.reshape(batch_size, num_frames, height, width)


def _validate_size(size_hw: tuple[int, int]) -> tuple[int, int]:
    if len(size_hw) != 2:
        raise ValueError(f"size_hw must be (height, width), got {size_hw!r}.")
    height, width = int(size_hw[0]), int(size_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"size_hw must be positive, got {(height, width)}.")
    return height, width

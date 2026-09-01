"""Differentiable-free full-resolution raster targets for ground courts."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from src.tasks.court_alignment.geometry.court import (
    GROUND_COURT_KP14_COUNT,
    GroundCourtInstance,
    court_keypoints_for_instance,
    court_line_segments_for_instance,
)


def _validate_size(size_hw: tuple[int, int]) -> tuple[int, int]:
    if len(size_hw) != 2:
        raise ValueError("Ground raster size must be (height, width).")
    height, width = (int(value) for value in size_hw)
    if height <= 0 or width <= 0:
        raise ValueError("Ground raster dimensions must be positive.")
    return height, width


def _validate_sigma(sigma_px: float) -> float:
    sigma = float(sigma_px)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_px must be finite and positive.")
    return sigma


def _as_instances(instances: Sequence[GroundCourtInstance]) -> tuple[GroundCourtInstance, ...]:
    result = tuple(instances)
    if any(not isinstance(instance, GroundCourtInstance) for instance in result):
        raise TypeError("instances must contain GroundCourtInstance values.")
    return result


def _pixel_grid(
    height: int,
    width: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    y, x = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    return x, y


def render_court_line_mask(
    size_hw: tuple[int, int],
    instances: Sequence[GroundCourtInstance],
    *,
    line_width_px: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Render one binary ``[1,H,W]`` mask for one or more court instances.

    This uses a point-to-segment distance rasterizer rather than a library
    drawing primitive, so the exact same result is obtained on every worker.
    Values are exactly 0 or 1.  An empty instance sequence is valid and gives
    an all-zero input, which is useful for negative examples in future data
    variants.
    """

    height, width = _validate_size(size_hw)
    line_width = float(line_width_px)
    if not math.isfinite(line_width) or line_width <= 0.0:
        raise ValueError("line_width_px must be finite and positive.")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("Ground line masks require a floating-point dtype.")
    output = torch.zeros((height, width), dtype=dtype, device=device)
    grid_x, grid_y = _pixel_grid(height, width, dtype=dtype, device=device)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    half_width_sq = (line_width * 0.5) ** 2
    for instance in _as_instances(instances):
        segments = court_line_segments_for_instance(instance).to(
            dtype=dtype, device=output.device
        )
        for segment in segments:
            start, end = segment
            direction = end - start
            denominator = torch.dot(direction, direction).clamp_min(1.0e-12)
            relative = grid - start
            parameter = (relative * direction).sum(dim=-1) / denominator
            parameter = parameter.clamp(0.0, 1.0)
            closest = start + parameter.unsqueeze(-1) * direction
            distance_sq = ((grid - closest) ** 2).sum(dim=-1)
            output = torch.maximum(output, (distance_sq <= half_width_sq).to(dtype))
    return output.unsqueeze(0)


def render_keypoint_heatmaps(
    size_hw: tuple[int, int],
    keypoints_xy_px: Tensor,
    visibility: Tensor,
    *,
    sigma_px: float,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Render max-composed Gaussian heatmaps with shape ``[14,H,W]``.

    Args:
        keypoints_xy_px: Tensor ``[M,14,2]`` in pixel ``(x,y)`` order.
        visibility: Boolean tensor ``[M,14]``.  Invisible or out-of-canvas
            points contribute no Gaussian.
        sigma_px: Pixel-space Gaussian sigma; this is intentionally an
            absolute value so sigma ablations are reproducible across runs.
    """

    height, width = _validate_size(size_hw)
    sigma = _validate_sigma(sigma_px)
    if keypoints_xy_px.ndim != 3 or keypoints_xy_px.shape[1:] != (
        GROUND_COURT_KP14_COUNT,
        2,
    ):
        raise ValueError("keypoints_xy_px must have shape [M,14,2].")
    if visibility.shape != keypoints_xy_px.shape[:2] or visibility.dtype != torch.bool:
        raise ValueError("visibility must be boolean with shape [M,14].")
    if not keypoints_xy_px.is_floating_point() or not torch.empty(
        (), dtype=dtype
    ).is_floating_point():
        raise TypeError("keypoints and heatmap output must be floating point.")
    if not bool(torch.isfinite(keypoints_xy_px).all()):
        raise ValueError("keypoints_xy_px must be finite floating-point values.")
    points = keypoints_xy_px.to(dtype=dtype)
    if points.shape[0] == 0:
        return torch.zeros(
            (GROUND_COURT_KP14_COUNT, height, width),
            dtype=dtype,
            device=points.device,
        )
    grid_x, grid_y = _pixel_grid(height, width, dtype=dtype, device=points.device)
    dx = grid_x[None, None] - points[..., 0, None, None]
    dy = grid_y[None, None] - points[..., 1, None, None]
    heatmaps = torch.exp(-(dx.square() + dy.square()) / (2.0 * sigma * sigma))
    in_canvas = (
        (points[..., 0] >= 0.0)
        & (points[..., 0] <= float(width - 1))
        & (points[..., 1] >= 0.0)
        & (points[..., 1] <= float(height - 1))
    )
    valid = visibility & in_canvas
    heatmaps = heatmaps * valid[..., None, None].to(dtype)
    # Normalise each continuous Gaussian by its actual lattice maximum.  The
    # maximum is normally the nearest lattice cell, but using ``amax`` keeps
    # the contract correct for ties, tiny sigmas, and rectangular canvases
    # without changing the subpixel shape around the peak.
    lattice_maximum = heatmaps.amax(dim=(-2, -1), keepdim=True)
    heatmaps = torch.where(
        valid[..., None, None],
        heatmaps / lattice_maximum.clamp_min(torch.finfo(dtype).tiny),
        torch.zeros_like(heatmaps),
    )
    return heatmaps.amax(dim=0)


def render_keypoint_heatmaps_from_instances(
    size_hw: tuple[int, int],
    instances: Sequence[GroundCourtInstance],
    *,
    sigma_px: float,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convenience renderer returning heatmaps, points, and point visibility."""

    instances_tuple = _as_instances(instances)
    if instances_tuple:
        points = torch.stack(
            [court_keypoints_for_instance(instance) for instance in instances_tuple]
        ).to(dtype=dtype)
    else:
        points = torch.empty(
            (0, GROUND_COURT_KP14_COUNT, 2), dtype=dtype
        )
    height, width = _validate_size(size_hw)
    visibility = (
        (points[..., 0] >= 0.0)
        & (points[..., 0] <= float(width - 1))
        & (points[..., 1] >= 0.0)
        & (points[..., 1] <= float(height - 1))
    )
    return (
        render_keypoint_heatmaps(size_hw, points, visibility, sigma_px=sigma_px, dtype=dtype),
        points,
        visibility,
    )


def render_center_vote_targets(
    size_hw: tuple[int, int],
    keypoints_xy_px: Tensor,
    centers_xy_px: Tensor,
    visibility: Tensor,
    *,
    sigma_px: float,
    vote_radius_px: float = 3.0,
) -> tuple[Tensor, Tensor]:
    """Render per-pixel vectors towards the owning court center.

    The returned votes have shape ``[2,H,W]`` and are absolute pixel vectors
    ``center - pixel``.  The boolean mask has shape ``[1,H,W]``.  Every pixel
    in a ``vote_radius_px`` disk around a visible KP votes for that
    KP's court.  If disks overlap, the candidate with the smallest
    geometric distance wins deterministically (ties use flattened KP order).
    """

    height, width = _validate_size(size_hw)
    # Keep sigma in the API for ablation/config compatibility.  Ownership is
    # deliberately geometric, so changing Gaussian sigma cannot change either
    # the vote support or which court owns an overlap pixel.
    _validate_sigma(sigma_px)
    radius = float(vote_radius_px)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("vote_radius_px must be finite and positive.")
    if keypoints_xy_px.ndim != 3 or keypoints_xy_px.shape[1:] != (
        GROUND_COURT_KP14_COUNT,
        2,
    ):
        raise ValueError("keypoints_xy_px must have shape [M,14,2].")
    if centers_xy_px.shape != (keypoints_xy_px.shape[0], 2):
        raise ValueError("centers_xy_px must have shape [M,2].")
    if visibility.shape != keypoints_xy_px.shape[:2] or visibility.dtype != torch.bool:
        raise ValueError("visibility must be boolean with shape [M,14].")
    if not keypoints_xy_px.is_floating_point() or not centers_xy_px.is_floating_point():
        raise TypeError("Center-vote coordinates must be floating point.")
    if not bool(torch.isfinite(keypoints_xy_px).all()) or not bool(torch.isfinite(centers_xy_px).all()):
        raise ValueError("Center-vote coordinates must be finite.")
    points = keypoints_xy_px
    centers = centers_xy_px.to(device=points.device, dtype=points.dtype)
    if points.shape[0] == 0:
        return (
            torch.zeros((2, height, width), dtype=points.dtype, device=points.device),
            torch.zeros((1, height, width), dtype=torch.bool, device=points.device),
        )
    grid_x, grid_y = _pixel_grid(height, width, dtype=points.dtype, device=points.device)
    valid = visibility & (
        (points[..., 0] >= 0.0)
        & (points[..., 0] <= float(width - 1))
        & (points[..., 1] >= 0.0)
        & (points[..., 1] <= float(height - 1))
    )
    candidate_points = points.reshape(-1, 2)
    candidate_centers = centers[:, None].expand(-1, GROUND_COURT_KP14_COUNT, -1).reshape(-1, 2)
    candidate_valid = valid.reshape(-1)
    dx = grid_x[None] - candidate_points[:, 0, None, None]
    dy = grid_y[None] - candidate_points[:, 1, None, None]
    distance_sq = dx.square() + dy.square()
    candidate_support = candidate_valid[:, None, None] & (distance_sq <= radius * radius)
    # Derive validity directly from the geometric radius, never from the
    # Gaussian value (which can underflow for an unusually small sigma).
    vote_mask = candidate_support.any(dim=0)
    owner_distance = distance_sq.masked_fill(~candidate_support, float("inf"))
    _, best_index = owner_distance.min(dim=0)
    pixel = torch.stack((grid_x, grid_y), dim=-1)
    selected_centers = candidate_centers[best_index]
    votes = (selected_centers - pixel).permute(2, 0, 1)
    votes = torch.where(vote_mask.unsqueeze(0), votes, torch.zeros_like(votes))
    return votes, vote_mask.unsqueeze(0)


__all__ = [
    "render_center_vote_targets",
    "render_court_line_mask",
    "render_keypoint_heatmaps",
    "render_keypoint_heatmaps_from_instances",
]

"""Deterministic, plateau-aware sparse heatmap decoding.

Selection is non-differentiable. Selected scores retain their tensor dtype,
device and gradient; only the sparse candidate graph is inspected on CPU.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor


def heatmaps_to_peaks(
    heatmaps: Tensor,
    *,
    threshold: float,
    nms_kernel: int,
    max_peaks: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode regional maxima to normalized XY coordinates.

    Equal-valued, eight-connected plateaus represent one peak, not one peak
    per pixel. A plateau touching a higher value is not a regional maximum.
    Constant maps contain no localized evidence and produce no valid peaks.
    Ties are resolved by row-major pixel order, without modifying scores.

    ``threshold`` is inclusive. ``nms_kernel`` is a positive odd integer;
    distinct maxima within its Chebyshev radius suppress each other. The
    peak axis is always exactly ``max_peaks``, including for tiny maps.
    Missing slots have zero coordinates/scores and ``valid=False``.
    """
    if heatmaps.ndim < 2 or min(heatmaps.shape[-2:]) <= 0:
        raise ValueError("heatmaps must have non-empty shape (..., H, W).")
    if not heatmaps.is_floating_point():
        raise TypeError("heatmaps must have floating dtype.")
    if not bool(torch.isfinite(heatmaps).all()):
        raise ValueError("heatmaps must contain only finite values.")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative.")
    if type(nms_kernel) is not int or nms_kernel <= 0 or nms_kernel % 2 == 0:
        raise ValueError("nms_kernel must be a positive odd integer.")
    if type(max_peaks) is not int or max_peaks <= 0:
        raise ValueError("max_peaks must be a positive integer.")

    *leading_shape, height, width = heatmaps.shape
    count = math.prod(leading_shape) if leading_shape else 1
    flat = heatmaps.reshape(count, height * width)
    coords = heatmaps.new_zeros((count, max_peaks, 2))
    values = heatmaps.new_zeros((count, max_peaks))
    valid = torch.zeros((count, max_peaks), dtype=torch.bool, device=heatmaps.device)
    if count == 0:
        return (
            coords.reshape(*leading_shape, max_peaks, 2),
            values.reshape(*leading_shape, max_peaks),
            valid.reshape(*leading_shape, max_peaks),
        )

    maps = flat.reshape(count, 1, height, width)
    pooled = F.max_pool2d(maps, nms_kernel, stride=1, padding=nms_kernel // 2)
    # Reject unlocalized constant backgrounds, including a uniform map at
    # exactly the threshold. Do not perturb scores to break plateau ties.
    candidates = (
        (maps == pooled)
        & (maps >= threshold)
        & (maps > flat.amin(dim=1).view(count, 1, 1, 1))
    ).reshape(count, height * width)
    indices = candidates.nonzero(as_tuple=False)
    batch_indices, pixel_indices = indices.unbind(dim=1)
    candidate_scores = flat[batch_indices, pixel_indices]
    rows = torch.div(pixel_indices, width, rounding_mode="floor")
    columns = pixel_indices % width
    leaks = torch.zeros_like(pixel_indices, dtype=torch.bool)
    # A broad level region can have an interior that survives max pooling
    # even though its boundary touches a higher pixel. Reject the whole
    # candidate component if it reaches an equal-valued non-candidate.
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        yy, xx = rows + dy, columns + dx
        in_bounds = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        neighbors = yy.clamp(0, height - 1) * width + xx.clamp(0, width - 1)
        leaks |= (
            in_bounds
            & (flat[batch_indices, neighbors] == candidate_scores)
            & ~candidates[batch_indices, neighbors]
        )

    locations = cast(list[list[int]], indices.detach().cpu().tolist())
    scores_cpu = cast(list[float], candidate_scores.detach().cpu().tolist())
    leaks_cpu = cast(list[bool], leaks.detach().cpu().tolist())
    sparse = {
        (batch, pixel): index
        for index, (batch, pixel) in enumerate(locations)
    }
    visited: set[int] = set()
    representatives: list[list[tuple[float, int]]] = [[] for _ in range(count)]
    for index, (batch, pixel) in enumerate(locations):
        if index in visited:
            continue
        visited.add(index)
        stack = [index]
        representative = pixel
        rejected = False
        while stack:
            current = stack.pop()
            current_pixel = locations[current][1]
            representative = min(representative, current_pixel)
            rejected |= leaks_cpu[current]
            yy, xx = divmod(current_pixel, width)
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                ny, nx = yy + dy, xx + dx
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                neighbor = sparse.get((batch, ny * width + nx))
                if (
                    neighbor is not None
                    and neighbor not in visited
                    and scores_cpu[neighbor] == scores_cpu[index]
                ):
                    visited.add(neighbor)
                    stack.append(neighbor)
        if not rejected:
            representatives[batch].append((scores_cpu[index], representative))

    radius = nms_kernel // 2
    for batch, peaks in enumerate(representatives):
        retained: list[tuple[int, int]] = []
        for _, pixel in sorted(peaks, key=lambda item: (-item[0], item[1])):
            yy, xx = divmod(pixel, width)
            if any(max(abs(yy - y), abs(xx - x)) <= radius for y, x in retained):
                continue
            slot = len(retained)
            coords[batch, slot, 0] = xx / max(width - 1, 1)
            coords[batch, slot, 1] = yy / max(height - 1, 1)
            values[batch, slot] = flat[batch, pixel]
            valid[batch, slot] = True
            retained.append((yy, xx))
            if len(retained) == max_peaks:
                break

    return (
        coords.reshape(*leading_shape, max_peaks, 2),
        values.reshape(*leading_shape, max_peaks),
        valid.reshape(*leading_shape, max_peaks),
    )

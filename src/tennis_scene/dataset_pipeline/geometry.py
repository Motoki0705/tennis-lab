"""Measured-observation triangulation with explicit rejection and gap provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.tennis_scene.dataset_pipeline.quality import project
from src.utils.geometry.triangulation import solve_homogeneous_dlt


@dataclass(frozen=True)
class TriangulationSettings:
    max_reprojection_px: float
    max_speed_mps: float
    max_abs_xy_m: tuple[float, float]
    height_range_m: tuple[float, float]


@dataclass(frozen=True)
class TriangulationResult:
    position: np.ndarray
    valid: np.ndarray
    reprojection_px: np.ndarray
    rejection_code: np.ndarray


def triangulate_ball(
    uv: np.ndarray,
    visibility: np.ndarray,
    cameras: list[dict[str, Any]],
    *,
    size: tuple[int, int],
    fps: float,
    settings: TriangulationSettings,
) -> TriangulationResult:
    """Solve DLT only where at least two independent cameras support a frame.

    These are pseudo labels from approximate calibration, not measured 3D GT.
    Rejection codes: 1 insufficient views, 2 degenerate rays, 3 behind camera,
    4 outside physical bounds, 5 reprojection, 6 isolated excessive speed.
    """
    if uv.ndim != 3 or uv.shape[-1] != 2 or visibility.shape != uv.shape[:-1]:
        raise ValueError("Expected ball UV (V,T,2), visibility (V,T)")
    if visibility.dtype != bool or len(cameras) != len(uv):
        raise ValueError("Visibility must be boolean and cameras must match views")
    if not np.isfinite(uv[visibility]).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Observed coordinates and positive fps must be finite")
    matrices = np.stack(
        [np.asarray(c["K"]) @ np.column_stack((c["R"], c["t"])) for c in cameras]
    )
    count = visibility.sum(0)
    total = uv.shape[1]
    codes = np.where(count >= 2, 0, 1).astype(np.uint8)
    positions = np.full((total, 3), np.nan, np.float64)
    selected = np.flatnonzero(count >= 2)
    if len(selected):
        xy = np.where(visibility[..., None], uv, 0)[:, selected] * np.asarray(size)
        design = (
            np.stack(
                [
                    xy[..., 0, None] * matrices[:, None, 2] - matrices[:, None, 0],
                    xy[..., 1, None] * matrices[:, None, 2] - matrices[:, None, 1],
                ],
                axis=-2,
            )
            * visibility[:, selected, None, None]
        )
        design = design.transpose(1, 0, 2, 3).reshape(len(selected), -1, 4)
        solved, independent = solve_homogeneous_dlt(design)
        codes[selected[~independent]] = 2
        positions[selected[independent]] = solved[independent]
    errors = np.full(visibility.shape, np.nan)
    in_front_all = np.ones(total, bool)
    for view, camera in enumerate(cameras):
        projected, in_front = project(positions, camera)
        in_front_all &= ~visibility[view] | in_front
        errors[view] = np.where(
            visibility[view] & in_front,
            np.linalg.norm(projected - uv[view] * size, axis=-1),
            np.nan,
        )
    codes[(codes == 0) & ~in_front_all] = 3
    bounds = (
        (np.abs(positions[:, :2]) <= settings.max_abs_xy_m).all(-1)
        & (positions[:, 2] >= settings.height_range_m[0])
        & (positions[:, 2] <= settings.height_range_m[1])
    )
    codes[(codes == 0) & ~bounds] = 4
    mean_error = np.nansum(errors, axis=0) / np.maximum(1, count)
    codes[(codes == 0) & (mean_error > settings.max_reprojection_px)] = 5
    indices = np.flatnonzero(codes == 0)
    if len(indices) >= 3:
        speeds = (
            np.linalg.norm(np.diff(positions[indices], axis=0), axis=-1)
            * fps
            / np.diff(indices)
        )
        chord_speed = (
            np.linalg.norm(positions[indices[2:]] - positions[indices[:-2]], axis=-1)
            * fps
            / (indices[2:] - indices[:-2])
        )
        isolated = (
            (speeds[:-1] > settings.max_speed_mps)
            & (speeds[1:] > settings.max_speed_mps)
            & (chord_speed <= settings.max_speed_mps)
        )
        codes[indices[1:-1][isolated]] = 6
        if speeds[0] > settings.max_speed_mps:
            codes[indices[0]] = 6
        if speeds[-1] > settings.max_speed_mps:
            codes[indices[-1]] = 6
    valid = codes == 0
    positions[~valid] = np.nan
    return TriangulationResult(positions.astype(np.float32), valid, errors, codes)


def fill_triangulation_gaps(
    result: TriangulationResult,
    prior: np.ndarray,
    *,
    max_gap_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use bounded interior interpolation; retain the learned prior elsewhere.

    The returned source code is 0 learned prior, 1 triangulation, 2 interpolation.
    Only source 1 is a multi-camera-supported geometry label.
    """
    if prior.shape != result.position.shape or not np.isfinite(prior).all():
        raise ValueError("Finite learned prior must match the triangulated trajectory")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be nonnegative")
    output = prior.copy()
    sources: np.ndarray = result.valid.astype(np.uint8)
    output[result.valid] = result.position[result.valid]
    indices = np.flatnonzero(result.valid)
    for start, end in zip(indices[:-1], indices[1:], strict=True):
        if 1 < end - start <= max_gap_frames + 1:
            fraction = np.arange(1, end - start)[:, None] / (end - start)
            output[start + 1 : end] = (1 - fraction) * output[
                start
            ] + fraction * output[end]
            sources[start + 1 : end] = 2
    return output, sources

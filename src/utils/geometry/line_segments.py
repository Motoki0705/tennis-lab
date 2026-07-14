"""Deterministic finite-line extraction from binary or probabilistic maps."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RansacLineConfig:
    """Configuration for iterative finite-line extraction."""

    probability_threshold: float = 0.5
    max_iterations: int = 200
    distance_threshold_px: float = 2.0
    min_inliers: int = 20
    min_segment_length_px: float = 8.0
    max_lines: int = 12
    skeletonize: bool = True
    min_component_size: int = 10
    max_points: int = 4000

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("probability_threshold must be in [0, 1].")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if self.distance_threshold_px <= 0:
            raise ValueError("distance_threshold_px must be positive.")
        if self.min_inliers < 2:
            raise ValueError("min_inliers must be at least 2.")
        if self.min_segment_length_px <= 0:
            raise ValueError("min_segment_length_px must be positive.")
        if self.max_lines <= 0:
            raise ValueError("max_lines must be positive.")
        if self.min_component_size < 1:
            raise ValueError("min_component_size must be positive.")
        if self.max_points < self.min_inliers:
            raise ValueError("max_points must be >= min_inliers.")


@dataclass(frozen=True)
class LineExtractionDiagnostics:
    """Diagnostics kept separate from the model input contract."""

    input_point_count: int
    retained_point_count: int
    extracted_line_count: int
    mean_inlier_ratio: float
    mean_residual_px: float
    line_coverage: float


@dataclass(frozen=True)
class LineExtractionResult:
    """Fixed-length normalized segments and extractor-only diagnostics."""

    segments: NDArray[np.float32]
    diagnostics: LineExtractionDiagnostics


def canonicalize_segment(segment: NDArray[np.floating]) -> NDArray[np.float32]:
    """Return ``(u1,v1,u2,v2)`` with a stable endpoint order."""
    values = np.asarray(segment, dtype=np.float32)
    if values.shape != (4,):
        raise ValueError(f"segment must have shape (4,), got {values.shape}.")
    u1, v1, u2, v2 = (float(value) for value in values)
    swap = (u1, v1) > (u2, v2) if abs(u2 - u1) >= abs(v2 - v1) else (v1, u1) > (v2, u2)
    if swap:
        return np.asarray([u2, v2, u1, v1], dtype=np.float32)
    return np.asarray([u1, v1, u2, v2], dtype=np.float32)


def sort_and_pad_segments(
    segments: NDArray[np.floating],
    *,
    max_lines: int,
) -> NDArray[np.float32]:
    """Canonicalize, deterministically sort, truncate, and zero-pad segments."""
    values = np.asarray(segments, dtype=np.float32)
    if values.size == 0:
        values = np.empty((0, 4), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"segments must have shape (N, 4), got {values.shape}.")
    if max_lines <= 0:
        raise ValueError("max_lines must be positive.")

    canonical: list[NDArray[np.float32]] = []
    for segment in values:
        current = canonicalize_segment(segment)
        if not np.isfinite(current).all():
            raise ValueError("segments must contain only finite values.")
        du = float(current[2] - current[0])
        dv = float(current[3] - current[1])
        if np.hypot(du, dv) <= 0.0:
            continue
        canonical.append(current)

    def key(segment: NDArray[np.float32]) -> tuple[float, ...]:
        du = float(segment[2] - segment[0])
        dv = float(segment[3] - segment[1])
        orientation = 0.0 if abs(du) >= abs(dv) else 1.0
        center_u = float(segment[0] + segment[2]) * 0.5
        center_v = float(segment[1] + segment[3]) * 0.5
        length = float(np.hypot(du, dv))
        return (
            orientation,
            center_v,
            center_u,
            -length,
            *(float(value) for value in segment),
        )

    canonical.sort(key=key)
    output: NDArray[np.float32] = np.zeros((max_lines, 4), dtype=np.float32)
    if canonical:
        kept = np.stack(canonical[:max_lines], axis=0)
        output[: len(kept)] = kept
    return output


def extract_line_segments(
    line_map: NDArray[np.floating] | NDArray[np.integer] | NDArray[np.bool_],
    *,
    config: RansacLineConfig,
    rng: np.random.Generator | None = None,
) -> LineExtractionResult:
    """Extract normalized finite line segments with iterative RANSAC."""
    values = np.asarray(line_map)
    if values.ndim != 2:
        raise ValueError(f"line_map must have shape (H, W), got {values.shape}.")
    height, width = (int(values.shape[0]), int(values.shape[1]))
    if height <= 1 or width <= 1:
        raise ValueError("line_map height and width must both exceed 1.")
    if not np.isfinite(values).all():
        raise ValueError("line_map must contain only finite values.")

    generator = rng or np.random.default_rng(0)
    binary = (values >= config.probability_threshold).astype(np.uint8)
    input_point_count = int(binary.sum())
    binary = _filter_components(binary, config.min_component_size)
    if config.skeletonize:
        binary = _morphological_skeleton(binary)

    yx = np.argwhere(binary > 0)
    points = yx[:, [1, 0]].astype(np.float64, copy=False)
    if len(points) > config.max_points:
        selected = generator.choice(len(points), size=config.max_points, replace=False)
        points = points[np.sort(selected)]
    retained_point_count = int(len(points))
    remaining = points.copy()
    raw_segments: list[NDArray[np.float32]] = []
    inlier_ratios: list[float] = []
    residuals: list[float] = []
    covered_points = 0

    while len(raw_segments) < config.max_lines and len(remaining) >= config.min_inliers:
        model = _best_ransac_model(remaining, config=config, rng=generator)
        if model is None:
            break
        point, direction, inliers, residual = model
        projections = (remaining[inliers] - point) @ direction
        start = point + direction * float(projections.min())
        end = point + direction * float(projections.max())
        start[0] = np.clip(start[0], 0.0, float(width - 1))
        start[1] = np.clip(start[1], 0.0, float(height - 1))
        end[0] = np.clip(end[0], 0.0, float(width - 1))
        end[1] = np.clip(end[1], 0.0, float(height - 1))
        length = float(np.linalg.norm(end - start))
        if length < config.min_segment_length_px:
            remaining = remaining[~inliers]
            continue

        pixel_segment = canonicalize_segment(
            np.asarray([start[0], start[1], end[0], end[1]], dtype=np.float32)
        )
        pixel_segment[[0, 2]] /= float(width)
        pixel_segment[[1, 3]] /= float(height)
        raw_segments.append(pixel_segment)
        inlier_count = int(inliers.sum())
        inlier_ratios.append(float(inlier_count) / float(len(remaining)))
        residuals.append(residual)
        covered_points += inlier_count
        remaining = remaining[~inliers]

    padded = sort_and_pad_segments(
        np.asarray(raw_segments, dtype=np.float32),
        max_lines=config.max_lines,
    )
    diagnostics = LineExtractionDiagnostics(
        input_point_count=input_point_count,
        retained_point_count=retained_point_count,
        extracted_line_count=len(raw_segments),
        mean_inlier_ratio=float(np.mean(inlier_ratios)) if inlier_ratios else 0.0,
        mean_residual_px=float(np.mean(residuals)) if residuals else 0.0,
        line_coverage=(
            float(covered_points) / float(retained_point_count)
            if retained_point_count > 0
            else 0.0
        ),
    )
    return LineExtractionResult(segments=padded, diagnostics=diagnostics)


def _filter_components(binary: NDArray[np.uint8], min_size: int) -> NDArray[np.uint8]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_size:
            output[labels == label] = 1
    return output


def _morphological_skeleton(binary: NDArray[np.uint8]) -> NDArray[np.uint8]:
    image = (binary > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image) > 0:
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        skeleton = np.asarray(
            cv2.bitwise_or(skeleton, cv2.subtract(image, opened)), dtype=np.uint8
        )
        image = np.asarray(cv2.erode(image, element), dtype=np.uint8)
    return (skeleton > 0).astype(np.uint8)


def _best_ransac_model(
    points: NDArray[np.float64],
    *,
    config: RansacLineConfig,
    rng: np.random.Generator,
) -> (
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        float,
    ]
    | None
):
    best_inliers: NDArray[np.bool_] | None = None
    best_count = 0
    best_residual = float("inf")
    for _ in range(config.max_iterations):
        indices = rng.choice(len(points), size=2, replace=False)
        point = points[int(indices[0])]
        delta = points[int(indices[1])] - point
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-8:
            continue
        direction = delta / norm
        distances = np.abs(
            (points[:, 0] - point[0]) * direction[1]
            - (points[:, 1] - point[1]) * direction[0]
        )
        inliers = distances <= config.distance_threshold_px
        count = int(inliers.sum())
        if count < config.min_inliers:
            continue
        residual = float(distances[inliers].mean())
        if count > best_count or (count == best_count and residual < best_residual):
            best_inliers = inliers
            best_count = count
            best_residual = residual

    if best_inliers is None:
        return None
    inlier_points = points[best_inliers]
    center = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
    direction = vh[0]
    distances = np.abs(
        (points[:, 0] - center[0]) * direction[1]
        - (points[:, 1] - center[1]) * direction[0]
    )
    refined_inliers = distances <= config.distance_threshold_px
    if int(refined_inliers.sum()) < config.min_inliers:
        refined_inliers = best_inliers
        distances = np.abs(
            (points[:, 0] - center[0]) * direction[1]
            - (points[:, 1] - center[1]) * direction[0]
        )
    return center, direction, refined_inliers, float(distances[refined_inliers].mean())


__all__ = [
    "LineExtractionDiagnostics",
    "LineExtractionResult",
    "RansacLineConfig",
    "canonicalize_segment",
    "extract_line_segments",
    "sort_and_pad_segments",
]

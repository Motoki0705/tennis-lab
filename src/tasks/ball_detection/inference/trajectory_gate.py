"""Trajectory-consistency gate for ball-detection time series.

The public contract is pixel coordinates: callers must scale normalized
``(x, y)`` coordinates by ``(width - 1, height - 1)`` before calling this
module. Keeping the gate in pixel space makes ``max_residual_px`` independent
of the source video resolution normalization used elsewhere in the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrajectoryGateConfig:
    """Configuration for the local linear trajectory gate.

    The defaults are intentionally conservative for issue #620: tennis_clip
    normal visible-frame jumps had p99 around 20 px, while the observed
    scratch-checkpoint teleport was 179.9 px. ``max_residual_px=60`` sits well
    between those values, ``k_support=2`` gives a local constant-velocity fit
    without smoothing over longer rallies, ``max_support_gap=5`` avoids linking
    across long occlusions, and ``max_passes=2`` lets a first rejection stop a
    neighboring false detection from acting as support in the second pass.
    """

    enabled: bool = False
    max_residual_px: float = 60.0
    k_support: int = 2
    max_support_gap: int = 5
    max_passes: int = 2

    def __post_init__(self) -> None:
        """Validate gate parameters eagerly."""
        if not math.isfinite(self.max_residual_px) or self.max_residual_px <= 0:
            raise ValueError(
                "max_residual_px must be a positive finite value, "
                f"got {self.max_residual_px}"
            )
        if self.k_support <= 0:
            raise ValueError(f"k_support must be positive, got {self.k_support}")
        if self.max_support_gap <= 0:
            raise ValueError(
                f"max_support_gap must be positive, got {self.max_support_gap}"
            )
        if self.max_passes <= 0:
            raise ValueError(f"max_passes must be positive, got {self.max_passes}")


@dataclass(frozen=True)
class TrajectoryGateRejection:
    """Diagnostic record for one rejected frame."""

    frame_index: int
    pass_index: int
    residual_px: float
    previous_residual_px: float | None
    next_residual_px: float | None
    score: float


@dataclass(frozen=True)
class TrajectoryGateDiagnostics:
    """Diagnostics emitted by :func:`apply_trajectory_gate`."""

    rejected: tuple[TrajectoryGateRejection, ...]
    passes_run: int

    @property
    def rejected_indices(self) -> list[int]:
        """Return rejected frame indices in pass order."""
        return [record.frame_index for record in self.rejected]


def apply_trajectory_gate(
    positions_px: NDArray[np.floating[Any]],
    visibility: NDArray[np.bool_],
    score: NDArray[np.floating[Any]],
    *,
    max_residual_px: float = 60.0,
    k_support: int = 2,
    max_support_gap: int = 5,
    max_passes: int = 2,
) -> tuple[NDArray[np.bool_], TrajectoryGateDiagnostics]:
    """Reject visible detections that violate a local linear trajectory.

    Args:
        positions_px: Ball positions in pixel coordinates, shape ``(T, 2)``.
            Normalized coordinates are intentionally not accepted; scale by
            ``(width - 1, height - 1)`` at the call site.
        visibility: Boolean visibility mask, shape ``(T,)``.
        score: Detection confidence, shape ``(T,)``. Scores are validated and
            copied into diagnostics, but the gate decision is purely geometric.
        max_residual_px: Reject only when every available local prediction is
            farther than this many pixels. The default 60 px is between the
            tennis_clip normal-jump p99 (~20 px) and the issue #620 teleport
            example (179.9 px).
        k_support: Maximum visible support points to take from each side. The
            default 2 estimates local constant velocity without using distant
            rally history.
        max_support_gap: Maximum frame distance for a support point. The
            default 5 frames avoids linking across longer occlusion gaps while
            still covering short missed spans.
        max_passes: Number of rejection/re-evaluation passes. The default 2
            handles short runs of false detections by removing first-pass
            outliers from the support set before the second pass.

    Returns:
        A new boolean visibility mask and diagnostics. Inputs are never
        modified. A visible frame is kept when it has no support points, because
        isolated detections should remain available to downstream gap handling.

    Raises:
        ValueError: If shapes, dtypes, finiteness, or gate parameters are
            invalid.
    """
    config = TrajectoryGateConfig(
        enabled=True,
        max_residual_px=max_residual_px,
        k_support=k_support,
        max_support_gap=max_support_gap,
        max_passes=max_passes,
    )
    positions, current_visibility, scores = _validate_inputs(
        positions_px=positions_px,
        visibility=visibility,
        score=score,
    )

    rejections: list[TrajectoryGateRejection] = []
    passes_run = 0
    for pass_index in range(1, config.max_passes + 1):
        passes_run = pass_index
        visible_indices: NDArray[np.int64] = np.flatnonzero(
            current_visibility
        ).astype(np.int64)
        pass_rejections: list[TrajectoryGateRejection] = []

        for target_index_raw in visible_indices:
            target_index = int(target_index_raw)
            previous_support = _previous_support_indices(
                visible_indices=visible_indices,
                target_index=target_index,
                max_support_gap=config.max_support_gap,
                k_support=config.k_support,
            )
            next_support = _next_support_indices(
                visible_indices=visible_indices,
                target_index=target_index,
                max_support_gap=config.max_support_gap,
                k_support=config.k_support,
            )

            previous_residual = _support_residual(
                positions=positions,
                support_indices=previous_support,
                target_index=target_index,
            )
            next_residual = _support_residual(
                positions=positions,
                support_indices=next_support,
                target_index=target_index,
            )
            residuals = [
                residual
                for residual in (previous_residual, next_residual)
                if residual is not None
            ]
            if not residuals:
                continue
            if all(residual > config.max_residual_px for residual in residuals):
                pass_rejections.append(
                    TrajectoryGateRejection(
                        frame_index=target_index,
                        pass_index=pass_index,
                        residual_px=float(min(residuals)),
                        previous_residual_px=previous_residual,
                        next_residual_px=next_residual,
                        score=float(scores[target_index]),
                    )
                )

        if not pass_rejections:
            break
        for rejection in pass_rejections:
            current_visibility[rejection.frame_index] = False
        rejections.extend(pass_rejections)

    return current_visibility, TrajectoryGateDiagnostics(
        rejected=tuple(rejections),
        passes_run=passes_run,
    )


def _validate_inputs(
    *,
    positions_px: NDArray[np.floating[Any]],
    visibility: NDArray[np.bool_],
    score: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    positions = np.asarray(positions_px, dtype=np.float64)
    visibility_array = np.asarray(visibility)
    scores = np.asarray(score, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions_px must have shape (T, 2), got {positions.shape}")
    if visibility_array.ndim != 1:
        raise ValueError(
            f"visibility must have shape (T,), got {visibility_array.shape}"
        )
    if visibility_array.dtype != np.bool_:
        raise ValueError(f"visibility must have bool dtype, got {visibility_array.dtype}")
    if scores.ndim != 1:
        raise ValueError(f"score must have shape (T,), got {scores.shape}")
    if visibility_array.shape[0] != positions.shape[0]:
        raise ValueError(
            "visibility length must match positions_px length, "
            f"got {visibility_array.shape[0]} and {positions.shape[0]}"
        )
    if scores.shape[0] != positions.shape[0]:
        raise ValueError(
            "score length must match positions_px length, "
            f"got {scores.shape[0]} and {positions.shape[0]}"
        )
    if not np.isfinite(positions).all():
        raise ValueError("positions_px must contain only finite values")
    if not np.isfinite(scores).all():
        raise ValueError("score must contain only finite values")

    return positions, visibility_array.astype(np.bool_, copy=True), scores


def _previous_support_indices(
    *,
    visible_indices: NDArray[np.int64],
    target_index: int,
    max_support_gap: int,
    k_support: int,
) -> NDArray[np.int64]:
    support = visible_indices[
        (visible_indices < target_index)
        & ((target_index - visible_indices) <= max_support_gap)
    ]
    return support[-k_support:]


def _next_support_indices(
    *,
    visible_indices: NDArray[np.int64],
    target_index: int,
    max_support_gap: int,
    k_support: int,
) -> NDArray[np.int64]:
    support = visible_indices[
        (visible_indices > target_index)
        & ((visible_indices - target_index) <= max_support_gap)
    ]
    return support[:k_support]


def _support_residual(
    *,
    positions: NDArray[np.float64],
    support_indices: NDArray[np.int64],
    target_index: int,
) -> float | None:
    if support_indices.size == 0:
        return None

    expected_position = _predict_position(
        positions=positions,
        support_indices=support_indices,
        target_index=target_index,
    )
    residual = np.linalg.norm(positions[target_index] - expected_position)
    return float(residual)


def _predict_position(
    *,
    positions: NDArray[np.float64],
    support_indices: NDArray[np.int64],
    target_index: int,
) -> NDArray[np.float64]:
    if support_indices.size == 1:
        return positions[int(support_indices[0])].copy()

    support_times = support_indices.astype(np.float64)
    support_positions = positions[support_indices]
    time_mean = float(np.mean(support_times))
    centered_time = support_times - time_mean
    denominator = float(np.dot(centered_time, centered_time))
    if denominator == 0.0:
        raise ValueError(f"support_indices must be unique, got {support_indices}")

    position_mean = np.mean(support_positions, axis=0)
    slope = np.sum(
        centered_time[:, np.newaxis] * (support_positions - position_mean),
        axis=0,
    ) / denominator
    return position_mean + slope * (float(target_index) - time_mean)

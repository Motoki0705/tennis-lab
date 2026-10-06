"""Finite-difference kinematics restricted to free-flight segments.

A difference of order ``n`` anchored at frame ``k`` uses frames ``k..k+n``; it is
valid only when all of them belong to the same flight segment, so impacts never
enter a velocity, acceleration or jerk.  Predictions and ground truth are
differenced with the same stencil, so their comparison is exact in timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

_BINOMIAL = {1: (-1.0, 1.0), 2: (1.0, -2.0, 1.0), 3: (-1.0, 3.0, -3.0, 1.0)}


def segment_difference(
    positions: NDArray[np.floating],
    segment: NDArray[np.integer],
    fps: float,
    order: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Order-``order`` derivative estimate ``(T,3)`` and its validity ``(T,)``.

    ``segment`` labels each frame; negative labels never form a valid stencil.
    """
    if order not in _BINOMIAL:
        raise ValueError("order must be 1, 2 or 3")
    frames = len(positions)
    if positions.shape != (frames, 3) or segment.shape != (frames,):
        raise ValueError("Require positions (T,3) and segment labels (T,)")
    values = np.zeros((frames, 3))
    valid: NDArray[np.bool_] = np.zeros(frames, dtype=bool)
    if frames <= order:
        return values, valid
    anchors = frames - order
    same = segment[:anchors] >= 0
    for offset in range(1, order + 1):
        same &= segment[offset : offset + anchors] == segment[:anchors]
    total = np.zeros((anchors, 3))
    for offset, weight in enumerate(_BINOMIAL[order]):
        total += weight * positions[offset : offset + anchors]
    values[:anchors] = total * fps**order
    valid[:anchors] = same
    return values, valid


def stencil_any(mask: NDArray[np.bool_], order: int) -> NDArray[np.bool_]:
    """Whether any frame of the stencil anchored at each frame is in ``mask``."""
    result = mask.copy()
    for offset in range(1, order + 1):
        result[: len(mask) - offset] |= mask[offset:]
    return result


@dataclass(frozen=True)
class KinematicThresholds:
    """Limits that flag physically implausible predictions."""

    acceleration_mps2: float
    below_ground_m: float

    def __post_init__(self) -> None:
        if not self.acceleration_mps2 > 0 or not self.below_ground_m >= 0:
            raise ValueError("Require a positive acceleration and nonnegative depth")


def _rmse(values: NDArray[np.float64]) -> float | None:
    return float(np.sqrt(np.mean(values**2))) if len(values) else None


def kinematic_report(
    prediction: NDArray[np.floating],
    target: NDArray[np.floating],
    segment: NDArray[np.integer],
    fps: float,
    groups: dict[str, NDArray[np.bool_]],
    thresholds: KinematicThresholds,
) -> dict[str, Any]:
    """Velocity/acceleration errors, jerk ratio and implausibility rates.

    ``groups`` maps a name to a per-frame mask; a difference belongs to a group
    when any frame of its stencil does.  Errors are Euclidean, in m/s, m/s^2.
    """
    report: dict[str, Any] = {}
    differences = {}
    for order in (1, 2, 3):
        predicted, valid = segment_difference(prediction, segment, fps, order)
        truth, _ = segment_difference(target, segment, fps, order)
        differences[order] = (predicted, truth, valid)
    for name, mask in groups.items():
        entry: dict[str, Any] = {}
        for order, label in ((1, "velocity"), (2, "acceleration")):
            predicted, truth, valid = differences[order]
            selected = valid & stencil_any(mask, order)
            error = np.linalg.norm(predicted[selected] - truth[selected], axis=-1)
            entry[f"{label}_rmse"] = _rmse(error)
            entry[f"{label}_count"] = int(selected.sum())
        predicted, truth, valid = differences[3]
        selected = valid & stencil_any(mask, 3)
        truth_jerk = (
            np.linalg.norm(truth[selected], axis=-1).mean() if selected.any() else 0.0
        )
        entry["jerk_ratio"] = (
            float(np.linalg.norm(predicted[selected], axis=-1).mean() / truth_jerk)
            if truth_jerk > 0
            else None
        )
        predicted, _, valid = differences[2]
        selected = valid & stencil_any(mask, 2)
        magnitude = np.linalg.norm(predicted[selected], axis=-1)
        entry["implausible_acceleration_rate"] = (
            float(np.mean(magnitude > thresholds.acceleration_mps2))
            if len(magnitude)
            else None
        )
        entry["below_ground_rate"] = (
            float(np.mean(prediction[mask, 2] < -thresholds.below_ground_m))
            if mask.any()
            else None
        )
        report[name] = entry
    return report

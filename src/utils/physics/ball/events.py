"""Discrete event frames from per-frame probabilities, and their accuracy."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


def pick_event_frames(
    probability: NDArray[np.floating], threshold: float, min_separation: int
) -> NDArray[np.int64]:
    """Frames of local maxima above ``threshold``, at least ``min_separation`` apart.

    Peaks are accepted in decreasing probability; ties keep the earlier frame.
    """
    if probability.ndim != 1 or not 0 < threshold < 1 or min_separation < 1:
        raise ValueError("Require 1D probabilities, threshold in (0,1), separation>=1")
    padded = np.concatenate(([-np.inf], probability, [-np.inf]))
    peaks = np.flatnonzero(
        (probability >= threshold)
        & (probability >= padded[:-2])
        & (probability > padded[2:])
    )
    order = peaks[np.argsort(-probability[peaks], kind="stable")]
    accepted: list[int] = []
    for frame in order:
        if all(abs(int(frame) - other) >= min_separation for other in accepted):
            accepted.append(int(frame))
    return np.array(sorted(accepted), dtype=np.int64)


def match_event_offsets(
    predicted: NDArray[np.integer], truth: NDArray[np.integer], tolerance: int
) -> NDArray[np.int64]:
    """Signed offsets for the most matches, then the smallest total absolute error."""
    frames = np.unique(predicted.astype(np.int64))
    targets = np.unique(truth.astype(np.int64))
    offsets = frames[:, None] - targets[None, :]
    distances = np.abs(offsets)
    # One invalid pair costs more than all possible valid timing errors combined.
    penalty = min(len(frames), len(targets)) * tolerance + 1
    rows, columns = linear_sum_assignment(
        np.where(distances <= tolerance, distances, penalty)
    )
    return np.asarray(
        offsets[rows, columns][distances[rows, columns] <= tolerance], dtype=np.int64
    )


def event_detection_report(
    predicted: list[NDArray[np.integer]],
    truth: list[NDArray[np.integer]],
    tolerance: int,
) -> dict[str, Any]:
    """Optimal one-to-one matching within ``tolerance`` frames, pooled over rallies.

    ``segmentation_failure_rate`` is the share of rallies with any missed or
    spurious event, i.e. whose predicted flight segments differ from the truth.
    """
    if len(predicted) != len(truth) or tolerance < 0:
        raise ValueError("Require one prediction per rally and tolerance >= 0")
    matched, offsets, predicted_count, true_count, exact = 0, [], 0, 0, 0
    for frames, targets in zip(predicted, truth, strict=True):
        predicted_count += len(frames)
        true_count += len(targets)
        rally_offsets = match_event_offsets(frames, targets, tolerance)
        offsets.extend(rally_offsets.tolist())
        matched += len(rally_offsets)
        exact += len(rally_offsets) == len(frames) == len(targets)
    return {
        "tolerance_frames": tolerance,
        "predicted": predicted_count,
        "true": true_count,
        "matched": matched,
        "precision": matched / predicted_count if predicted_count else None,
        "recall": matched / true_count if true_count else None,
        "f1": (
            2 * matched / (predicted_count + true_count)
            if predicted_count + true_count
            else None
        ),
        # A rally is segmented correctly when every event, and nothing else, is found.
        "segmentation_failure_rate": 1 - exact / len(truth) if truth else None,
        "mean_abs_offset_frames": float(np.mean(np.abs(offsets))) if offsets else None,
        "mean_offset_frames": float(np.mean(offsets)) if offsets else None,
    }

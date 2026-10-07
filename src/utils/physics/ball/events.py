"""Discrete event frames from per-frame probabilities, and their accuracy."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


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


def event_detection_report(
    predicted: list[NDArray[np.integer]],
    truth: list[NDArray[np.integer]],
    tolerance: int,
) -> dict[str, Any]:
    """Greedy nearest matching within ``tolerance`` frames, pooled over rallies.

    ``segmentation_failure_rate`` is the share of rallies with any missed or
    spurious event, i.e. whose predicted flight segments differ from the truth.
    """
    if len(predicted) != len(truth) or tolerance < 0:
        raise ValueError("Require one prediction per rally and tolerance >= 0")
    matched, offsets, predicted_count, true_count, exact = 0, [], 0, 0, 0
    for frames, targets in zip(predicted, truth, strict=True):
        predicted_count += len(frames)
        true_count += len(targets)
        pairs = sorted(
            (abs(int(p) - int(t)), int(p), int(t)) for p in frames for t in targets
        )
        used_p: set[int] = set()
        used_t: set[int] = set()
        for distance, p, t in pairs:
            if distance <= tolerance and p not in used_p and t not in used_t:
                used_p.add(p)
                used_t.add(t)
                offsets.append(p - t)
        matched += len(used_t)
        exact += len(used_t) == len(frames) == len(targets)
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

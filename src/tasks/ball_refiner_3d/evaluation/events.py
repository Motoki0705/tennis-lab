"""Held-out refiner event detection metrics in frame and time units."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.events import EVENT_CLASS_BITS, EVENT_CLASSES
from src.tasks.ball_refiner_3d.inference.segmentation import (
    DEFAULT_PICKING,
    EventPicking,
    pick_class_event_frames,
)
from src.utils.physics.ball.events import event_detection_report


def event_report(
    probabilities: list[NDArray[np.floating]],
    truth: list[NDArray[np.uint8]],
    fps: float,
    *,
    picking: EventPicking = DEFAULT_PICKING,
    tolerance: int = 2,
) -> dict[str, Any]:
    """Match discrete peaks to the physical event bits, per rally and class."""
    peaks = [pick_class_event_frames(probs, picking) for probs in probabilities]
    report = event_detection_report(
        [np.unique(np.concatenate(list(frames.values()))) for frames in peaks],
        [np.flatnonzero(bits & sum(EVENT_CLASS_BITS)) for bits in truth],
        tolerance,
    )
    by_kind = {}
    for kind, bit in zip(EVENT_CLASSES, EVENT_CLASS_BITS, strict=True):
        result = event_detection_report(
            [frames[kind] for frames in peaks],
            [np.flatnonzero(bits & bit) for bits in truth],
            tolerance,
        )
        for key in ("mean_abs_offset", "mean_offset"):
            offset = result[f"{key}_frames"]
            result[f"{key}_s"] = None if offset is None else offset / fps
        by_kind[kind] = result
    return {
        **report,
        "threshold": picking.threshold,
        "min_separation_frames": picking.min_separation,
        "fps": fps,
        "by_kind": by_kind,
    }


def event_metrics(report: dict[str, Any]) -> dict[str, float]:
    """Expose defined per-class values in the compact training metrics."""
    metrics = {}
    for kind, values in report["by_kind"].items():
        for key in ("precision", "recall", "f1", "mean_abs_offset_s", "mean_offset_s"):
            if values[key] is not None:
                metrics[f"test_event_{kind}_{key}"] = float(values[key])
    return metrics

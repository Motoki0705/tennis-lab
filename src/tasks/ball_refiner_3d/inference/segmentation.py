"""Flight segmentation from predicted event probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.events import EVENT_CLASSES
from src.tasks.ball_refiner_3d.physics.targets import segments_from_events
from src.utils.physics.ball.events import pick_event_frames


@dataclass(frozen=True)
class EventPicking:
    """Peaks above ``threshold``, at least ``min_separation`` frames apart."""

    threshold: float = 0.5
    min_separation: int = 3


DEFAULT_PICKING = EventPicking()


def pick_class_event_frames(
    event_probability: NDArray[np.floating], picking: EventPicking = DEFAULT_PICKING
) -> dict[str, NDArray[np.int64]]:
    """Pick peaks independently in each (shot, bounce, net) channel of (T,3)."""
    if event_probability.ndim != 2 or event_probability.shape[1] != len(EVENT_CLASSES):
        raise ValueError("Require event probabilities (T,3), ordered shot, bounce, net")
    return {
        kind: pick_event_frames(values, picking.threshold, picking.min_separation)
        for kind, values in zip(EVENT_CLASSES, event_probability.T, strict=True)
    }


def predicted_segments(
    event_probability: NDArray[np.floating], picking: EventPicking = DEFAULT_PICKING
) -> NDArray[np.int64]:
    """Labels ``(T,)``: flights open at frame 0 and at every picked event."""
    peaks = pick_class_event_frames(event_probability, picking)
    frames = np.unique(np.concatenate(list(peaks.values())))
    return segments_from_events(frames, len(event_probability))

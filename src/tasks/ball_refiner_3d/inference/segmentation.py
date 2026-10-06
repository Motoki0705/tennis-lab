"""Flight segmentation from predicted event probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.physics.targets import segments_from_events
from src.utils.physics.ball.events import pick_event_frames


@dataclass(frozen=True)
class EventPicking:
    """Peaks above ``threshold``, at least ``min_separation`` frames apart."""

    threshold: float = 0.5
    min_separation: int = 3


DEFAULT_PICKING = EventPicking()


def predicted_segments(
    event_probability: NDArray[np.floating], picking: EventPicking = DEFAULT_PICKING
) -> NDArray[np.int64]:
    """Labels ``(T,)``: flights open at frame 0 and at every picked event."""
    frames = pick_event_frames(
        event_probability, picking.threshold, picking.min_separation
    )
    return segments_from_events(frames, len(event_probability))

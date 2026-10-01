"""The play region: the doubles court expanded by run-off margins, in court coordinates.

Court coordinates put the net on ``y = 0`` and the baselines on
``y = +-HALF_LENGTH``; ``x`` runs across the court. A rally player spends the
rally inside the play region on one side of the net, while spectators behind
the fence and people on adjacent courts are outside it or only briefly inside.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH


@dataclass(frozen=True)
class PlayRegionConfig:
    sideline_margin_m: float
    baseline_margin_m: float

    def __post_init__(self) -> None:
        if not (self.sideline_margin_m >= 0 and self.baseline_margin_m >= 0):
            raise ValueError(f"Play region margins must be nonnegative: {self}")


def in_play_region(points_xy: NDArray[np.floating], config: PlayRegionConfig) -> NDArray[np.bool_]:
    """``(...)`` whether court-plane points ``(..., 2)`` lie in the play region."""
    points = np.asarray(points_xy, np.float64)
    if points.shape[-1:] != (2,):
        raise ValueError("Points must be (..., 2)")
    inside: NDArray[np.bool_] = ((np.abs(points[..., 0]) <= HALF_DOUBLES_WIDTH + config.sideline_margin_m)
                                 & (np.abs(points[..., 1]) <= HALF_LENGTH + config.baseline_margin_m))
    return inside

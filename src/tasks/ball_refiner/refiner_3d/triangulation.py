"""The sole task boundary: consume #935 BallGMM2D, never raw detections."""

from __future__ import annotations

import numpy as np
from torch import Tensor

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.utils.geometry.probabilistic_triangulation import CameraGMM


def frame_observations(
    distribution: BallGMM2D, source_size_wh: Tensor, *, frame: int
) -> CameraGMM:
    """Treat B as synchronized cameras in caller-declared order; detach to CPU.

    Synchronization and camera identity belong to the caller. Uses the existing
    pixel_moments transformation; neither averaging nor top-1 selection occurs.
    """
    if not 0 <= frame < distribution.means.shape[1]:
        raise ValueError("Frame index is outside the refiner sequence")
    means, covariance = distribution.pixel_moments(source_size_wh)
    return CameraGMM(
        means[:, frame].detach().cpu().numpy().astype(np.float64),
        covariance[:, frame].detach().cpu().numpy().astype(np.float64),
        distribution.weights[:, frame].detach().cpu().numpy().astype(np.float64),
        distribution.presence_probability[:, frame]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
    )

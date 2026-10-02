"""Unfiltered maximum-weight component means of the immutable full GMM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.components.ball_refiner import BallRefiner2DOutput
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort


@dataclass(frozen=True)
class BallPointsOutput:
    """One finite refiner point for every frame; presence is diagnostic only."""

    camera_id: str
    source_size_wh: tuple[int, int]
    frame_indices: NDArray[np.int64]
    uv_px: NDArray[np.float32]
    presence_probability: NDArray[np.float64]

    def __post_init__(self) -> None:
        n = len(self.frame_indices)
        if (not self.camera_id or n < 1 or self.frame_indices.dtype != np.int64
                or not np.array_equal(self.frame_indices, np.arange(n))):
            raise ValueError("Ball points require a camera and the full source frame axis")
        if len(self.source_size_wh) != 2 or any(type(x) is not int or x < 2 for x in self.source_size_wh):
            raise ValueError("Ball points require source image size")
        if self.uv_px.shape != (n, 2) or self.presence_probability.shape != (n,):
            raise ValueError("Ball point array shape mismatch")
        if self.uv_px.dtype != np.float32 or self.presence_probability.dtype != np.float64:
            raise TypeError("Ball points require float32 coordinates and float64 presence")
        if not np.isfinite(self.uv_px).all() or not np.isfinite(self.presence_probability).all():
            raise ValueError("Ball points and presence must be finite")
        if np.any((self.presence_probability < 0) | (self.presence_probability > 1)):
            raise ValueError("Ball point presence must be in [0,1]")


class BallPointsModule:
    def __init__(self, *, distribution_version: int) -> None:
        if distribution_version not in {1, 2}:
            raise ValueError("Unsupported ball distribution version")
        self.io = ComponentIO("ball_points", BallRefiner2DOutput, BallPointsOutput,
                              {"distribution": InputPort("ball_distribution_2d", distribution_version)}, "ball_points", version=2)

    def process(self, inputs: BallRefiner2DOutput) -> BallPointsOutput:
        distribution = inputs.prediction.distribution
        if distribution.means.shape[0] != 1:
            raise ValueError("Ball points require exactly one camera distribution")
        # Argmax on logits is the maximum-weight component, first on ties.
        # Keep float64 source scaling before the point consumer's float32 cast.
        means = distribution.means.detach().cpu().double().numpy()[0]
        indices = distribution.mixture_logits.detach().cpu().numpy()[0].argmax(-1)
        point = means[np.arange(len(indices)), indices] * (np.array(inputs.source_size_wh) - 1)
        probability = distribution.presence_logits.detach().cpu().double().sigmoid().numpy()[0]
        return BallPointsOutput(inputs.camera_id, inputs.source_size_wh, inputs.frame_indices.copy(),
                                point.astype(np.float32), probability)

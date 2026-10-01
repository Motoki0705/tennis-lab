"""One explicit confidence-filtered point view of the immutable full GMM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_refiner.refiner_2d.confidence import (
    PointConfidenceRule,
    point_confidence,
)
from src.tennis_scene.pipeline.components.ball_refiner import BallRefiner2DOutput
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort


@dataclass(frozen=True)
class BallPointsOutput:
    camera_id: str
    source_size_wh: tuple[int, int]
    frame_indices: NDArray[np.int64]
    uv_px: NDArray[np.float32]
    confidence: NDArray[np.float32]
    observed: NDArray[np.bool_]
    presence_probability: NDArray[np.float64]
    area_px2: NDArray[np.float64]
    rejection_codes: NDArray[np.uint8]
    rule: PointConfidenceRule

    def __post_init__(self) -> None:
        n = len(self.frame_indices)
        if not self.camera_id or n < 1 or not np.array_equal(self.frame_indices, np.arange(n)):
            raise ValueError("Ball points require a camera and the full source frame axis")
        if len(self.source_size_wh) != 2 or min(self.source_size_wh) < 2:
            raise ValueError("Ball points require source image size")
        if self.uv_px.shape != (n, 2) or any(v.shape != (n,) for v in (
            self.confidence, self.observed, self.presence_probability, self.area_px2, self.rejection_codes,
        )):
            raise ValueError("Ball point array shape mismatch")
        expected = self.rule.rejection_codes(self.presence_probability, self.area_px2)
        if (not np.array_equal(self.rejection_codes, expected)
                or not np.array_equal(self.observed, expected == 0)):
            raise ValueError("Ball point missing mask differs from the confidence rule")
        if (not np.isfinite(self.uv_px).all() or not np.isfinite(self.confidence).all()
                or np.any(self.uv_px[~self.observed] != 0) or np.any(self.confidence[~self.observed] != 0)
                or not np.array_equal(self.confidence[self.observed], self.presence_probability[self.observed].astype(np.float32))):
            raise ValueError("Rejected ball points must remain missing, with zero coordinates/confidence")


class BallPointsModule:
    def __init__(self, rule: PointConfidenceRule, *, distribution_version: int) -> None:
        if distribution_version not in {1, 2}:
            raise ValueError("Unsupported ball distribution version")
        self.rule = rule
        self.io = ComponentIO("ball_points", BallRefiner2DOutput, BallPointsOutput,
                              {"distribution": InputPort("ball_distribution_2d", distribution_version)}, "ball_points")

    def process(self, inputs: BallRefiner2DOutput) -> BallPointsOutput:
        point, probability, area = point_confidence(inputs.prediction.distribution, inputs.source_size_wh)
        codes = self.rule.rejection_codes(probability[0], area[0])
        observed = codes == 0
        return BallPointsOutput(inputs.camera_id, inputs.source_size_wh, inputs.frame_indices.copy(),
                                np.where(observed[:, None], point[0], 0).astype(np.float32),
                                np.where(observed, probability[0], 0).astype(np.float32), observed,
                                probability[0], area[0], codes, self.rule)

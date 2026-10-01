"""Bind a point projection to exactly one source camera's full GMM timeline."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.tennis_scene.pipeline.components.ball_refiner import BallRefiner2DOutput
from src.tennis_scene.pipeline.contracts import AssemblyContext


@dataclass(frozen=True)
class BallPointsInputAssembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> BallRefiner2DOutput:
        row = artifacts["distribution"]
        if not isinstance(row, BallRefiner2DOutput):
            raise TypeError("Ball points require a refiner distribution; detector/annotation points are not accepted")
        if (row.camera_id != context.camera_id or row.source_size_wh != context.source.size
                or not np.array_equal(row.frame_indices, np.arange(context.source.num_frames))):
            raise ValueError("Ball distribution camera/source size/timeline mismatch")
        return row

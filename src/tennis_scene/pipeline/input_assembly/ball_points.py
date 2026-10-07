"""Bind observed detector points to one source camera and its complete frame axis."""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_points import BallPointsInput
from src.tennis_scene.pipeline.contracts import AssemblyContext


@dataclass(frozen=True)
class BallPointsInputAssembler:
    version: int = 2

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> BallPointsInput:
        row = artifacts["detections"]
        if not isinstance(row, BallDetectionOutput):
            raise TypeError("Ball points require detector observations")
        if row.camera_id != context.camera_id or not np.array_equal(row.frame_indices, np.arange(context.source.num_frames)):
            raise ValueError("Ball detection camera/timeline mismatch")
        return BallPointsInput(row, context.source.size)

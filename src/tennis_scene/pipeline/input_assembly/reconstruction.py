"""Join identity, camera, pose and ball artifacts on declared camera/time axes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.tennis_scene.pipeline.components.ball_reconstruction import (
    single_ball_observations,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentInput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.triangulation import TriangulationInput
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.input_assembly.observations import (
    gather_balls,
    gather_people,
    identified_people,
)


@dataclass(frozen=True)
class CameraAlignmentInputAssembler:
    human_threshold: float
    ball_threshold: float
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> CameraAlignmentInput:
        calibration: CourtCalibrationOutput = artifacts["calibration"]
        active = tuple(v.source_index for v in calibration.calibration.views)
        people = identified_people(gather_people(context.source, artifacts).select_views(active), artifacts["identities"], self.human_threshold)
        ball = single_ball_observations(gather_balls(context.source, artifacts).select_views(active), threshold=self.ball_threshold)
        return CameraAlignmentInput(context.source, calibration, artifacts["side"], people, ball)


@dataclass(frozen=True)
class PlayerTriangulationInputAssembler:
    human_threshold: float
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> TriangulationInput:
        calibration: CourtCalibrationOutput = artifacts["calibration"]
        active = tuple(v.source_index for v in calibration.calibration.views)
        grouped = identified_people(gather_people(context.source, artifacts).select_views(active), artifacts["identities"], self.human_threshold)
        return TriangulationInput(context.source, artifacts["alignment"], grouped)


@dataclass(frozen=True)
class BallTriangulationInputAssembler:
    ball_threshold: float
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> TriangulationInput:
        calibration: CourtCalibrationOutput = artifacts["calibration"]
        active = tuple(v.source_index for v in calibration.calibration.views)
        grouped = single_ball_observations(gather_balls(context.source, artifacts).select_views(active), threshold=self.ball_threshold)
        return TriangulationInput(context.source, artifacts["alignment"], grouped)

"""Join identity, camera, pose and ball artifacts on declared camera/time axes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.tennis_scene.pipeline.components.association_appearance import (
    SavedAssociationAppearance,
)
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    single_ball_observations,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentInput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import CourtSideInput
from src.tennis_scene.pipeline.components.identity import PlayerAssociationInput
from src.tennis_scene.pipeline.components.triangulation import TriangulationInput
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.input_assembly.observations import (
    gather_balls,
    gather_people,
    identified_people,
)
from src.utils.video.reader import OpenCVDecodeIdentity


@dataclass(frozen=True)
class CourtSideInputAssembler:
    ball_threshold: float
    version: int = 2

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> CourtSideInput:
        calibration: CourtCalibrationOutput = artifacts["calibration"]
        active = tuple(v.source_index for v in calibration.calibration.views)
        ball = single_ball_observations(gather_balls(context.source, artifacts).select_views(active), threshold=self.ball_threshold)
        return CourtSideInput(context.source, calibration, ball)


@dataclass(frozen=True)
class PlayerAssociationInputAssembler:
    reuse_clip_embeddings: bool = False
    version: int = 3
    expected_decoders: Mapping[str, OpenCVDecodeIdentity] = field(default_factory=dict)

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> PlayerAssociationInput:
        tracks = tuple(artifacts[f"tracks_{camera}"].tracks for camera in context.source.camera_ids)
        saved = {camera: SavedAssociationAppearance(artifacts[f"detections_{camera}"], artifacts[f"appearance_{camera}"])
                 for camera in context.source.camera_ids} if self.reuse_clip_embeddings else {}
        return PlayerAssociationInput(context.source, artifacts["calibration"], artifacts["side"], tracks, saved, self.expected_decoders)


@dataclass(frozen=True)
class CameraAlignmentInputAssembler:
    human_threshold: float
    ball_threshold: float
    version: int = 2

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
    version: int = 2

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> TriangulationInput:
        calibration: CourtCalibrationOutput = artifacts["calibration"]
        active = tuple(v.source_index for v in calibration.calibration.views)
        grouped = single_ball_observations(gather_balls(context.source, artifacts).select_views(active), threshold=self.ball_threshold)
        return TriangulationInput(context.source, artifacts["alignment"], grouped)

"""Bind source media and stored observations to preprocessing inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.tennis_scene.pipeline.components.ball_detection import BallDetectionInput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationInput,
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_kp import CourtDetectionInput
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionInput
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingInput
from src.tennis_scene.pipeline.components.pose_estimation import PoseEstimationInput
from src.tennis_scene.pipeline.contracts import AssemblyContext


@dataclass(frozen=True)
class BallDetectionInputAssembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> BallDetectionInput:
        if context.camera_id is None:
            raise ValueError("Ball detection requires a camera scope")
        return BallDetectionInput(context.source.video(context.camera_id))


@dataclass(frozen=True)
class CourtDetectionInputAssembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> CourtDetectionInput:
        if context.camera_id is None:
            raise ValueError("Court detection requires a camera scope")
        return CourtDetectionInput(context.source.video(context.camera_id))


@dataclass(frozen=True)
class CourtCalibrationInputAssembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> CourtCalibrationInput:
        return CourtCalibrationInput(context.source, tuple(artifacts[c] for c in context.source.camera_ids))


@dataclass(frozen=True)
class PersonDetectionInputAssembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> PersonDetectionInput:
        if context.camera_id is None:
            raise ValueError("Person detection requires a camera scope")
        calibration: CourtCalibrationOutput = artifacts["calibration"]
        polygon = calibration.footpoint_polygons[context.camera_id]
        return PersonDetectionInput(context.source.video(context.camera_id), polygon)


@dataclass(frozen=True)
class PersonTrackingInputAssembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> PersonTrackingInput:
        if context.camera_id is None:
            raise ValueError("Tracking requires a camera scope")
        return PersonTrackingInput(context.source.video(context.camera_id), artifacts["detections"])


@dataclass(frozen=True)
class PoseEstimationInputAssembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> PoseEstimationInput:
        if context.camera_id is None:
            raise ValueError("Pose requires a camera scope")
        return PoseEstimationInput(context.source.video(context.camera_id), artifacts["tracks"])

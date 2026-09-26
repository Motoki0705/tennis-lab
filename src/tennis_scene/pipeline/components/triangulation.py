"""Independent player and single-ball triangulation components."""

from __future__ import annotations

from dataclasses import dataclass

from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
    reconstruct_ball,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.identity import IDENTITIES_PORT
from src.tennis_scene.pipeline.components.player_reconstruction import (
    PlayerSkeleton,
    triangulate_players,
)
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.tennis_scene.pipeline.observation_types import GroupedObservations


@dataclass(frozen=True)
class TriangulationInput:
    source: ClipSource
    alignment: CameraAlignmentOutput
    observations: GroupedObservations


@dataclass(frozen=True)
class PlayerTriangulationOutput:
    skeleton: PlayerSkeleton | None


@dataclass(frozen=True)
class BallTriangulationOutput:
    ball: BallReconstructionResult | None


class PlayerTriangulationModule:
    def __init__(self, camera_ids: tuple[str, ...], *, reprojection_px: float, joint_confidence: float, enabled: bool = True) -> None:
        self.reprojection_px, self.joint_confidence, self.enabled = reprojection_px, joint_confidence, enabled
        self.io = ComponentIO("player_triangulation", TriangulationInput, PlayerTriangulationOutput,
            {"alignment": InputPort("aligned_cameras"), "calibration": InputPort("local_court_calibration"),
             "identities": IDENTITIES_PORT, **{f"pose_{c}": InputPort("person_poses") for c in camera_ids}}, "player_skeletons")

    def process(self, inputs: TriangulationInput) -> PlayerTriangulationOutput:
        geometry = inputs.alignment.geometry
        if geometry is None or not self.enabled:
            return PlayerTriangulationOutput(None)
        scale = inputs.source.pixel_threshold_scale
        return PlayerTriangulationOutput(triangulate_players(inputs.observations, geometry.cameras,
            reprojection_px=self.reprojection_px * scale, joint_confidence=self.joint_confidence))


class BallTriangulationModule:
    def __init__(self, camera_ids: tuple[str, ...], *, reprojection_px: float, min_frames: int, enabled: bool = True) -> None:
        self.reprojection_px, self.min_frames, self.enabled = reprojection_px, min_frames, enabled
        self.io = ComponentIO("ball_triangulation", TriangulationInput, BallTriangulationOutput,
            {"alignment": InputPort("aligned_cameras"), "calibration": InputPort("local_court_calibration"),
             **{f"ball_{c}": InputPort("ball_detections") for c in camera_ids}}, "ball_trajectory")

    def process(self, inputs: TriangulationInput) -> BallTriangulationOutput:
        geometry = inputs.alignment.geometry
        if geometry is None or not self.enabled:
            return BallTriangulationOutput(None)
        scale = inputs.source.pixel_threshold_scale
        return BallTriangulationOutput(reconstruct_ball(inputs.observations, geometry.cameras,
            fps=inputs.source.fps, reprojection_px=self.reprojection_px * scale, min_frames=self.min_frames))

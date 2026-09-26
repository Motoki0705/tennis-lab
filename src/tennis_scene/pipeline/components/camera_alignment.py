"""Validate the independently inferred court sides against matched observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.tennis_scene.pipeline.components.camera_geometry import (
    CameraGeometryConfig,
    CameraGeometryResult,
    SideEvidence,
    resolve_camera_geometry,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.identity import (
    IDENTITIES_PORT,
    SIDE_PORT,
    CourtSideOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices
from src.tennis_scene.pipeline.observation_types import GroupedObservations


@dataclass(frozen=True)
class CameraAlignmentInput:
    source: ClipSource
    calibration: CourtCalibrationOutput
    side: CourtSideOutput
    people: GroupedObservations
    balls: GroupedObservations


@dataclass(frozen=True)
class CameraAlignmentOutput:
    geometry: CameraGeometryResult | None


def observation_ports(camera_ids: tuple[str, ...]) -> dict[str, InputPort]:
    return {"calibration": InputPort("local_court_calibration"), "identities": IDENTITIES_PORT,
        **{f"pose_{c}": InputPort("person_poses") for c in camera_ids},
        **{f"ball_{c}": InputPort("ball_detections") for c in camera_ids}}


class CameraAlignmentModule:
    def __init__(self, camera_ids: tuple[str, ...], config: CameraGeometryConfig, *, player_reprojection_px: float,
                 ball_reprojection_px: float, joint_confidence: float, max_frames: int) -> None:
        self.config = config
        self.player_reprojection_px, self.ball_reprojection_px = player_reprojection_px, ball_reprojection_px
        self.joint_confidence, self.max_frames = joint_confidence, max_frames
        self.io = ComponentIO("camera_alignment", CameraAlignmentInput, CameraAlignmentOutput,
            {**observation_ports(camera_ids), "side": SIDE_PORT}, "aligned_cameras")

    def process(self, inputs: CameraAlignmentInput) -> CameraAlignmentOutput:
        if not inputs.people.visibility.any() and not inputs.balls.visibility.any():
            return CameraAlignmentOutput(None)
        if inputs.side.camera_ids != inputs.calibration.calibration.camera_ids or inputs.side.reference_camera != inputs.calibration.reference_camera:
            raise ValueError("Camera alignment requires side results for the declared cameras/reference")
        source = inputs.source
        sample = sampled_frame_indices(source.num_frames, source.fps, max_frames=self.max_frames)
        scale = float(np.hypot(*source.size) / np.hypot(1920, 1080))
        evidence: list[SideEvidence] = []
        if inputs.people.visibility.any():
            torso = [5, 6, 11, 12]
            evidence.append(SideEvidence("plcs", inputs.people.uv_px[:, :, sample][:, :, :, torso],
                (inputs.people.visibility & (inputs.people.confidence >= self.joint_confidence))[:, :, sample][:, :, :, torso], self.player_reprojection_px * scale))
        if inputs.balls.visibility.any():
            evidence.append(SideEvidence("ball", inputs.balls.uv_px[:, :, sample], inputs.balls.visibility[:, :, sample], self.ball_reprojection_px * scale))
        geometry = resolve_camera_geometry(inputs.calibration.calibration, inputs.calibration.reference_camera,
            (np.asarray(inputs.side.view_half_turns, bool),), tuple(evidence), config=self.config)
        return CameraAlignmentOutput(geometry)

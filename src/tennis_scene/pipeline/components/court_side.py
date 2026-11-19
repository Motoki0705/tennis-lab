"""Court side nodes using the half-turn hypothesis tests of ``src.tasks.court_side``."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.court_side.hypothesis import (
    CourtSideConfig,
    CourtSideDecision,
    CourtSideUndecided,
    HypothesisScore,
    PlayerCourtSideConfig,
    decide_court_side,
    judge_side_evidence,
)
from src.tasks.court_side.player import PlayerFootpoints, collect_player_side_evidence
from src.tasks.player_association.geometry.footpoints import (
    FootpointConfig,
    ground_footpoints,
)
from src.tennis_scene.pipeline.components.ball_points import BALL_POINTS_PORT
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.components.player_selection import PlayerSelectionModule
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices
from src.tennis_scene.pipeline.observation_types import GroupedObservations

COURT_SIDE = "court_side"
# Version 3 records every hypothesis' score and the decided half-turns.
SIDE_PORT = InputPort("court_side", 3)


@dataclass(frozen=True)
class CourtSideOutput:
    """Per-camera half-turn of the camera-local court relative to the reference.

    ``hypotheses`` holds every assignment's score, best first; ``frames`` is
    the number of scored multi-view frames in the chosen clip or video scope.
    """

    camera_ids: tuple[str, ...]
    reference_camera: str
    view_half_turns: tuple[bool, ...]
    hypotheses: tuple[HypothesisScore, ...]
    margin: float
    frames: int

    def __post_init__(self) -> None:
        if self.reference_camera not in self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("Invalid side artifact camera/reference IDs")
        if len(self.view_half_turns) != len(self.camera_ids) or any(type(v) is not bool for v in self.view_half_turns):
            raise ValueError("Side artifact must declare one boolean per camera")
        if self.view_half_turns[self.camera_ids.index(self.reference_camera)]:
            raise ValueError("Reference camera cannot be half-turned")
        if len(self.hypotheses) != 2 ** (len(self.camera_ids) - 1) or self.hypotheses[0].view_half_turns != self.view_half_turns:
            raise ValueError("Side artifact must score every hypothesis with the decided one first")
        if self.margin < 0 or self.frames < 1:
            raise ValueError("Side artifact needs a nonnegative margin and scored frames")


@dataclass(frozen=True)
class CourtSideInput:
    source: ClipSource
    calibration: CourtCalibrationOutput
    balls: GroupedObservations  # calibrated views only

    def sample(self, max_frames: int) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.bool_]]:
        source, calibration = self.source, self.calibration.calibration
        sample = sampled_frame_indices(source.num_frames, source.fps, max_frames=max_frames)
        count, views, frames, joints = self.balls.uv_px.shape[:4]
        if count > 1 or joints != 1 or views != len(calibration.views) or frames != source.num_frames:
            raise ValueError("Court side requires one ball stream over the calibrated cameras and the source timeline")
        if count:
            uv, visible = self.balls.uv_px[0][:, sample, 0], self.balls.visibility[0][:, sample, 0]
        else:
            uv, visible = np.zeros((views, len(sample), 2), np.float32), np.zeros((views, len(sample)), bool)
        return sample, np.ascontiguousarray(uv), np.ascontiguousarray(visible)

    def scaled_config(self, config: CourtSideConfig) -> CourtSideConfig:
        scale = self.source.pixel_threshold_scale
        return replace(config, reprojection_px=config.reprojection_px * scale, min_motion_px=config.min_motion_px * scale)


def court_side_io(camera_ids: tuple[str, ...]) -> ComponentIO[CourtSideInput, CourtSideOutput]:
    return ComponentIO(COURT_SIDE, CourtSideInput, CourtSideOutput,
            {"calibration": InputPort("local_court_calibration"), **{f"ball_{c}": BALL_POINTS_PORT for c in camera_ids}},
            SIDE_PORT.schema, SIDE_PORT.version)


@dataclass(frozen=True)
class PlayerCourtSideInput:
    source: ClipSource
    calibration: CourtCalibrationOutput
    tracks: tuple[PersonTrackingOutput, ...]  # calibrated views only

    def sample(
        self, max_frames: int, config: PlayerCourtSideConfig,
    ) -> tuple[NDArray[np.int64], tuple[PlayerFootpoints, ...]]:
        source = self.source
        indices = sampled_frame_indices(source.num_frames, source.fps, max_frames=max_frames)
        footpoint_config = FootpointConfig(config.bottom_border_px * source.pixel_threshold_scale)
        sampled = []
        for view, tracks in zip(self.calibration.calibration.views, self.tracks, strict=True):
            if tracks.camera_id != view.camera.camera_id or tracks.observed.shape[1] != source.num_frames:
                raise ValueError("Player court side requires tracks matching the calibrated cameras and source timeline")
            xy, valid = ground_footpoints(tracks.boxes_xyxy[:, indices], tracks.observed[:, indices],
                                          view.camera, source.size[1], footpoint_config)
            sampled.append(PlayerFootpoints(xy.transpose(1, 0, 2), valid.T))
        return indices, tuple(sampled)


def player_court_side_io(camera_ids: tuple[str, ...]) -> ComponentIO[PlayerCourtSideInput, CourtSideOutput]:
    selection = PlayerSelectionModule.io
    return ComponentIO(COURT_SIDE, PlayerCourtSideInput, CourtSideOutput,
        {"calibration": InputPort("local_court_calibration"),
         **{f"tracks_{camera}": InputPort(selection.output_schema, selection.version) for camera in camera_ids}},
        SIDE_PORT.schema, SIDE_PORT.version)


def _side_output(decision: CourtSideDecision) -> CourtSideOutput:
    return CourtSideOutput(decision.camera_ids, decision.reference_camera, decision.view_half_turns,
                           decision.hypotheses, decision.margin, decision.frames)


def _undecided_error(error: CourtSideUndecided, sampled_frames: int, *, player: bool = False) -> ReconstructionUnavailable:
    diagnostics = {
        "hypotheses": [{"view_half_turns": list(h.view_half_turns), "cost": h.cost, "support": h.support, "frames": h.frames}
                       for h in error.hypotheses],
        "frames": error.frames, "pair_frames": error.pair_frames, "sampled_frames": sampled_frames,
    }
    if player:
        diagnostics.update(evidence="player", margin=error.margin)
    return ReconstructionUnavailable(f"court_side_{error.reason}", str(error), diagnostics=diagnostics)


class CourtSideModule:
    """Pixel thresholds of ``config`` are configured at 1920x1080 and scaled to the source."""

    def __init__(self, camera_ids: tuple[str, ...], config: CourtSideConfig, *, max_frames: int) -> None:
        self.config, self.max_frames = config, max_frames
        self.io = court_side_io(camera_ids)

    def cache_identity(self) -> dict[str, Any]:
        return {'inputs': tuple(self.io.inputs), 'config': self.config, 'max_frames': self.max_frames}

    def process(self, inputs: CourtSideInput) -> CourtSideOutput:
        sample, uv, visible = inputs.sample(self.max_frames)
        config = inputs.scaled_config(self.config)
        try:
            decision = decide_court_side(tuple(v.camera for v in inputs.calibration.calibration.views),
                                         inputs.calibration.reference_camera, uv, visible, config)
        except CourtSideUndecided as undecided:
            raise _undecided_error(undecided, len(sample)) from undecided
        return _side_output(decision)


class PlayerCourtSideModule:
    def __init__(self, camera_ids: tuple[str, ...], config: PlayerCourtSideConfig, *, max_frames: int) -> None:
        self.config, self.max_frames = config, max_frames
        self.io = player_court_side_io(camera_ids)

    def cache_identity(self) -> dict[str, Any]:
        return {"evidence": "player", "inputs": tuple(self.io.inputs), "config": self.config, "max_frames": self.max_frames}

    def process(self, inputs: PlayerCourtSideInput) -> CourtSideOutput:
        indices, sampled = inputs.sample(self.max_frames, self.config)
        evidence, _ = collect_player_side_evidence(inputs.calibration.calibration.camera_ids,
                                                  inputs.calibration.reference_camera, sampled, self.config)
        try:
            return _side_output(judge_side_evidence(evidence, self.config))
        except CourtSideUndecided as undecided:
            raise _undecided_error(undecided, len(indices), player=True) from undecided

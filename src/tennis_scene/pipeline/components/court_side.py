"""Court side node: the ball-only half-turn hypothesis test of ``src.tasks.court_side``."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.tasks.court_side.hypothesis import (
    CourtSideConfig,
    CourtSideUndecided,
    HypothesisScore,
    decide_court_side,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices
from src.tennis_scene.pipeline.observation_types import GroupedObservations

COURT_SIDE = "court_side"
# Version 3: decided by the ball hypothesis test and records every hypothesis' score.
SIDE_PORT = InputPort("court_side", 3)


@dataclass(frozen=True)
class CourtSideOutput:
    """Per-camera half-turn of the camera-local court relative to the reference.

    ``hypotheses`` holds every assignment's score, best first; ``frames`` is
    the number of sampled frames with the ball in two or more views.
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


class CourtSideModule:
    """Pixel thresholds of ``config`` are configured at 1920x1080 and scaled to the source."""

    def __init__(self, camera_ids: tuple[str, ...], config: CourtSideConfig, *, max_frames: int) -> None:
        self.config, self.max_frames = config, max_frames
        self.io = ComponentIO(COURT_SIDE, CourtSideInput, CourtSideOutput,
            {"calibration": InputPort("local_court_calibration"), **{f"ball_{c}": InputPort("ball_detections") for c in camera_ids}},
            SIDE_PORT.schema, SIDE_PORT.version)

    def process(self, inputs: CourtSideInput) -> CourtSideOutput:
        source, calibration = inputs.source, inputs.calibration.calibration
        sample = sampled_frame_indices(source.num_frames, source.fps, max_frames=self.max_frames)
        count, views, frames, joints = inputs.balls.uv_px.shape[:4]
        if count > 1 or joints != 1 or views != len(calibration.views) or frames != source.num_frames:
            raise ValueError("Court side requires one ball stream over the calibrated cameras and the source timeline")
        if count:
            uv, visible = inputs.balls.uv_px[0][:, sample, 0], inputs.balls.visibility[0][:, sample, 0]
        else:
            uv, visible = np.zeros((views, len(sample), 2), np.float32), np.zeros((views, len(sample)), bool)
        scale = source.pixel_threshold_scale
        config = replace(self.config, reprojection_px=self.config.reprojection_px * scale, min_motion_px=self.config.min_motion_px * scale)
        try:
            decision = decide_court_side(tuple(v.camera for v in calibration.views), inputs.calibration.reference_camera,
                                         np.ascontiguousarray(uv), np.ascontiguousarray(visible), config)
        except CourtSideUndecided as undecided:
            raise ReconstructionUnavailable(f"court_side_{undecided.reason}", str(undecided), diagnostics={
                "hypotheses": [{"view_half_turns": list(h.view_half_turns), "cost": h.cost, "support": h.support, "frames": h.frames}
                               for h in undecided.hypotheses],
                "frames": undecided.frames, "pair_frames": undecided.pair_frames, "sampled_frames": len(sample)}) from undecided
        return CourtSideOutput(decision.camera_ids, decision.reference_camera, decision.view_half_turns,
                               decision.hypotheses, decision.margin, decision.frames)

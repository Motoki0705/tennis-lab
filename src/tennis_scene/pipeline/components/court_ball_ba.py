"""Optional ball-only camera/clock refinement; other scene geometry is unchanged."""

from __future__ import annotations

from dataclasses import dataclass

from src.tennis_scene.ball_calibration.bundle_adjustment import (
    BundleAdjustmentConfig,
    CalibrationRefinement,
    refine_court_ball,
)
from src.tennis_scene.ball_calibration.court import CourtViewObservations
from src.tennis_scene.pipeline.components.ball_points import BALL_POINTS_PORT
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.components.triangulation import TriangulationInput
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort


@dataclass(frozen=True)
class CourtBallBAInput(TriangulationInput):
    courts: tuple[CourtViewObservations, ...]
    half_turns: tuple[bool, ...]


@dataclass(frozen=True)
class CalibratedBallTriangulationOutput:
    ball: BallReconstructionResult | None
    calibration: CalibrationRefinement | None


class CourtBallBAModule:
    def __init__(self, camera_ids: tuple[str, ...], *, reprojection_px: float,
                 min_frames: int, config: BundleAdjustmentConfig, enabled: bool = True) -> None:
        self.reprojection_px, self.min_frames, self.enabled = reprojection_px, min_frames, enabled
        self.config = config
        self.io = ComponentIO(
            "ball_triangulation", CourtBallBAInput, CalibratedBallTriangulationOutput,
            {"alignment": InputPort("aligned_cameras"), "calibration": InputPort("local_court_calibration"),
             **{f"ball_{c}": BALL_POINTS_PORT for c in camera_ids}}, "ball_trajectory", 2,
        )

    def process(self, inputs: CourtBallBAInput) -> CalibratedBallTriangulationOutput:
        geometry = inputs.alignment.geometry
        if geometry is None or not self.enabled:
            return CalibratedBallTriangulationOutput(None, None)
        result = refine_court_ball(
            inputs.observations, geometry.cameras, inputs.courts, inputs.half_turns,
            fps=inputs.source.fps, reprojection_px=self.reprojection_px * inputs.source.pixel_threshold_scale,
            min_frames=self.min_frames, config=self.config,
        )
        return CalibratedBallTriangulationOutput(result.ball, result.calibration)

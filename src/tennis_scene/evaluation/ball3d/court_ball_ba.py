"""Fixed-input adapter to the production court/ball calibration stage."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.tennis_scene.ball_calibration.bundle_adjustment import (
    BundleAdjustmentConfig,
    refine_court_ball,
)
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    single_ball_observations,
)
from src.tennis_scene.pipeline.observation_types import ObjectObservations

from .contracts import Ball3DResult, MethodInput


@dataclass(frozen=True)
class CourtBallBAMethod:
    adjustment: BundleAdjustmentConfig = field(default_factory=BundleAdjustmentConfig)
    reprojection_px: float = 20.0
    min_frames: int = 5
    ball_threshold: float = 0.0

    @property
    def name(self) -> str:
        return "court_ball_ba"

    @property
    def requires_court_lines(self) -> bool:
        return False

    @property
    def config(self) -> Mapping[str, Any]:
        return {**asdict(self), "requires_court_lines": False,
                "input": "pinned_consensus_loess_points",
                "distortion": "fixed_zero", "principal_point": "fixed"}

    def __call__(self, inputs: MethodInput) -> Ball3DResult:
        uv = np.stack([view.points.uv_px for view in inputs.views])
        confidence = np.stack([view.points.confidence for view in inputs.views])
        observed = np.stack([view.points.observed for view in inputs.views])
        raw = ObjectObservations(
            inputs.camera_ids, inputs.source_size_wh, inputs.fps,
            uv[:, :, None, None], confidence[:, :, None, None],
            observed[:, :, None], np.zeros((len(inputs.views), 1), np.int64),
        )
        fitted = refine_court_ball(
            single_ball_observations(raw, threshold=self.ball_threshold),
            tuple(view.camera for view in inputs.views),
            tuple(view.court for view in inputs.views),
            tuple(view.half_turn for view in inputs.views),
            fps=inputs.fps,
            reprojection_px=self.reprojection_px * np.hypot(*inputs.source_size_wh) / np.hypot(1920, 1080),
            min_frames=self.min_frames, config=self.adjustment,
        )
        trajectory = fitted.ball.trajectory
        return Ball3DResult(
            inputs.camera_ids, inputs.frame_indices, trajectory.positions,
            trajectory.valid, trajectory.inliers,
            cameras=fitted.calibration.cameras,
            time_offsets_seconds=fitted.calibration.time_offsets_seconds,
            diagnostics={**fitted.calibration.diagnostics,
                         "fit_frame_indices": fitted.calibration.fit_frame_indices,
                         "rejection_codes": trajectory.reasons,
                         "accepted_uv_px": fitted.ball.uv_px,
                         "input_observed": observed,
                         "corrected_visibility": fitted.ball.visibility},
        )

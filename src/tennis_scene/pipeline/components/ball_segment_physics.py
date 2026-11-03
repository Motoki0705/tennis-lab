"""Opt-in physical reconstruction; the default triangulation is unchanged."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.tennis_scene.ball_trajectory.config import SegmentPhysicsConfig
from src.tennis_scene.ball_trajectory.fit import SegmentPhysicsResult, fit_trajectory
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_points import BALL_POINTS_PORT
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
    reconstruct_ball,
)
from src.tennis_scene.pipeline.components.triangulation import (
    BallTriangulationOutput,
    TriangulationInput,
)
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort
from src.utils.geometry.triangulation import TriangulatedPoints


@dataclass(frozen=True)
class SegmentPhysicsInput:
    triangulation: TriangulationInput
    detections: tuple[BallDetectionOutput, ...]

    def __post_init__(self) -> None:
        source = self.triangulation.source
        geometry = self.triangulation.alignment.geometry
        if geometry is None:
            return
        if tuple(row.camera_id for row in self.detections) != tuple(
            c.camera_id for c in geometry.cameras
        ):
            raise ValueError("Physics evidence camera axes disagree with alignment")
        for row in self.detections:
            if row.score_semantics != "model_score" or row.evidence is None:
                raise ValueError(
                    "Physics requires saved model evidence, never annotation inputs"
                )
            if row.evidence.source_size_wh != source.size or not np.array_equal(
                row.frame_indices, np.arange(source.num_frames)
            ):
                raise ValueError("Physics evidence source timeline/image axes disagree")


@dataclass(frozen=True)
class SegmentPhysicsOutput(BallTriangulationOutput):
    fit: SegmentPhysicsResult | None


class SegmentPhysicsModule:
    def __init__(
        self,
        camera_ids: tuple[str, ...],
        config: SegmentPhysicsConfig,
        *,
        reprojection_px: float,
        min_frames: int,
        enabled: bool = True,
    ) -> None:
        self.config, self.reprojection_px, self.min_frames, self.enabled = (
            config,
            reprojection_px,
            min_frames,
            enabled,
        )
        self.io = ComponentIO(
            "ball_triangulation",
            SegmentPhysicsInput,
            SegmentPhysicsOutput,
            {
                "alignment": InputPort("aligned_cameras"),
                "calibration": InputPort("local_court_calibration"),
                **{f"ball_{c}": BALL_POINTS_PORT for c in camera_ids},
                **{
                    f"detections_{c}": InputPort("ball_detections", 2)
                    for c in camera_ids
                },
            },
            "ball_trajectory",
        )

    def process(self, inputs: SegmentPhysicsInput) -> SegmentPhysicsOutput:
        data = inputs.triangulation
        geometry = data.alignment.geometry
        if geometry is None or not self.enabled:
            return SegmentPhysicsOutput(None, None)
        initial = reconstruct_ball(
            data.observations,
            geometry.cameras,
            fps=data.source.fps,
            reprojection_px=self.reprojection_px * data.source.pixel_threshold_scale,
            min_frames=self.min_frames,
        )
        evidence = tuple(row.evidence for row in inputs.detections)
        fit = fit_trajectory(
            geometry.cameras,
            tuple(e for e in evidence if e is not None),
            initial.trajectory,
            fps=data.source.fps,
            config=self.config,
        )
        reprojection = np.zeros_like(initial.trajectory.reprojection_px)
        for v, camera in enumerate(geometry.cameras):
            uv, front = camera.project(fit.positions)
            selected = fit.used_views[v] & front
            reprojection[v, selected] = np.linalg.norm(
                uv[selected] - initial.uv_px[v, selected], axis=1
            )
        reasons = np.where(fit.valid, 0, 7).astype(np.uint8)
        trajectory = TriangulatedPoints(
            fit.positions, fit.valid, reasons, fit.used_views, reprojection
        )
        return SegmentPhysicsOutput(
            BallReconstructionResult(
                initial.uv_px,
                initial.visibility,
                trajectory,
                "ok"
                if fit.diagnostics["status"] == "ok"
                else f"physics_{fit.diagnostics['status']}",
            ),
            fit,
        )

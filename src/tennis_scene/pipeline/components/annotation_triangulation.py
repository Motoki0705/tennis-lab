"""Equal-weight geometry of usable v3 positions, with no temporal or outlier gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.components.ball_annotations import (
    ANNOTATION_POINTS_PORT,
    BallAnnotationOutput,
)
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.triangulation import BallTriangulationOutput
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.utils.geometry.triangulation import (
    PointRejection,
    TriangulatedPoints,
    triangulate_multiview,
)

ANNOTATION_TRAJECTORY_PORT = InputPort("ball_annotation_trajectory", 1)
ANNOTATION_GEOMETRY_POLICY = {
    "weights": "one_per_usable_position_observed_and_estimated_equal",
    "outlier_rejection": "none",
    "smoothing": "none",
    "temporal_gates": "none",
    "solver": "all_view_dlt_and_pixel_least_squares",
    "invalid_geometry": "missing_with_reason_no_view_selection",
}


@dataclass(frozen=True)
class BallAnnotationTriangulationInput:
    source: ClipSource
    alignment: CameraAlignmentOutput
    points: tuple[BallAnnotationOutput, ...]  # calibrated camera order


@dataclass(frozen=True)
class BallAnnotationTriangulationOutput(BallTriangulationOutput):
    camera_ids: tuple[str, ...]
    position_usable: NDArray[np.bool_]
    observed: NDArray[np.bool_]
    estimated: NDArray[np.bool_]
    point_kind: tuple[tuple[str, ...], ...]
    used_views: NDArray[np.bool_]  # contributed to a valid 3D point
    missing_reasons: tuple[str, ...]
    annotations: tuple[BallAnnotationOutput, ...]
    policy: dict[str, str]

    def __post_init__(self) -> None:
        if self.ball is None:
            raise ValueError("Annotation triangulation must retain its input and missingness")
        views, frames = self.ball.visibility.shape
        masks = (self.position_usable, self.observed, self.estimated, self.used_views)
        if (len(self.camera_ids) != views or len(set(self.camera_ids)) != views
                or any(m.dtype != np.bool_ or m.shape != (views, frames) for m in masks)
                or not np.array_equal(self.position_usable, self.observed | self.estimated)
                or (self.observed & self.estimated).any()
                or not np.array_equal(self.ball.visibility, self.position_usable)
                or not np.array_equal(self.used_views, self.position_usable & self.ball.trajectory.valid[None])
                or not np.array_equal(self.ball.trajectory.inliers, self.used_views)
                or len(self.point_kind) != views or any(len(k) != frames for k in self.point_kind)
                or len(self.annotations) != views or len(self.missing_reasons) != frames
                or self.policy != ANNOTATION_GEOMETRY_POLICY):
            raise ValueError("Invalid annotation triangulation usage/provenance")
        if (tuple(p.camera_id for p in self.annotations) != self.camera_ids
                or self.point_kind != tuple(p.point_kind for p in self.annotations)
                or not np.array_equal(self.ball.uv_px, np.stack([p.uv_px for p in self.annotations]))
                or any(not np.array_equal(getattr(self, field), np.stack([getattr(p, field) for p in self.annotations]))
                       for field in ("position_usable", "observed", "estimated"))):
            raise ValueError("Triangulation fields differ from task-validated source annotations")


class BallAnnotationTriangulationModule:
    def __init__(self, camera_ids: tuple[str, ...]) -> None:
        self.io = ComponentIO("ball_triangulation", BallAnnotationTriangulationInput,
            BallAnnotationTriangulationOutput,
            {"alignment": InputPort("aligned_cameras"), "calibration": InputPort("local_court_calibration"),
             **{f"ball_{c}": ANNOTATION_POINTS_PORT for c in camera_ids}},
            ANNOTATION_TRAJECTORY_PORT.schema, ANNOTATION_TRAJECTORY_PORT.version)

    def process(self, inputs: BallAnnotationTriangulationInput) -> BallAnnotationTriangulationOutput:
        geometry = inputs.alignment.geometry
        if geometry is None or tuple(p.camera_id for p in inputs.points) != geometry.camera_ids:
            raise ValueError("Annotation triangulation requires fixed aligned cameras in point order")
        if tuple(c.camera_id for c in geometry.cameras) != geometry.camera_ids:
            raise ValueError("Aligned camera parameters disagree with the camera axis")
        uv = np.stack([p.uv_px for p in inputs.points])
        usable = np.stack([p.position_usable for p in inputs.points])
        # Every usable coordinate has weight 1. No pair consensus, reprojection
        # threshold, speed/height bound, minimum trajectory length or gap filling.
        fitted = triangulate_multiview(uv.transpose(1, 0, 2), usable.T.astype(np.float64),
            np.stack([c.matrix for c in geometry.cameras]), min_score=0.)
        enough = usable.sum(0) >= 2
        used = usable & fitted.valid[None]
        reasons = np.where(fitted.valid, int(PointRejection.VALID),
            np.where(enough, int(PointRejection.OPTIMIZATION_FAILED), int(PointRejection.INSUFFICIENT_VIEWS))).astype(np.uint8)
        missing = tuple("" if valid else "invalid_multiview_geometry" if supported else "insufficient_views"
                        for valid, supported in zip(fitted.valid, enough, strict=True))
        error = np.where(used, np.linalg.norm(fitted.reprojected.transpose(1, 0, 2) - uv, axis=-1), 0).astype(np.float32)
        trajectory = TriangulatedPoints(np.where(fitted.valid[:, None], fitted.points, 0).astype(np.float32),
                                       fitted.valid, reasons, used, error)
        status = "ok" if fitted.valid.any() else "ball_invalid_geometry" if enough.any() else "ball_insufficient_support"
        ball = BallReconstructionResult(uv, usable, trajectory, status)
        return BallAnnotationTriangulationOutput(ball, geometry.camera_ids, usable,
            np.stack([p.observed for p in inputs.points]), np.stack([p.estimated for p in inputs.points]),
            tuple(p.point_kind for p in inputs.points), used, missing,
            inputs.points, dict(ANNOTATION_GEOMETRY_POLICY))

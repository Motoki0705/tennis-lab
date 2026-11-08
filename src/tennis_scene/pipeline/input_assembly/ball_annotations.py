"""Bind task-validated annotation positions without using detector observed masks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.tennis_scene.pipeline.components.annotation_triangulation import (
    BallAnnotationTriangulationInput,
)
from src.tennis_scene.pipeline.components.ball_annotations import (
    BallAnnotationOutput,
    BallAnnotationPointsInput,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext


def _points(context: AssemblyContext, row: Any, camera_id: str) -> BallAnnotationOutput:
    if not isinstance(row, BallAnnotationOutput):
        raise TypeError("Annotation consumers require task-validated v3 ball positions")
    video = context.source.video(camera_id)
    source = row.annotation["source"]
    if (row.camera_id != camera_id or row.source_size_wh != context.source.size
            or not np.array_equal(row.frame_indices, np.arange(context.source.num_frames))
            or source["media_sha256"] != video.sha256):
        raise ValueError("Ball annotation source/camera/timeline mismatch")
    return row


@dataclass(frozen=True)
class BallAnnotationPointsInputAssembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> BallAnnotationPointsInput:
        if context.camera_id is None:
            raise ValueError("Annotation points require a source camera")
        return BallAnnotationPointsInput(_points(context, artifacts["detections"], context.camera_id))


@dataclass(frozen=True)
class BallAnnotationTriangulationInputAssembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> BallAnnotationTriangulationInput:
        calibration: CourtCalibrationOutput = artifacts["calibration"]
        alignment: CameraAlignmentOutput = artifacts["alignment"]
        if alignment.geometry is None:
            raise ValueError("Annotation triangulation requires fixed aligned geometry")
        active = tuple(view.source_index for view in calibration.calibration.views)
        cameras = tuple(context.source.camera_ids[i] for i in active)
        if (cameras != calibration.calibration.camera_ids or cameras != alignment.geometry.camera_ids
                or len(set(active)) != len(active) or len(cameras) < 2):
            raise ValueError("Fixed calibration/alignment camera axis mismatch")
        rows = tuple(_points(context, artifacts[f"ball_{c}"], c) for c in cameras)
        return BallAnnotationTriangulationInput(context.source, alignment, rows)

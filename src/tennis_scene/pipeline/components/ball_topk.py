"""Optional ball component consuming pre-gate evidence, without ball_points gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.tennis_scene.ball_association.topk import (
    AssociationResult,
    TopKConfig,
    associate_topk,
)
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.triangulation import BallTriangulationOutput
from src.tennis_scene.pipeline.contracts import (
    AssemblyContext,
    ClipSource,
    ComponentIO,
    InputPort,
)
from src.utils.geometry.triangulation import PointRejection, TriangulatedPoints


@dataclass(frozen=True)
class TopKBallInput:
    source: ClipSource
    alignment: CameraAlignmentOutput
    detections: tuple[BallDetectionOutput, ...]


@dataclass(frozen=True)
class TopKBallOutput(BallTriangulationOutput):
    association: AssociationResult | None


@dataclass(frozen=True)
class TopKBallInputAssembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> TopKBallInput:
        rows = []
        for camera in context.source.camera_ids:
            value = artifacts[f"detections_{camera}"]
            if not isinstance(value, BallDetectionOutput):
                raise TypeError("Top-K requires typed detector observations")
            if value.camera_id != camera or not np.array_equal(value.frame_indices, np.arange(context.source.num_frames)):
                raise ValueError("Top-K detector camera/frame axis mismatch")
            if (value.score_semantics != "model_score" or value.evidence is None
                    or value.evidence.source_size_wh != context.source.size):
                raise ValueError("Top-K requires source-sized model evidence; no fallback")
            rows.append(value)
        return TopKBallInput(context.source, artifacts["alignment"], tuple(rows))


class TopKBallTriangulationModule:
    def __init__(self, camera_ids: tuple[str, ...], config: TopKConfig, *, min_frames: int, enabled: bool = True) -> None:
        self.config, self.min_frames, self.enabled = config, min_frames, enabled
        # The base ball payload is unchanged; the optional component additionally
        # persists typed association evidence. The default component/schema is intact.
        self.io = ComponentIO("ball_triangulation", TopKBallInput, TopKBallOutput,
            {"alignment": InputPort("aligned_cameras"),
             **{f"detections_{c}": InputPort("ball_detections", 2) for c in camera_ids}}, "ball_trajectory")

    def process(self, inputs: TopKBallInput) -> TopKBallOutput:
        geometry = inputs.alignment.geometry
        if geometry is None or not self.enabled:
            return TopKBallOutput(None, None)
        by_id = {d.camera_id: d for d in inputs.detections}
        if len(by_id) != len(inputs.detections):
            raise ValueError("Duplicate top-K input camera IDs")
        evidence = tuple(by_id[c.camera_id].evidence for c in geometry.cameras)
        if any(e is None for e in evidence):
            raise ValueError("Top-K requires model evidence; no fallback")
        row = associate_topk(tuple(e for e in evidence if e is not None), geometry.cameras,
                             fps=inputs.source.fps, config=self.config, min_frames=self.min_frames)
        valid = row.valid.copy()
        used = (row.selected_indices >= 0) & valid[None]
        trajectory = TriangulatedPoints(np.where(valid[:, None], row.positions, 0).astype(np.float32), valid,
            np.where(valid, PointRejection.VALID, PointRejection.INSUFFICIENT_VIEWS).astype(np.uint8),
            used, np.where(used, row.reprojection_px, 0).astype(np.float32))
        ball = BallReconstructionResult(row.selected_uv_px, used, trajectory,
                                         "ok" if valid.any() else "ball_insufficient_support")
        return TopKBallOutput(ball, row)

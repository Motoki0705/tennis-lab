"""Opt-in dense evidence consumer; side/alignment still use their original ports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.tennis_scene.ball_likelihood import (
    DenseLikelihoodConfig,
    DenseTrajectory,
    reconstruct_dense,
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
from src.utils.geometry.triangulation import TriangulatedPoints


@dataclass(frozen=True)
class DenseBallInput:
    source: ClipSource
    alignment: CameraAlignmentOutput
    detections: tuple[BallDetectionOutput, ...]


@dataclass(frozen=True)
class DenseBallOutput(BallTriangulationOutput):
    dense: DenseTrajectory | None = None


@dataclass(frozen=True)
class DenseBallInputAssembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> DenseBallInput:
        alignment = artifacts["alignment"]
        geometry = alignment.geometry
        ids = () if geometry is None else geometry.camera_ids
        rows = tuple(artifacts[f"ball_{c}"] for c in ids)
        for camera_id, row in zip(ids, rows, strict=True):
            if row.camera_id != camera_id or row.evidence is None or row.score_semantics != "model_score":
                raise ValueError("Dense pipeline requires camera-bound model evidence")
            if not np.array_equal(row.frame_indices, np.arange(context.source.num_frames)) or row.evidence.source_size_wh != context.source.size:
                raise ValueError("Dense pipeline evidence source timeline/image mismatch")
        return DenseBallInput(context.source, alignment, rows)


class DenseBallModule:
    def __init__(self, camera_ids: tuple[str, ...], config: DenseLikelihoodConfig, *, enabled: bool = True) -> None:
        self.config, self.enabled = config, enabled
        self.io = ComponentIO("ball_triangulation", DenseBallInput, DenseBallOutput,
                              {"alignment": InputPort("aligned_cameras"),
                               **{f"ball_{c}": InputPort("ball_detections", 2) for c in camera_ids}}, "ball_trajectory")

    def process(self, inputs: DenseBallInput) -> DenseBallOutput:
        geometry = inputs.alignment.geometry
        if not self.enabled or geometry is None:
            return DenseBallOutput(None)
        if tuple(row.camera_id for row in inputs.detections) != geometry.camera_ids:
            raise ValueError("Dense component camera order mismatch")
        evidence = tuple(row.evidence for row in inputs.detections)
        if any(e is None for e in evidence):
            raise ValueError("Dense component is missing model heatmaps")
        # Explicit validation above, without a fallback to decoded/gated points.
        result = reconstruct_dense(tuple(e for e in evidence if e is not None), geometry.cameras,
                                   fps=inputs.source.fps, config=self.config)
        uv = np.zeros((len(evidence), inputs.source.num_frames, 2), np.float32)
        for i, camera in enumerate(geometry.cameras):
            projected, _ = camera.project(result.positions)
            uv[i] = np.where(result.used_views[i, :, None], projected, 0)
        # uv is the dense mode projection, not a new detector measurement.
        trajectory = TriangulatedPoints(result.positions, result.valid, result.rejection_codes, result.used_views,
                                         np.zeros(result.used_views.shape, np.float32))
        ball = BallReconstructionResult(uv, result.used_views, trajectory, "ok" if result.valid.any() else "ball_insufficient_support")
        return DenseBallOutput(ball, result)

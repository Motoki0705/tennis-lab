"""Optional evidence-based component; the default ball component is unchanged."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.ball_gmm.config import CandidateGMMConfig
from src.tennis_scene.ball_gmm.reconstruction import (
    CandidateGMMTrajectory,
    as_ball_result,
    reconstruct_candidates,
)
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.contracts import (
    AssemblyContext,
    ClipSource,
    ComponentIO,
    InputPort,
)


@dataclass(frozen=True)
class CandidateGMMInput:
    source: ClipSource
    alignment: CameraAlignmentOutput
    detections: tuple[BallDetectionOutput, ...]


@dataclass(frozen=True)
class CandidateGMMOutput:
    ball: BallReconstructionResult | None
    posterior: CandidateGMMTrajectory | None
    covariance: NDArray[np.float64] | None


@dataclass(frozen=True)
class CandidateGMMInputAssembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> CandidateGMMInput:
        detections = tuple(artifacts[f"detections_{c}"] for c in context.source.camera_ids)
        for camera, d in zip(context.source.camera_ids, detections, strict=True):
            if (not isinstance(d, BallDetectionOutput) or d.camera_id != camera
                    or not np.array_equal(d.frame_indices, np.arange(context.source.num_frames))
                    or d.evidence is None or d.evidence.source_size_wh != context.source.size):
                raise ValueError("Candidate GMM assembly requires camera/frame-bound model evidence")
        return CandidateGMMInput(context.source, artifacts["alignment"], detections)


class CandidateGMMModule:
    def __init__(self, camera_ids: tuple[str, ...], config: CandidateGMMConfig, *, enabled: bool = True):
        self.config, self.enabled = config, enabled
        self.io = ComponentIO("ball_triangulation", CandidateGMMInput, CandidateGMMOutput,
            {"alignment": InputPort("aligned_cameras"),
             **{f"detections_{c}": InputPort("ball_detections", 2) for c in camera_ids}}, "ball_trajectory")

    def process(self, inputs: CandidateGMMInput) -> CandidateGMMOutput:
        geometry = inputs.alignment.geometry
        if geometry is None or not self.enabled:
            return CandidateGMMOutput(None, None, None)
        ids = tuple(c.camera_id for c in geometry.cameras)
        selected = tuple(inputs.detections[inputs.source.camera_ids.index(c)] for c in ids)
        posterior = reconstruct_candidates(selected, geometry.cameras, config=self.config)
        ball, covariance = as_ball_result(posterior, geometry.cameras, config=self.config, fps=inputs.source.fps)
        return CandidateGMMOutput(ball, posterior, covariance)

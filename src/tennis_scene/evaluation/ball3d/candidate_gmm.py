"""Thin fixed-input adapter for the production candidate GMM algorithm."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from src.tennis_scene.ball_gmm.config import CandidateGMMConfig
from src.tennis_scene.ball_gmm.reconstruction import (
    point_estimate,
    reconstruct_candidates,
)

from .contracts import Ball3DResult, MethodInput


@dataclass(frozen=True)
class CandidateGMM:
    settings: CandidateGMMConfig

    @property
    def name(self) -> str:
        return f"candidate_gmm_{self.settings.estimator}_{self.settings.temporal}"

    @property
    def config(self) -> Mapping[str, Any]:
        return {**asdict(self.settings), "requires_court_lines": False}

    @property
    def requires_court_lines(self) -> bool:
        return False

    def __call__(self, inputs: MethodInput) -> Ball3DResult:
        posterior = reconstruct_candidates(tuple(v.detections for v in inputs.views),
            tuple(v.camera for v in inputs.views), config=self.settings)
        positions, covariance, views = point_estimate(posterior, config=self.settings, fps=inputs.fps)
        return Ball3DResult(inputs.camera_ids, inputs.frame_indices, positions, posterior.valid,
            views, covariance=covariance, diagnostics=posterior.diagnostics)

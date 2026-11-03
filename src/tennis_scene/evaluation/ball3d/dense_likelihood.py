"""Thin adapter to the same dense reconstruction used by the standard pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from src.tennis_scene.ball_likelihood import DenseLikelihoodConfig, reconstruct_dense

from .contracts import Ball3DResult, MethodInput


@dataclass(frozen=True)
class DenseHeatmapMethod:
    parameters: DenseLikelihoodConfig = DenseLikelihoodConfig()
    name: str = "dense_likelihood"
    requires_court_lines: bool = False

    @property
    def config(self) -> Mapping[str, Any]:
        return {**asdict(self.parameters), "requires_court_lines": False,
                "input": "pre_gate_dense_model_evidence", "loess": False}

    def __call__(self, inputs: MethodInput) -> Ball3DResult:
        evidence = tuple(v.detections.evidence for v in inputs.views)
        if any(e is None for e in evidence):
            raise ValueError("Dense method requires saved model heatmaps")
        result = reconstruct_dense(tuple(e for e in evidence if e is not None),
                                   tuple(v.camera for v in inputs.views), fps=inputs.fps, config=self.parameters)
        return Ball3DResult(inputs.camera_ids, inputs.frame_indices, result.positions, result.valid, result.used_views,
                            diagnostics={"log_likelihood": result.log_likelihood, "event_boundaries": result.event_boundaries,
                                         "solver_converged": result.solver_converged, "saturated_pixels": result.saturated_pixels,
                                         "rejection_codes": result.rejection_codes})

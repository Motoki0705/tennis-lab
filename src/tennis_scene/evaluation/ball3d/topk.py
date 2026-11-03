"""Thin evaluation adapter for the same top-K algorithm used by the pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from src.tennis_scene.ball_association.topk import TopKConfig, associate_topk

from .contracts import Ball3DResult, MethodInput


@dataclass(frozen=True)
class TopKMultiHypothesis:
    settings: TopKConfig = field(default_factory=TopKConfig)
    min_frames: int = 5

    @property
    def name(self) -> str:
        return "topk_multihypothesis"

    @property
    def requires_court_lines(self) -> bool:
        return False

    @property
    def config(self) -> Mapping[str, Any]:
        return {**asdict(self.settings), "min_frames": self.min_frames, "requires_court_lines": False,
                "score_semantics": "exp_negative_emission_cost_uncalibrated",
                "proposal_policy": "all_pairs_and_best_compatible_extensions",
                "temporal_policy": "second_order_beam_dp_bounded_missing_history"}

    def __call__(self, inputs: MethodInput) -> Ball3DResult:
        evidence = tuple(v.detections.evidence for v in inputs.views)
        if any(e is None for e in evidence):
            raise ValueError("Top-K requires model evidence in every selected view")
        row = associate_topk(tuple(e for e in evidence if e is not None),
                             tuple(v.camera for v in inputs.views), fps=inputs.fps, config=self.settings, min_frames=self.min_frames)
        return Ball3DResult(inputs.camera_ids, inputs.frame_indices, row.positions,
                            row.valid, row.selected_indices >= 0,
                            diagnostics={
                                "selected_candidate_indices": row.selected_indices,
                                "selected_candidate_uv_px": row.selected_uv_px,
                                "hypothesis_cost": row.hypothesis_cost,
                                "hypothesis_score": row.hypothesis_score,
                                "hypothesis_count": row.hypothesis_count,
                                "temporal_restart": row.temporal_restart,
                                "reprojection_px": row.reprojection_px,
                            })

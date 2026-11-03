"""Frozen-input adapter for the production segment/drag/clock solver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from src.tennis_scene.ball_trajectory.config import SegmentPhysicsConfig
from src.tennis_scene.ball_trajectory.fit import fit_trajectory
from src.utils.geometry.triangulation import TriangulatedPoints

from .baseline import PipelineBaseline
from .contracts import Ball3DResult, MethodInput


@dataclass(frozen=True)
class SegmentPhysicsMethod:
    settings: SegmentPhysicsConfig
    baseline: PipelineBaseline

    @property
    def name(self) -> str:
        return f"segment_physics_{self.settings.observation}"

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "physics": asdict(self.settings),
            "initialization": dict(self.baseline.config),
            "requires_court_lines": False,
        }

    @property
    def requires_court_lines(self) -> bool:
        return False

    def __call__(self, inputs: MethodInput) -> Ball3DResult:
        initial = self.baseline(inputs)
        evidence = tuple(v.detections.evidence for v in inputs.views)
        if any(e is None for e in evidence):
            raise ValueError("Physical replay requires model evidence")
        assert initial.diagnostics is not None
        trajectory = TriangulatedPoints(
            initial.positions,
            initial.valid,
            initial.diagnostics["rejection_codes"],
            initial.used_views,
            initial.diagnostics["accepted_uv_px"][..., 0] * 0,
        )
        fit = fit_trajectory(
            tuple(v.camera for v in inputs.views),
            tuple(e for e in evidence if e is not None),
            trajectory,
            fps=inputs.fps,
            config=self.settings,
        )
        return Ball3DResult(
            inputs.camera_ids,
            inputs.frame_indices,
            fit.positions,
            fit.valid,
            fit.used_views,
            time_offsets_seconds=fit.time_offsets_seconds,
            diagnostics=fit.diagnostics,
        )

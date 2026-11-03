"""Join real ball points and raw evidence using declared camera/frame axes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.tennis_scene.pipeline.components.ball_segment_physics import (
    SegmentPhysicsInput,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    BallTriangulationInputAssembler,
)


@dataclass(frozen=True)
class SegmentPhysicsInputAssembler:
    version: int = 1

    def assemble(
        self, context: AssemblyContext, artifacts: Mapping[str, Any]
    ) -> SegmentPhysicsInput:
        base = BallTriangulationInputAssembler(0.0).assemble(context, artifacts)
        geometry = base.alignment.geometry
        detections = (
            ()
            if geometry is None
            else tuple(artifacts[f"detections_{c.camera_id}"] for c in geometry.cameras)
        )
        return SegmentPhysicsInput(base, detections)

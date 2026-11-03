"""Join measured court and aligned ball observations on explicit view axes."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.tennis_scene.ball_calibration.court import court_views_from_diagnostics
from src.tennis_scene.pipeline.components.court_ball_ba import CourtBallBAInput
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    BallTriangulationInputAssembler,
)


@dataclass(frozen=True)
class CourtBallBAInputAssembler:
    ball_threshold: float
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> CourtBallBAInput:
        base = BallTriangulationInputAssembler(self.ball_threshold).assemble(context, artifacts)
        geometry = base.alignment.geometry
        if geometry is None:
            return CourtBallBAInput(base.source, base.alignment, base.observations, (), ())
        courts = court_views_from_diagnostics(
            artifacts["calibration"].court.diagnostics, context.source.camera_ids,
            context.source.size, context.source.num_frames,
        )
        by_id = {court.camera_id: court for court in courts}
        return CourtBallBAInput(
            base.source, base.alignment, base.observations,
            tuple(by_id[camera_id] for camera_id in geometry.camera_ids), geometry.view_half_turns,
        )

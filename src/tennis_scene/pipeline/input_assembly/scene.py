"""Input assembly for final scene export."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.scene_assembly import SceneAssemblyInput
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.input_assembly.observations import (
    gather_people,
    identified_people,
)


@dataclass(frozen=True)
class SceneAssemblyInputAssembler:
    human_threshold: float
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> SceneAssemblyInput:
        calibration: CourtCalibrationOutput = artifacts["calibration"]
        active = tuple(v.source_index for v in calibration.calibration.views)
        raw = gather_people(context.source, artifacts).select_views(active)
        grouped = identified_people(raw, artifacts["identities"], self.human_threshold)
        return SceneAssemblyInput(context.source, calibration, artifacts["alignment"], grouped,
            artifacts["skeleton"], artifacts["ball"], artifacts["placement"])

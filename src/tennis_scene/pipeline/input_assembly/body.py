"""Assemble body inputs from persisted choices and parameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.tennis_scene.pipeline.components.body_placement import BodyPlacementInput
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionInput,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.fasthmr import FastHMRInput
from src.tennis_scene.pipeline.components.gvhmr import GVHMRInput
from src.tennis_scene.pipeline.components.hmr2_features import HMR2FeaturesInput
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.input_assembly.observations import (
    gather_people,
    identified_people,
)


@dataclass(frozen=True)
class BodyViewSelectionInputAssembler:
    human_threshold: float
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> BodyViewSelectionInput:
        calibration: CourtCalibrationOutput = artifacts["calibration"]
        active = tuple(v.source_index for v in calibration.calibration.views)
        raw = gather_people(context.source, artifacts).select_views(active)
        return BodyViewSelectionInput(context.source, artifacts["alignment"], raw,
            identified_people(raw, artifacts["identities"], self.human_threshold))


@dataclass(frozen=True)
class HMR2FeaturesInputAssembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> HMR2FeaturesInput:
        return HMR2FeaturesInput(context.source, artifacts["selection"])


@dataclass(frozen=True)
class GVHMRInputAssembler:
    version: int = 2

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> GVHMRInput:
        return GVHMRInput(context.source, artifacts["selection"], artifacts["features"])


@dataclass(frozen=True)
class FastHMRInputAssembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> FastHMRInput:
        return FastHMRInput(context.source, artifacts["selection"])


@dataclass(frozen=True)
class BodyPlacementInputAssembler:
    human_threshold: float
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> BodyPlacementInput:
        calibration: CourtCalibrationOutput = artifacts["calibration"]
        active = tuple(v.source_index for v in calibration.calibration.views)
        raw = gather_people(context.source, artifacts).select_views(active)
        return BodyPlacementInput(context.source, artifacts["alignment"], raw,
            identified_people(raw, artifacts["identities"], self.human_threshold), artifacts["skeleton"], artifacts["recovered"])

"""Explicit per-stage X3D/CSN residual choices, separate from legacy artifacts."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from src.tasks.ball_detection.models.mdd.config import (
    MDDHeatmapConfig,
    _temporal_config_fields,
)


@dataclass(frozen=True)
class MDDStageHeatmapConfig(MDDHeatmapConfig):
    """Stages 1..5; the spatial stem and all downsample transitions are fixed."""

    stage_blocks: tuple[str, str, str, str, str]
    expansion_ratios: tuple[float, float, float, float, float]

    def __post_init__(self) -> None:
        MDDHeatmapConfig.__post_init__(self)
        if self.encoder_variant != "x3d_csn":
            raise ValueError("Per-stage blocks require the X3D/CSN encoder")
        if len(self.stage_blocks) != 5 or any(x not in {"ib", "convnext", "ffn"} for x in self.stage_blocks):
            raise ValueError("Declare five stage_blocks: ib, convnext or ffn")
        if len(self.expansion_ratios) != 5 or any(type(x) not in (int, float) or not math.isfinite(x) or x < 1 for x in self.expansion_ratios):
            raise ValueError("Declare five finite expansion_ratios >= 1")

    @classmethod
    def load(cls, path: Path) -> MDDStageHeatmapConfig:
        raw = _temporal_config_fields(path, "mdd_stage_dpt_heatmap", cls)
        raw["stage_blocks"] = tuple(raw["stage_blocks"])
        raw["expansion_ratios"] = tuple(raw["expansion_ratios"])
        return cls(**raw)

"""Layer-wise MSDA/CA query configuration; legacy mHC checkpoints stay explicit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.tasks.ball_detection.models.mdd.config import (
    MDDMHCQueryConfig,
    _temporal_config_fields,
)


@dataclass(frozen=True)
class MDDMultiScaleQueryConfig(MDDMHCQueryConfig):
    spatial_layers: tuple[str, ...]
    msda_levels: tuple[int, ...]
    msda_points: int
    msda_chunk_size: int
    msda_im2col_step: int
    checkpoint_scope: str = "encoder_msda"

    def __post_init__(self) -> None:
        MDDMHCQueryConfig.__post_init__(self)
        if len(self.spatial_layers) != self.layers or any(x not in {"C", "M"} for x in self.spatial_layers):
            raise ValueError("Declare one C or M for every query layer")
        if tuple(self.msda_levels) not in ((8, 16, 32, 64), (16, 32, 64)):
            raise ValueError("MSDA levels must be /8,/16,/32,/64 or /16,/32,/64 in order")
        if any(type(x) is not int or x < 1 for x in (self.msda_points, self.msda_chunk_size, self.msda_im2col_step)):
            raise ValueError("MSDA points, chunk size and im2col_step must be positive integers")
        if self.msda_chunk_size > self.msda_im2col_step and self.msda_chunk_size % self.msda_im2col_step:
            raise ValueError("MSDA chunk size must satisfy the official batch/im2col_step condition")
        if self.checkpoint_scope not in {"msda", "encoder_msda"}:
            raise ValueError("checkpoint_scope must be msda or encoder_msda")

    @classmethod
    def load(cls, path: Path) -> MDDMultiScaleQueryConfig:
        raw = _temporal_config_fields(path, "mdd_multiscale_query", cls,
                                      optional_defaults={"checkpoint_scope": "encoder_msda"})
        for key in ("spatial_layers", "msda_levels"):
            raw[key] = tuple(raw[key])
        return cls(**raw)

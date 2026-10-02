"""Explicit construction config; defaults live in configs/model/refiner_2d.yaml."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Refiner2DConfig:
    components: int
    hidden_dim: int
    attention_heads: int
    temporal_layers: int# Per stage, before and after context fusion.
    patch_size: int
    court_keypoints: int
    dropout: float
    pose_dropout: float
    min_std: float
    max_std: float
    initial_std: float
    max_correlation: float
    use_detector: bool
    use_pose: bool
    use_court: bool

    def __post_init__(self) -> None:
        for name in (
            "components",
            "hidden_dim",
            "attention_heads",
            "temporal_layers",
            "patch_size",
            "court_keypoints",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.patch_size % 2 != 1:
            raise ValueError("patch_size must be odd")
        for name in ("dropout", "pose_dropout"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not all(
            math.isfinite(x)
            for x in (
                self.min_std,
                self.initial_std,
                self.max_std,
                self.max_correlation,
            )
        ):
            raise ValueError("Covariance settings must be finite")
        if not 0 < self.min_std < self.initial_std < self.max_std:
            raise ValueError("Require 0 < min_std < initial_std < max_std")
        if not 0 < self.max_correlation < 1:
            raise ValueError("max_correlation must lie strictly between 0 and 1")
        for name in ("use_detector", "use_pose", "use_court"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class CandidateAnchoredConfig(Refiner2DConfig):
    """Adopted candidate-residual schema; absolute checkpoints keep their schema."""

    mean_parameterization: str
    anchored_components: int
    max_offset_uv: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mean_parameterization != 'candidate_residual_v1':
            raise ValueError('Unsupported mean_parameterization')
        if type(self.anchored_components) is not int or not 0 < self.anchored_components < self.components:
            raise ValueError('Require anchored components and at least one free component')
        if not math.isfinite(self.max_offset_uv) or not 0 < self.max_offset_uv < .5:
            raise ValueError('max_offset_uv must lie in (0, .5)')
        if not self.use_detector:
            raise ValueError('Candidate residual means require detector evidence')


def parse_model_config(values: dict[str, Any]) -> Refiner2DConfig:
    """Recognize two complete schemas without adding absent fields/defaults."""
    if 'mean_parameterization' in values:
        return CandidateAnchoredConfig(**values)
    return Refiner2DConfig(**values)

"""Construction-time configuration; no inference-time architecture selection."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Refiner2DConfig:
    components: int = 4
    hidden_dim: int = 128
    attention_heads: int = 4
    temporal_layers: int = 1  # Per stage, before and after context fusion.
    patch_size: int = 5
    court_keypoints: int = 20
    dropout: float = 0.1
    pose_dropout: float = 0.25
    min_std: float = 0.001
    max_std: float = 0.5
    initial_std: float = 0.05
    max_correlation: float = 0.95
    use_detector: bool = True
    use_pose: bool = True
    use_court: bool = True

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

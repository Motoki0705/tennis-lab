"""Explicit budgets and model selection; experiment matrices live outside src."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from src.tasks.ball_detection.models.mdd import MDDHeatmapConfig, MDDHeatmapHybridConfig


def load_model_config(path: Path, architecture: str) -> MDDHeatmapConfig:
    raw = yaml.safe_load(path.read_text())
    names = dict(dpt="mdd_dpt_heatmap", hybrid="mdd_heatmap_hybrid")
    if architecture not in names or not isinstance(raw, dict) or raw.pop("name", None) != names[architecture]:
        raise ValueError("Model config must match the explicitly selected heatmap architecture")
    cls = MDDHeatmapConfig if architecture == "dpt" else MDDHeatmapHybridConfig
    if set(raw) != set(cls.__dataclass_fields__):
        raise ValueError("Declare every heatmap model field, including frames and memory controls")
    for key in ("stem_channels", "mixed_channels", "residual_blocks"):
        raw[key] = tuple(raw[key])
    return cls(**raw)


@dataclass(frozen=True)
class UpdateBudget:
    updates: int
    input_frames: int
    microbatch_windows: int
    learning_rate: float
    warmup_updates: int
    minimum_lr_ratio: float
    weight_decay: float
    gradient_clip: float
    evaluation_interval: int
    checkpoint_interval: int
    seed: int
    sigma_ratio: float
    gamma: float

    def __post_init__(self) -> None:
        for key in ("updates", "input_frames", "microbatch_windows", "evaluation_interval", "checkpoint_interval"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be an explicit positive integer")
        if self.input_frames % 128:
            raise ValueError("Input frame budget must be an integer multiple of 128")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32)")
        if type(self.warmup_updates) is not int or not 0 <= self.warmup_updates < self.updates:
            raise ValueError("Warmup must be in [0, updates)")
        for key in ("learning_rate", "minimum_lr_ratio", "weight_decay", "gradient_clip", "sigma_ratio", "gamma"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"Nonfinite {key}")
        if min(self.learning_rate, self.gradient_clip, self.sigma_ratio) <= 0 or min(self.weight_decay, self.gamma) < 0:
            raise ValueError("Invalid optimizer or objective parameters")
        if not 0 <= self.minimum_lr_ratio <= 1:
            raise ValueError("Minimum LR ratio must be in [0,1]")

    def lr(self, step: int) -> float:
        if step < self.warmup_updates:
            return self.learning_rate * (step + 1) / self.warmup_updates
        progress = (step - self.warmup_updates) / max(self.updates - self.warmup_updates - 1, 1)
        return self.learning_rate * (self.minimum_lr_ratio + (1 - self.minimum_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2)

    def payload(self) -> dict[str, Any]:
        return dict(asdict(self), schedule="linear_warmup_cosine", optimizer="AdamW",
                    betas=[.9, .999], eps=1e-8, loss_normalization="sum_valid_frame_losses/sum_valid_frames_per_update")

"""Explicit budgets and model selection; experiment matrices live outside src."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from src.tasks.ball_detection.model_io.mdd_temporal import TemporalConfig
from src.tasks.ball_detection.models.mdd import (
    MDDHeatmapConfig,
    MDDHeatmapHybridConfig,
    MDDTemporalQueryConfig,
)


def load_model_config(path: Path, architecture: str) -> TemporalConfig:
    raw = yaml.safe_load(path.read_text())
    names = dict(
        dpt="mdd_dpt_heatmap", hybrid="mdd_heatmap_hybrid", query="mdd_temporal_query"
    )
    if (
        architecture not in names
        or not isinstance(raw, dict)
        or raw.pop("name", None) != names[architecture]
    ):
        raise ValueError(
            "Model config must match the explicitly selected heatmap architecture"
        )
    cls = dict(
        dpt=MDDHeatmapConfig,
        hybrid=MDDHeatmapHybridConfig,
        query=MDDTemporalQueryConfig,
    )[architecture]
    if set(raw) != set(cls.__dataclass_fields__):
        raise ValueError(
            "Declare every heatmap model field, including frames and memory controls"
        )
    for key in ("stem_channels", "mixed_channels", "residual_blocks"):
        raw[key] = tuple(raw[key])
    return cast(TemporalConfig, cls(**raw))


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
    schedule_updates: int
    schedule_offset: int
    initial_learning_rate: float
    freeze_updates: int
    encoder_lr_ratio: float
    optimizer_transfer: str

    def __post_init__(self) -> None:
        for key in (
            "updates",
            "input_frames",
            "microbatch_windows",
            "evaluation_interval",
            "checkpoint_interval",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be an explicit positive integer")
        if self.input_frames % 128:
            raise ValueError("Input frame budget must be an integer multiple of 128")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32)")
        if (
            type(self.warmup_updates) is not int
            or not 0 <= self.warmup_updates < self.updates
        ):
            raise ValueError("Warmup must be in [0, updates)")
        for key in (
            "learning_rate",
            "minimum_lr_ratio",
            "weight_decay",
            "gradient_clip",
            "sigma_ratio",
            "gamma",
        ):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"Nonfinite {key}")
        if (
            min(self.learning_rate, self.gradient_clip, self.sigma_ratio) <= 0
            or min(self.weight_decay, self.gamma) < 0
        ):
            raise ValueError("Invalid optimizer or objective parameters")
        if not 0 <= self.minimum_lr_ratio <= 1:
            raise ValueError("Minimum LR ratio must be in [0,1]")
        if (
            type(self.schedule_updates) is not int
            or type(self.schedule_offset) is not int
            or self.schedule_offset < 0
            or self.schedule_updates < self.schedule_offset + self.updates
            or self.warmup_updates >= self.schedule_updates
        ):
            raise ValueError(
                "Stage updates must lie within their explicit continuous schedule"
            )
        if (
            type(self.freeze_updates) is not int
            or not 0 <= self.freeze_updates < self.updates
        ):
            raise ValueError(
                "CNN freeze must be an explicit local update count followed by joint training"
            )
        if (
            not math.isfinite(self.initial_learning_rate)
            or not 0 < self.initial_learning_rate <= self.learning_rate
            or not math.isfinite(self.encoder_lr_ratio)
            or not 0 < self.encoder_lr_ratio <= 1
        ):
            raise ValueError("Invalid initial LR or encoder LR ratio")
        if self.optimizer_transfer not in {"fresh", "inherit"}:
            raise ValueError("Declare optimizer_transfer=fresh or inherit")

    def lr(self, step: int) -> float:
        step += self.schedule_offset
        if step < self.warmup_updates:
            if self.warmup_updates == 1:
                return self.learning_rate
            return self.initial_learning_rate + (
                self.learning_rate - self.initial_learning_rate
            ) * step / (self.warmup_updates - 1)
        progress = (step - self.warmup_updates) / max(
            self.schedule_updates - self.warmup_updates - 1, 1
        )
        return self.learning_rate * (
            self.minimum_lr_ratio
            + (1 - self.minimum_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        )

    def payload(self) -> dict[str, Any]:
        return dict(
            asdict(self),
            schedule="linear_warmup_cosine",
            optimizer="AdamW",
            betas=[0.9, 0.999],
            eps=1e-8,
            loss_normalization="sum_valid_frame_losses/sum_valid_frames_per_update",
        )

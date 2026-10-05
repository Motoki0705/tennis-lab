"""Strict configuration for the coordinate refiner experiments (#991/#1014)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, cast, get_type_hints

from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import CompileConfig
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT

T = TypeVar("T")


def parse_section(cls: type[T], raw: dict[str, Any]) -> T:
    """Reject missing/unknown fields and nonfinite numbers before constructing."""
    expected = {field.name for field in fields(cast(Any, cls))}
    if set(raw) != expected:
        raise ValueError(f"{cls.__name__}: missing={expected - set(raw)}, unknown={set(raw) - expected}")
    types = get_type_hints(cls)
    for key, value in raw.items():
        allowed = (int, float) if types[key] is float else (types[key],)
        if type(value) not in allowed:
            raise TypeError(f"Invalid type for {cls.__name__}.{key}: {type(value).__name__}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Nonfinite {cls.__name__}.{key}")
    return cls(**raw)


@dataclass(frozen=True)
class CorruptionConfig:
    event_probability: float
    isolated_probability: float
    gap_min: int
    gap_max: int
    noise_p95_px: float
    jitter_sigma_px: float
    outlier_probability: float
    triangulation_steps: int

    def __post_init__(self) -> None:
        if not 0 <= self.event_probability <= 1 or not 0 <= self.isolated_probability <= 1:
            raise ValueError("Missingness probabilities must be in [0,1]")
        if not 1 <= self.gap_min < self.gap_max:
            raise ValueError("Require distinct positive left/right gap widths")
        if not 0.05 < self.outlier_probability < 1:
            raise ValueError("Outlier probability must exceed 5% for the specified P95 mixture")
        if not 0 <= self.jitter_sigma_px < self.noise_p95_px / 5:
            raise ValueError("Jitter must be small relative to the positive noise P95")
        if self.triangulation_steps < 0:
            raise ValueError("triangulation_steps must be nonnegative")


@dataclass(frozen=True)
class ModelConfig:
    dimensions: int
    architecture: str
    width: int
    layers: int
    heads: int
    dropout: float
    window_length: int
    flow_steps: int

    def __post_init__(self) -> None:
        if self.dimensions not in (2, 3) or self.architecture not in ("regression", "flow"):
            raise ValueError("Require dimensions=2|3 and architecture=regression|flow")
        if self.architecture == "flow" and self.dimensions != 3:
            raise ValueError("The diffusion comparison is defined for 3D")
        if self.width < 8 or self.heads < 1 or self.width % self.heads or self.width % 2:
            raise ValueError("width must be even and divisible by heads")
        if min(self.layers, self.flow_steps) < 1 or self.window_length < 3 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid temporal model configuration")


@dataclass(frozen=True)
class TrainingConfig:
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    evaluate_every: int
    log_every: int
    gan_weight: float
    gan_warmup_steps: int
    cpu_threads: int

    def __post_init__(self) -> None:
        if min(self.steps, self.batch_size, self.evaluate_every, self.log_every, self.cpu_threads) < 1:
            raise ValueError("Training counts must be positive")
        if min(self.learning_rate, self.gradient_clip) <= 0 or min(self.weight_decay, self.gan_weight, self.gan_warmup_steps) < 0:
            raise ValueError("Invalid optimizer/GAN settings")


def resolved_config(config: DictConfig, expected: set[str]) -> tuple[dict[str, Any], PathResolver]:
    raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"Require exactly the configuration sections {sorted(expected)}")
    raw = cast(dict[str, Any], raw)
    roots = RuntimePathRoots.from_mapping(raw["paths"], repository_root=PROJECT_ROOT)
    return raw, PathResolver(roots)


def training_config(config: DictConfig) -> tuple[dict[str, Any], Path, Path]:
    raw, resolver = resolved_config(config, {"paths", "data", "corruption", "model", "training", "run", "compile"})
    CompileConfig.from_mapping(raw["compile"])
    model = parse_section(ModelConfig, raw["model"])
    train = parse_section(TrainingConfig, raw["training"])
    parse_section(CorruptionConfig, raw["corruption"])
    if model.architecture == "flow" and train.gan_weight:
        raise ValueError("GAN is only used by the direct regression comparison")
    if set(raw["data"]) != {"dataset", "evaluation_event_probability", "evaluation_seed"}:
        raise ValueError("Invalid data configuration")
    if not 0 <= raw["data"]["evaluation_event_probability"] <= 1:
        raise ValueError("Invalid evaluation event probability")
    if set(raw["run"]) != {"output_dir", "seed", "device"} or raw["run"]["device"] not in ("cpu", "cuda"):
        raise ValueError("Require explicit run output, seed, and cpu/cuda device")
    if min(raw["run"]["seed"], raw["data"]["evaluation_seed"]) < 0:
        raise ValueError("Seeds must be nonnegative")
    dataset = resolver.resolve(PathRole.DATA, raw["data"]["dataset"])
    output = resolver.resolve(PathRole.OUTPUT, raw["run"]["output_dir"])
    if output == dataset or output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Training output must be separate from the dataset")
    raw["paths"] = {name: str(value) for name, value in asdict(resolver.roots).items()}
    return raw, dataset, output


def validate_training(config: DictConfig) -> None:
    training_config(config)

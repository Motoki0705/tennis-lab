"""Strict configuration for the coordinate refiner experiments (#991/#1014)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, cast, get_type_hints

from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import CompileConfig
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.models.components.ffn_layers import SUPPORTED_FFN_TYPES
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
        if self.noise_p95_px == 0:
            if self.jitter_sigma_px != 0 or self.outlier_probability != 0:
                raise ValueError("Zero noise requires zero jitter and zero outlier probability")
        elif not 0.05 < self.outlier_probability < 1:
            raise ValueError("Outlier probability must exceed 5% for the specified P95 mixture")
        elif not 0 <= self.jitter_sigma_px < self.noise_p95_px / 5:
            raise ValueError("Jitter must be small relative to the positive noise P95")
        if self.triangulation_steps < 0:
            raise ValueError("triangulation_steps must be nonnegative")


@dataclass(frozen=True)
class LegacyModelConfig:
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
class RoPEModelConfigV2(LegacyModelConfig):
    """The v2 checkpoint contract has a fixed SwiGLU FFN."""

    ffn_dim: int
    rope_dim: int
    rope_theta: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ffn_dim < 1 or self.rope_dim < 2 or self.rope_dim % 2 or self.rope_dim > self.width // self.heads or self.rope_theta <= 0:
            raise ValueError("Invalid FFN or RoPE dimensions")


@dataclass(frozen=True)
class ModelConfig(RoPEModelConfigV2):
    ffn_type: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ffn_type not in SUPPORTED_FFN_TYPES:
            raise ValueError(f"Unsupported FFN type: {self.ffn_type}")


@dataclass(frozen=True)
class DiscriminatorConfig:
    name: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    dropout: float
    rope_dim: int
    rope_theta: float
    ffn_type: str
    max_seq_len: int
    invalid_init_std: float
    cls_init_std: float

    def __post_init__(self) -> None:
        if self.name != "trajectory_transformer" or self.ffn_type not in SUPPORTED_FFN_TYPES:
            raise ValueError("Require trajectory_transformer with a supported FFN type")
        if min(self.hidden_dim, self.num_heads, self.num_layers, self.ffn_dim, self.max_seq_len) < 1 or self.hidden_dim % self.num_heads:
            raise ValueError("Invalid discriminator dimensions")
        if self.rope_dim < 2 or self.rope_dim % 2 or self.rope_dim > self.hidden_dim // self.num_heads or self.rope_theta <= 0:
            raise ValueError("Invalid discriminator RoPE configuration")
        if not 0 <= self.dropout < 1 or min(self.invalid_init_std, self.cls_init_std) < 0:
            raise ValueError("Invalid discriminator dropout or initialization")


@dataclass(frozen=True)
class GANConfig:
    enabled: bool
    target_weight: float
    start_step: int
    warmup_steps: int

    def __post_init__(self) -> None:
        if self.target_weight < 0 or self.start_step < 0 or self.warmup_steps < 1:
            raise ValueError("Invalid GAN target or transition schedule")
        if self.enabled and self.target_weight == 0:
            raise ValueError("Enabled GAN requires positive target_weight")


def parse_gan(raw: dict[str, Any]) -> tuple[GANConfig, DiscriminatorConfig]:
    if set(raw) != {"enabled", "target_weight", "transition", "warmup_steps", "discriminator"} or set(raw["transition"]) != {"start_step"}:
        raise ValueError("Require explicit GAN transition and discriminator configuration")
    return parse_section(GANConfig, {key: raw[key] for key in ("enabled", "target_weight", "warmup_steps")} | raw["transition"]), parse_section(DiscriminatorConfig, raw["discriminator"])


@dataclass(frozen=True)
class TrainingConfig:
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    evaluate_every: int
    log_every: int
    cpu_threads: int

    def __post_init__(self) -> None:
        if min(self.steps, self.batch_size, self.evaluate_every, self.log_every, self.cpu_threads) < 1:
            raise ValueError("Training counts must be positive")
        if min(self.learning_rate, self.gradient_clip) <= 0 or self.weight_decay < 0:
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
    train = parse_section(TrainingConfig, {key: value for key, value in raw["training"].items() if key != "gan"})
    gan, discriminator = parse_gan(raw["training"]["gan"])
    parse_section(CorruptionConfig, raw["corruption"])
    if model.architecture == "flow" and gan.enabled:
        raise ValueError("GAN is only used by the direct regression comparison")
    if gan.enabled and gan.start_step + gan.warmup_steps > train.steps:
        raise ValueError("GAN transition and warmup must reach target within training.steps")
    if discriminator.hidden_dim != model.width or discriminator.max_seq_len != model.window_length:
        raise ValueError("Discriminator width/window must match the generator")
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

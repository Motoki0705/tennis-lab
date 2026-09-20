"""Strict task configuration for the two geometry-residual profiles."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import BaseTrainingConfig, TrainingRuntimeConfig
from src.utils.hydra import register_boundary_validator
from src.utils.paths import PROJECT_ROOT

T = TypeVar("T")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    dropout: float
    rope_base: float


@dataclass(frozen=True)
class DataConfig:
    scene_dir: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    sequence_length: int
    target_fps: float
    min_views: int
    max_views: int
    train_limit: int
    val_limit: int
    test_limit: int
    cache_scenes: int


@dataclass(frozen=True)
class InitializerConfig:
    min_score: float
    refinement_steps: int


@dataclass(frozen=True)
class CorruptionConfig:
    observation_sigma_px: float
    temporal_sigma_px: float
    view_bias_px: float
    confidence_noise: float
    outlier_probability: float
    outlier_sigma_px: float
    dropout_probability: float
    burst_probability: float
    burst_max_frames: int
    time_shift_probability: float
    time_shift_max_frames: int
    camera_rotation_std_deg: float
    camera_center_std_m: float
    camera_focal_log_std: float
    camera_principal_std_px: float
    radial_std: float
    focal_scale_min: float
    focal_scale_max: float
    clean_probability: float
    hard_probability: float


@dataclass(frozen=True)
class LossConfig:
    root_weight: float
    relative_weight: float
    world_weight: float
    reprojection_weight: float
    velocity_weight: float
    bone_weight: float
    huber_delta_m: float


@dataclass(frozen=True)
class ResidualConfig:
    task: str
    model: ModelConfig
    data: DataConfig
    initializer: InitializerConfig
    corruption: CorruptionConfig
    loss: LossConfig
    runtime: TrainingRuntimeConfig

    @property
    def joints(self) -> int:
        return 17 if self.task == "plcs" else 1

    @property
    def root_indices(self) -> tuple[int, ...]:
        return (11, 12) if self.task == "plcs" else (0,)


def _section(cls: type[T], value: Any) -> T:
    import math
    from typing import get_type_hints

    raw = dict(value)
    expected = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    if set(raw) != expected:
        raise ValueError(
            f"{cls.__name__}: missing={expected - set(raw)}, unknown={set(raw) - expected}"
        )
    hints = get_type_hints(cls)
    for key, val in raw.items():
        hint = hints[key]
        if hint is float:
            if type(val) not in (int, float) or not math.isfinite(val):
                raise ValueError(f"{cls.__name__}.{key} must be finite numeric")
            raw[key] = float(val)
        elif type(val) is not hint:
            raise ValueError(f"{cls.__name__}.{key} must have type {hint}")
    return cls(**raw)


def validate_config(
    config: DictConfig, expected_task: str | None = None
) -> ResidualConfig:
    root = OmegaConf.to_container(config, resolve=True)
    if not isinstance(root, dict):
        raise ValueError("Residual config must be a mapping")
    keys = {
        "task",
        "model",
        "data",
        "initializer",
        "corruption",
        "loss",
        "training",
        "run",
        "paths",
        "court_keypoints",
    }
    if set(root) != keys:
        raise ValueError(f"Residual config keys differ: {set(root) ^ keys}")
    task = root["task"]
    if task not in ("plcs", "blcs") or (
        expected_task is not None and task != expected_task
    ):
        raise ValueError("Task/entrypoint mismatch")
    runtime = TrainingRuntimeConfig.from_config(config, repository_root=PROJECT_ROOT)
    BaseTrainingConfig.from_mapping(root["training"])
    result = ResidualConfig(
        str(task),
        _section(ModelConfig, root["model"]),
        _section(DataConfig, root["data"]),
        _section(InitializerConfig, root["initializer"]),
        _section(CorruptionConfig, root["corruption"]),
        _section(LossConfig, root["loss"]),
        runtime,
    )
    model, data, noise = result.model, result.data, result.corruption
    if model.name != f"{task}_triangulation_residual_v1":
        raise ValueError("This profile cannot load legacy position/yaw models")
    if (
        model.hidden_dim <= 0
        or model.num_heads <= 0
        or model.hidden_dim % model.num_heads
        or (model.hidden_dim // model.num_heads) % 2
    ):
        raise ValueError("Even head dimension and divisible hidden_dim required")
    if (
        model.num_layers < 1
        or model.ffn_dim < model.hidden_dim
        or not 0 <= model.dropout < 1
        or model.rope_base <= 0
    ):
        raise ValueError("Invalid residual architecture settings")
    if (
        data.batch_size < 1
        or data.num_workers < 0
        or data.sequence_length < 4
        or data.target_fps <= 0
    ):
        raise ValueError("Invalid loader or time sampling settings")
    if not 2 <= data.min_views <= data.max_views or data.cache_scenes < 0:
        raise ValueError("Residual triangulation needs 2 or more cameras")
    if min(data.train_limit, data.val_limit, data.test_limit) < 0:
        raise ValueError("Scene limits must be nonnegative; 0 means complete split")
    if (
        not 0 <= result.initializer.min_score <= 1
        or result.initializer.refinement_steps < 0
    ):
        raise ValueError("Invalid triangulation settings")
    for field in fields(noise):
        value = getattr(noise, field.name)
        if value < 0 or (field.name.endswith("probability") and value > 1):
            raise ValueError(f"Invalid corruption setting {field.name}")
    if (
        noise.clean_probability + noise.hard_probability > 1
        or not 0 < noise.focal_scale_min <= noise.focal_scale_max
    ):
        raise ValueError("Invalid corruption mixture/focal range")
    if (
        any(getattr(result.loss, f.name) < 0 for f in fields(result.loss))
        or result.loss.huber_delta_m <= 0
    ):
        raise ValueError("Loss weights must be nonnegative and Huber delta positive")
    if result.loss.root_weight == 0 or (
        task == "plcs" and result.loss.relative_weight == 0
    ):
        raise ValueError("Both requested residual heads need direct supervision")
    if task == "blcs" and (
        result.loss.relative_weight != 0 or result.loss.bone_weight != 0
    ):
        raise ValueError("BLCS has no relative pose or bones")
    if runtime.training.gan.enabled or runtime.training.qualitative_logging.enabled:
        raise ValueError(
            "GAN and legacy qualitative rendering are not part of this profile"
        )
    if runtime.training.checkpoint.monitor != "val/world_mpjpe_m":
        raise ValueError("Select checkpoints using held-out world 3D error")
    if dict(config.court_keypoints) != {"selector": "physical_v1"}:
        raise ValueError("Residual inputs/outputs use the physical court contract")
    return result


def _validate_plcs(config: DictConfig) -> None:
    validate_config(config, expected_task="plcs")


def _validate_blcs(config: DictConfig) -> None:
    validate_config(config, expected_task="blcs")


register_boundary_validator("plcs.triangulation_residual.train", _validate_plcs)
register_boundary_validator("blcs.triangulation_residual.train", _validate_blcs)

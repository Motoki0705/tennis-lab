"""Strict typed configuration and role-based paths for every SLCS boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.slcs.data.dataset import SLCSDataConfig
from src.tasks.slcs.data.dino_tokens import DinoTokenSpec
from src.tasks.slcs.data.quality import QualityConfig
from src.tasks.slcs.training.losses import SLCSLossConfig
from src.tennis_scene.generate_dataset.manifest import (
    DatasetManifestError,
    split_clip_id,
    validate_id_component,
)
from src.utils.configuration import (
    ConfigField,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    SemanticConfigurationError,
    StrictConfigSchema,
)
from src.utils.hydra import register_boundary_validator
from src.utils.models.components.ffn_layers import (
    SUPPORTED_FFN_TYPES,
    FFNType,
)
from src.utils.paths import PROJECT_ROOT

Number = float | int


def _schema(name: str, fields: dict[str, ConfigField]) -> StrictConfigSchema:
    return StrictConfigSchema(name=name, fields=fields)


def _mapping(schema: StrictConfigSchema) -> ConfigField:
    return ConfigField.mapping(schema)


def _number() -> ConfigField:
    return ConfigField.of(float, int)


SLCS_PATHS_SCHEMA = _schema(
    "paths",
    {
        "project_root": ConfigField.of(str),
        "data_root": ConfigField.of(str),
        "checkpoint_root": ConfigField.of(str),
        "artifact_root": ConfigField.of(str),
        "output_root": ConfigField.of(str),
        "cache_root": ConfigField.of(str),
        "external_asset_root": ConfigField.of(str),
    },
)
SLCS_DINO_SCHEMA = _schema(
    "data.dino",
    {
        "backbone": ConfigField.of(str),
        "patch_size": ConfigField.of(int),
        "image_height": ConfigField.of(int),
        "image_width": ConfigField.of(int),
        "embed_dim": ConfigField.of(int),
        "frame_stride": ConfigField.of(int),
    },
)
SLCS_QUALITY_SCHEMA = _schema(
    "data.quality",
    {
        "min_player_confidence": _number(),
        "min_ball_cameras": ConfigField.of(int),
        "label_weight_power": _number(),
        "min_window_label_ratio": _number(),
    },
)
SLCS_DATA_SCHEMA = _schema(
    "data",
    {
        "dataset_root": ConfigField.of(str),
        "split_file": ConfigField.of(str),
        "batch_size": ConfigField.of(int),
        "num_workers": ConfigField.of(int),
        "pin_memory": ConfigField.of(bool),
        "overfit": ConfigField.of(bool),
        "window_size": ConfigField.of(int),
        "train_stride": ConfigField.of(int),
        "eval_stride": ConfigField.of(int),
        "num_players": ConfigField.of(int),
        "num_court_kp": ConfigField.of(int),
        "require_dino": ConfigField.of(bool),
        "cache_dino_tokens": ConfigField.of(bool),
        "on_incomplete": ConfigField.of(str),
        "dino": _mapping(SLCS_DINO_SCHEMA),
        "quality": _mapping(SLCS_QUALITY_SCHEMA),
    },
)
SLCS_MODEL_SCHEMA = _schema(
    "model",
    {
        "name": ConfigField.of(str),
        "hidden_dim": ConfigField.of(int),
        "num_shared_layers": ConfigField.of(int),
        "num_position_layers": ConfigField.of(int),
        "num_rotation_layers": ConfigField.of(int),
        "num_heads": ConfigField.of(int),
        "ffn_dim": ConfigField.of(int),
        "dropout": _number(),
        "rope_dim": ConfigField.of(int),
        "rope_theta_time": _number(),
        "rope_theta_entity": _number(),
        "attention_type": ConfigField.of(str),
        "ffn_type": ConfigField.of(str),
        "invisible_init_std": _number(),
        "dino_patch_downsample_factor": ConfigField.of(int),
        "dino_cross_attn_every": ConfigField.of(int),
        "log_b_min": _number(),
        "log_b_max": _number(),
    },
)
_LOSS_FIELDS = {
    "player_position_weight": _number(),
    "player_rotation_weight": _number(),
    "player_angle_weight": _number(),
    "ball_position_weight": _number(),
    "player_position_nll_weight": _number(),
    "player_rotation_nll_weight": _number(),
    "ball_position_nll_weight": _number(),
    "player_position_smoothness_weight": _number(),
    "ball_position_smoothness_weight": _number(),
    "ground_penetration_weight": _number(),
    "smoothness_order": ConfigField.of(int),
}
SLCS_LOSS_SCHEMA = _schema("loss", _LOSS_FIELDS)
SLCS_RUN_SCHEMA = _schema(
    "run",
    {
        "output_dir": ConfigField.of(str),
        "seed": ConfigField.of(int),
        "gpus": ConfigField.of(int),
        "resume": ConfigField.of(str, type(None)),
        "init_weights": ConfigField.of(str, type(None)),
        "fast_dev_run": ConfigField.of(bool),
        "dry_run": ConfigField.of(bool),
        "test_after_fit": ConfigField.of(bool),
    },
)
SLCS_TRAINER_SCHEMA = _schema(
    "training.trainer",
    {
        "max_epochs": ConfigField.of(int),
        "gradient_clip_val": ConfigField.of(float, int, type(None)),
        "deterministic": ConfigField.of(bool, str),
        "precision": ConfigField.of(str),
        "log_every_n_steps": ConfigField.of(int),
        "check_val_every_n_epoch": ConfigField.of(int),
        "accumulate_grad_batches": ConfigField.of(int),
        "reload_dataloaders_every_n_epochs": ConfigField.of(int),
        "enable_progress_bar": ConfigField.of(bool),
        "enable_model_summary": ConfigField.of(bool),
        "benchmark": ConfigField.of(bool),
    },
)
SLCS_TRAINING_SCHEMA = _schema(
    "training",
    {
        "trainer": _mapping(SLCS_TRAINER_SCHEMA),
        "learning_rate": _number(),
        "weight_decay": _number(),
        "warmup_steps": ConfigField.of(int, type(None)),
        "warmup_epochs": ConfigField.of(int, type(None)),
        "min_lr": _number(),
        "steps_per_epoch": ConfigField.of(int, type(None)),
        "optimizer": _mapping(
            _schema("training.optimizer", {"betas": ConfigField.sequence(_number())})
        ),
        "compile": _mapping(
            _schema(
                "training.compile",
                {
                    "enabled": ConfigField.of(bool),
                    "backend": ConfigField.of(str),
                    "mode": ConfigField.of(str),
                    "fullgraph": ConfigField.of(bool),
                    "dynamic": ConfigField.of(bool),
                },
            )
        ),
        "checkpoint": _mapping(
            _schema(
                "training.checkpoint",
                {
                    "enabled": ConfigField.of(bool),
                    "filename": ConfigField.of(str),
                    "monitor": ConfigField.of(str),
                    "mode": ConfigField.of(str),
                    "save_top_k": ConfigField.of(int),
                    "save_last": ConfigField.of(bool),
                },
            )
        ),
        "early_stopping": _mapping(
            _schema(
                "training.early_stopping",
                {
                    "enabled": ConfigField.of(bool),
                    "monitor": ConfigField.of(str),
                    "mode": ConfigField.of(str),
                    "patience": ConfigField.of(int),
                    "min_delta": _number(),
                    "check_on_train_epoch_end": ConfigField.of(bool),
                },
            )
        ),
        "lr_monitor": _mapping(
            _schema(
                "training.lr_monitor",
                {
                    "enabled": ConfigField.of(bool),
                    "interval": ConfigField.of(str),
                },
            )
        ),
        "qualitative_logging": _mapping(
            _schema(
                "training.qualitative_logging",
                {
                    "enabled": ConfigField.of(bool),
                    "every_n_epochs": ConfigField.of(int),
                    "num_samples": ConfigField.of(int),
                    "selection_mode": ConfigField.of(str),
                    "selected_indices": ConfigField.of(list, tuple, type(None)),
                },
            )
        ),
        "gan": _mapping(
            _schema(
                "training.gan",
                {
                    "enabled": ConfigField.of(bool),
                    "target_weight": _number(),
                    "warmup_epochs": ConfigField.of(int),
                    "generator_gradient_clip_val": ConfigField.of(
                        float, int, type(None)
                    ),
                    "discriminator_gradient_clip_val": ConfigField.of(
                        float, int, type(None)
                    ),
                    "transition": _mapping(
                        _schema(
                            "training.gan.transition",
                            {"start_epoch": ConfigField.of(int)},
                        )
                    ),
                },
            )
        ),
        "matmul_precision": ConfigField.of(str),
        "allow_tf32": ConfigField.of(bool),
    },
)

SLCS_EVALUATION_SCHEMA = _schema(
    "evaluate",
    {
        "checkpoint": ConfigField.of(str),
        "split": ConfigField.of(str),
        "device": ConfigField.of(str),
        "batch_size": ConfigField.of(int),
        "checkpoint_strict": ConfigField.of(bool),
        "checkpoint_weights_only": ConfigField.of(bool),
        "output_dir": ConfigField.of(str),
    },
)
SLCS_PREDICTION_SCHEMA = _schema(
    "predict",
    {
        "checkpoint": ConfigField.of(str),
        "clip_id": ConfigField.of(str),
        "camera_id": ConfigField.of(str),
        "device": ConfigField.of(str),
        "batch_size": ConfigField.of(int),
        "checkpoint_strict": ConfigField.of(bool),
        "checkpoint_weights_only": ConfigField.of(bool),
        "frame_step": ConfigField.of(int),
        "render_3d": ConfigField.of(bool),
        "render_overlay": ConfigField.of(bool),
        "output_dir": ConfigField.of(str),
    },
)
SLCS_VISUALIZATION_SCHEMA = _schema(
    "visualization",
    {
        "figure_width": _number(),
        "figure_height": _number(),
        "dpi": ConfigField.of(int),
        "court_kp_indices": ConfigField.sequence(ConfigField.of(int)),
        "homography_min_points": ConfigField.of(int),
        "court_visibility_threshold": _number(),
    },
)
SLCS_PRECOMPUTE_SCHEMA = _schema(
    "precompute",
    {
        "device": ConfigField.of(str),
        "batch_size": ConfigField.of(int),
        "overwrite": ConfigField.of(bool),
        "strict": ConfigField.of(bool),
        "repository_path": ConfigField.of(str),
        "checkpoint_path": ConfigField.of(str),
    },
)
SLCS_SPLITS_SCHEMA = _schema(
    "splits",
    {
        "val_ratio": _number(),
        "test_ratio": _number(),
        "seed": ConfigField.of(int),
        "overwrite": ConfigField.of(bool),
        "overfit": ConfigField.of(bool),
    },
)
SLCS_ANALYSIS_SCHEMA = _schema(
    "analysis",
    {
        "arrays": ConfigField.of(str),
        "calibration_bins": ConfigField.of(int),
        "output_dir": ConfigField.of(str),
    },
)

SLCS_TRAINING_BOUNDARY_SCHEMA = _schema(
    "slcs.train",
    {
        "paths": _mapping(SLCS_PATHS_SCHEMA),
        "run": _mapping(SLCS_RUN_SCHEMA),
        "training": _mapping(SLCS_TRAINING_SCHEMA),
        "data": _mapping(SLCS_DATA_SCHEMA),
        "model": _mapping(SLCS_MODEL_SCHEMA),
        "loss": _mapping(SLCS_LOSS_SCHEMA),
    },
)
SLCS_EVALUATION_BOUNDARY_SCHEMA = _schema(
    "slcs.evaluate",
    {
        "paths": _mapping(SLCS_PATHS_SCHEMA),
        "data": _mapping(SLCS_DATA_SCHEMA),
        "evaluate": _mapping(SLCS_EVALUATION_SCHEMA),
    },
)
SLCS_PREDICTION_BOUNDARY_SCHEMA = _schema(
    "slcs.predict_clip",
    {
        "paths": _mapping(SLCS_PATHS_SCHEMA),
        "data": _mapping(SLCS_DATA_SCHEMA),
        "predict": _mapping(SLCS_PREDICTION_SCHEMA),
        "visualization": _mapping(SLCS_VISUALIZATION_SCHEMA),
    },
)
SLCS_PRECOMPUTE_BOUNDARY_SCHEMA = _schema(
    "slcs.precompute_dino_tokens",
    {
        "paths": _mapping(SLCS_PATHS_SCHEMA),
        "data": _mapping(SLCS_DATA_SCHEMA),
        "precompute": _mapping(SLCS_PRECOMPUTE_SCHEMA),
    },
)
SLCS_SPLITS_BOUNDARY_SCHEMA = _schema(
    "slcs.make_splits",
    {
        "paths": _mapping(SLCS_PATHS_SCHEMA),
        "data": _mapping(SLCS_DATA_SCHEMA),
        "splits": _mapping(SLCS_SPLITS_SCHEMA),
    },
)
SLCS_ANALYSIS_BOUNDARY_SCHEMA = _schema(
    "slcs.analyze_predictions",
    {
        "paths": _mapping(SLCS_PATHS_SCHEMA),
        "analysis": _mapping(SLCS_ANALYSIS_SCHEMA),
    },
)


def _container(config: DictConfig) -> dict[str, object]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("SLCS configuration must be a mapping.")
    return cast(dict[str, object], value)


def _validate_boundary(
    config: DictConfig, schema: StrictConfigSchema
) -> dict[str, object]:
    return dict(schema.validate(_container(config)))


def _finite_number(value: object, *, path: str) -> float:
    number = float(cast(Number, value))
    if not math.isfinite(number):
        raise SemanticConfigurationError(f"{path} must be finite.")
    return number


def _nonempty_string(value: object, *, path: str) -> str:
    text = cast(str, value)
    if not text or text != text.strip():
        raise SemanticConfigurationError(f"{path} must be non-empty and trimmed.")
    return text


def _device_string(value: object, *, path: str) -> str:
    device = _nonempty_string(value, path=path)
    try:
        torch.device(device)
    except RuntimeError as error:
        raise SemanticConfigurationError(
            f"{path} is not a valid torch device: {device!r}."
        ) from error
    return device


def _canonical_clip_id(value: object) -> str:
    clip_id = _nonempty_string(value, path="predict.clip_id")
    try:
        split_clip_id(clip_id)
    except DatasetManifestError as error:
        raise SemanticConfigurationError(str(error)) from error
    return clip_id


def _canonical_camera_id(value: object) -> str:
    camera_id = _nonempty_string(value, path="predict.camera_id")
    try:
        validate_id_component(camera_id, field_name="camera_id")
    except DatasetManifestError as error:
        raise SemanticConfigurationError(str(error)) from error
    return camera_id


def _resolver(raw: dict[str, object]) -> PathResolver:
    paths = cast(dict[str, object], raw["paths"])
    empty = [key for key, value in paths.items() if not cast(str, value)]
    if empty:
        raise SemanticConfigurationError(
            f"Runtime path roots must be non-empty; empty: {sorted(empty)}."
        )
    return PathResolver(
        RuntimePathRoots.from_mapping(paths, repository_root=PROJECT_ROOT)
    )


def _resolve_nonempty(
    resolver: PathResolver, role: PathRole, value: object, *, path: str
) -> Path:
    relative = cast(str, value)
    if not relative:
        raise SemanticConfigurationError(f"{path} must not be empty.")
    resolved: Path = resolver.resolve(role, relative)
    return resolved


@dataclass(frozen=True, slots=True)
class SLCSDataRuntimeConfig:
    dataset_root: Path
    split_file: Path
    batch_size: int
    num_workers: int
    pin_memory: bool
    overfit: bool
    pipeline: SLCSDataConfig

    @classmethod
    def from_mapping(
        cls, raw: dict[str, object], resolver: PathResolver
    ) -> SLCSDataRuntimeConfig:
        dino = cast(dict[str, object], raw["dino"])
        quality = cast(dict[str, object], raw["quality"])
        try:
            pipeline = SLCSDataConfig(
                window_size=cast(int, raw["window_size"]),
                train_stride=cast(int, raw["train_stride"]),
                eval_stride=cast(int, raw["eval_stride"]),
                num_players=cast(int, raw["num_players"]),
                num_court_kp=cast(int, raw["num_court_kp"]),
                require_dino=cast(bool, raw["require_dino"]),
                cache_dino_tokens=cast(bool, raw["cache_dino_tokens"]),
                on_incomplete=cast(Literal["error", "skip"], raw["on_incomplete"]),
                dino_spec=DinoTokenSpec(
                    backbone=_nonempty_string(
                        dino["backbone"], path="data.dino.backbone"
                    ),
                    patch_size=cast(int, dino["patch_size"]),
                    image_height=cast(int, dino["image_height"]),
                    image_width=cast(int, dino["image_width"]),
                    embed_dim=cast(int, dino["embed_dim"]),
                    frame_stride=cast(int, dino["frame_stride"]),
                ),
                quality=QualityConfig(
                    min_player_confidence=_finite_number(
                        quality["min_player_confidence"],
                        path="data.quality.min_player_confidence",
                    ),
                    min_ball_cameras=cast(int, quality["min_ball_cameras"]),
                    label_weight_power=_finite_number(
                        quality["label_weight_power"],
                        path="data.quality.label_weight_power",
                    ),
                    min_window_label_ratio=_finite_number(
                        quality["min_window_label_ratio"],
                        path="data.quality.min_window_label_ratio",
                    ),
                ),
            )
        except ValueError as error:
            if isinstance(error, SemanticConfigurationError):
                raise
            raise SemanticConfigurationError(
                f"Invalid SLCS data configuration: {error}"
            ) from error
        batch_size = cast(int, raw["batch_size"])
        num_workers = cast(int, raw["num_workers"])
        if batch_size <= 0 or num_workers < 0:
            raise SemanticConfigurationError(
                "data.batch_size must be positive and data.num_workers non-negative."
            )
        return cls(
            dataset_root=_resolve_nonempty(
                resolver, PathRole.DATA, raw["dataset_root"], path="data.dataset_root"
            ),
            split_file=_resolve_nonempty(
                resolver, PathRole.DATA, raw["split_file"], path="data.split_file"
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=cast(bool, raw["pin_memory"]),
            overfit=cast(bool, raw["overfit"]),
            pipeline=pipeline,
        )


@dataclass(frozen=True, slots=True)
class SLCSModelConfig:
    name: str
    hidden_dim: int
    num_shared_layers: int
    num_position_layers: int
    num_rotation_layers: int
    num_heads: int
    ffn_dim: int
    dropout: float
    rope_dim: int
    rope_theta_time: float
    rope_theta_entity: float
    attention_type: Literal["mha"]
    ffn_type: FFNType
    invisible_init_std: float
    dino_patch_downsample_factor: int
    dino_cross_attn_every: int
    log_b_min: float
    log_b_max: float

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> SLCSModelConfig:
        name = cast(str, raw["name"])
        attention_type = cast(str, raw["attention_type"])
        ffn_type = cast(str, raw["ffn_type"])
        if name != "slcs_fusion":
            raise SemanticConfigurationError(
                f"model.name must be 'slcs_fusion'; got {name!r}."
            )
        if ffn_type not in SUPPORTED_FFN_TYPES:
            raise SemanticConfigurationError(
                "model.ffn_type must be one of "
                f"{sorted(SUPPORTED_FFN_TYPES)!r}; got {ffn_type!r}."
            )
        if attention_type != "mha":
            raise SemanticConfigurationError(
                "model.attention_type must be 'mha' for the canonical SLCS "
                f"architecture; got {attention_type!r}."
            )
        result = cls(
            name=name,
            hidden_dim=cast(int, raw["hidden_dim"]),
            num_shared_layers=cast(int, raw["num_shared_layers"]),
            num_position_layers=cast(int, raw["num_position_layers"]),
            num_rotation_layers=cast(int, raw["num_rotation_layers"]),
            num_heads=cast(int, raw["num_heads"]),
            ffn_dim=cast(int, raw["ffn_dim"]),
            dropout=_finite_number(raw["dropout"], path="model.dropout"),
            rope_dim=cast(int, raw["rope_dim"]),
            rope_theta_time=_finite_number(
                raw["rope_theta_time"], path="model.rope_theta_time"
            ),
            rope_theta_entity=_finite_number(
                raw["rope_theta_entity"], path="model.rope_theta_entity"
            ),
            attention_type=cast(Literal["mha"], attention_type),
            ffn_type=cast(FFNType, ffn_type),
            invisible_init_std=_finite_number(
                raw["invisible_init_std"], path="model.invisible_init_std"
            ),
            dino_patch_downsample_factor=cast(int, raw["dino_patch_downsample_factor"]),
            dino_cross_attn_every=cast(int, raw["dino_cross_attn_every"]),
            log_b_min=_finite_number(raw["log_b_min"], path="model.log_b_min"),
            log_b_max=_finite_number(raw["log_b_max"], path="model.log_b_max"),
        )
        if result.hidden_dim <= 0 or result.num_heads <= 0:
            raise SemanticConfigurationError(
                "model.hidden_dim and model.num_heads must be positive."
            )
        if result.hidden_dim % result.num_heads:
            raise SemanticConfigurationError(
                "model.hidden_dim must be divisible by model.num_heads."
            )
        depths = (
            result.num_shared_layers,
            result.num_position_layers,
            result.num_rotation_layers,
        )
        if any(depth < 0 for depth in depths):
            raise SemanticConfigurationError("model layer counts must be non-negative.")
        if result.num_shared_layers + result.num_position_layers <= 0 or (
            result.num_shared_layers + result.num_rotation_layers <= 0
        ):
            raise SemanticConfigurationError(
                "model position and rotation paths must each contain a layer."
            )
        if not 0.0 <= result.dropout <= 1.0:
            raise SemanticConfigurationError("model.dropout must be in [0, 1].")
        if result.ffn_dim <= 0:
            raise SemanticConfigurationError("model.ffn_dim must be positive.")
        if result.invisible_init_std <= 0.0:
            raise SemanticConfigurationError(
                "model.invisible_init_std must be positive."
            )
        head_dim = result.hidden_dim // result.num_heads
        if result.rope_dim <= 0 or result.rope_dim > head_dim or result.rope_dim % 4:
            raise SemanticConfigurationError(
                "model.rope_dim must be positive, <= head_dim, and divisible by 4."
            )
        if result.rope_theta_time <= 0.0 or result.rope_theta_entity <= 0.0:
            raise SemanticConfigurationError(
                "model.rope_theta_time and model.rope_theta_entity must be positive."
            )
        if result.dino_patch_downsample_factor <= 0:
            raise SemanticConfigurationError(
                "model.dino_patch_downsample_factor must be positive."
            )
        if result.dino_cross_attn_every <= 0:
            raise SemanticConfigurationError(
                "model.dino_cross_attn_every must be positive."
            )
        if result.log_b_min >= result.log_b_max:
            raise SemanticConfigurationError(
                "model.log_b_min must be smaller than model.log_b_max."
            )
        return result


def _loss(raw: dict[str, object]) -> SLCSLossConfig:
    result = SLCSLossConfig(
        player_position_weight=_finite_number(
            raw["player_position_weight"], path="loss.player_position_weight"
        ),
        player_rotation_weight=_finite_number(
            raw["player_rotation_weight"], path="loss.player_rotation_weight"
        ),
        player_angle_weight=_finite_number(
            raw["player_angle_weight"], path="loss.player_angle_weight"
        ),
        ball_position_weight=_finite_number(
            raw["ball_position_weight"], path="loss.ball_position_weight"
        ),
        player_position_nll_weight=_finite_number(
            raw["player_position_nll_weight"],
            path="loss.player_position_nll_weight",
        ),
        player_rotation_nll_weight=_finite_number(
            raw["player_rotation_nll_weight"],
            path="loss.player_rotation_nll_weight",
        ),
        ball_position_nll_weight=_finite_number(
            raw["ball_position_nll_weight"], path="loss.ball_position_nll_weight"
        ),
        player_position_smoothness_weight=_finite_number(
            raw["player_position_smoothness_weight"],
            path="loss.player_position_smoothness_weight",
        ),
        ball_position_smoothness_weight=_finite_number(
            raw["ball_position_smoothness_weight"],
            path="loss.ball_position_smoothness_weight",
        ),
        ground_penetration_weight=_finite_number(
            raw["ground_penetration_weight"],
            path="loss.ground_penetration_weight",
        ),
        smoothness_order=cast(int, raw["smoothness_order"]),
    )
    weights = (
        result.player_position_weight,
        result.player_rotation_weight,
        result.player_angle_weight,
        result.ball_position_weight,
        result.player_position_nll_weight,
        result.player_rotation_nll_weight,
        result.ball_position_nll_weight,
        result.player_position_smoothness_weight,
        result.ball_position_smoothness_weight,
        result.ground_penetration_weight,
    )
    if any(weight < 0 for weight in weights):
        raise SemanticConfigurationError("SLCS loss weights must be non-negative.")
    if not any(weight > 0 for weight in weights):
        raise SemanticConfigurationError(
            "At least one SLCS loss weight must be positive."
        )
    if result.smoothness_order not in {1, 2, 3}:
        raise SemanticConfigurationError("loss.smoothness_order must be 1, 2, or 3.")
    return result


@dataclass(frozen=True, slots=True)
class SLCSTrainingRuntimeConfig(TrainingRuntimeConfig):
    data: SLCSDataRuntimeConfig
    model: SLCSModelConfig
    loss: SLCSLossConfig
    raw: DictConfig

    @classmethod
    def from_config(
        cls,
        value: object,
        *,
        repository_root: Path = PROJECT_ROOT,
    ) -> SLCSTrainingRuntimeConfig:
        if not isinstance(value, DictConfig):
            raise TypeError("SLCS training requires a composed DictConfig.")
        config = value
        raw = _validate_boundary(config, SLCS_TRAINING_BOUNDARY_SCHEMA)
        training_values = cast(dict[str, object], raw["training"])
        if (
            training_values["warmup_steps"] is not None
            and training_values["warmup_epochs"] is not None
        ):
            raise SemanticConfigurationError(
                "training.warmup_steps and training.warmup_epochs are mutually exclusive."
            )
        base = TrainingRuntimeConfig.from_config(
            config, repository_root=repository_root
        )
        data = SLCSDataRuntimeConfig.from_mapping(
            cast(dict[str, object], raw["data"]), base.resolver
        )
        model = SLCSModelConfig.from_mapping(cast(dict[str, object], raw["model"]))
        factor = model.dino_patch_downsample_factor
        dino = data.pipeline.dino_spec
        if dino.grid_h % factor or dino.grid_w % factor:
            raise SemanticConfigurationError(
                "model.dino_patch_downsample_factor must divide both data.dino grid dimensions."
            )
        return cls(
            run=base.run,
            training=base.training,
            resolver=base.resolver,
            data=data,
            model=model,
            loss=_loss(cast(dict[str, object], raw["loss"])),
            raw=config,
        )


@dataclass(frozen=True, slots=True)
class SLCSEvaluationConfig:
    resolver: PathResolver
    data: SLCSDataRuntimeConfig
    checkpoint: Path
    split: Literal["train", "val", "test"]
    device: str
    batch_size: int
    checkpoint_strict: bool
    checkpoint_weights_only: bool
    output_dir: Path

    @classmethod
    def from_config(cls, config: DictConfig) -> SLCSEvaluationConfig:
        raw = _validate_boundary(config, SLCS_EVALUATION_BOUNDARY_SCHEMA)
        resolver = _resolver(raw)
        values = cast(dict[str, object], raw["evaluate"])
        split = cast(str, values["split"])
        if split not in {"train", "val", "test"}:
            raise SemanticConfigurationError(f"evaluate.split is invalid: {split!r}.")
        if cast(int, values["batch_size"]) <= 0:
            raise SemanticConfigurationError("evaluate.batch_size must be positive.")
        device = _device_string(values["device"], path="evaluate.device")
        return cls(
            resolver,
            SLCSDataRuntimeConfig.from_mapping(
                cast(dict[str, object], raw["data"]), resolver
            ),
            _resolve_nonempty(
                resolver,
                PathRole.CHECKPOINT,
                values["checkpoint"],
                path="evaluate.checkpoint",
            ),
            cast(Literal["train", "val", "test"], split),
            device,
            cast(int, values["batch_size"]),
            cast(bool, values["checkpoint_strict"]),
            cast(bool, values["checkpoint_weights_only"]),
            _resolve_nonempty(
                resolver,
                PathRole.OUTPUT,
                values["output_dir"],
                path="evaluate.output_dir",
            ),
        )


@dataclass(frozen=True, slots=True)
class SLCSVisualizationConfig:
    figure_width: float
    figure_height: float
    dpi: int
    court_kp_indices: tuple[int, ...]
    homography_min_points: int
    court_visibility_threshold: float

    @classmethod
    def from_mapping(
        cls, raw: dict[str, object], *, num_court_kp: int
    ) -> SLCSVisualizationConfig:
        indices = cast(tuple[object, ...], raw["court_kp_indices"])
        if any(type(index) is not int for index in indices):
            raise SemanticConfigurationError(
                "visualization.court_kp_indices must contain only integers."
            )
        typed_indices = tuple(cast(int, index) for index in indices)
        result = cls(
            figure_width=_finite_number(
                raw["figure_width"], path="visualization.figure_width"
            ),
            figure_height=_finite_number(
                raw["figure_height"], path="visualization.figure_height"
            ),
            dpi=cast(int, raw["dpi"]),
            court_kp_indices=typed_indices,
            homography_min_points=cast(int, raw["homography_min_points"]),
            court_visibility_threshold=_finite_number(
                raw["court_visibility_threshold"],
                path="visualization.court_visibility_threshold",
            ),
        )
        if result.figure_width <= 0.0 or result.figure_height <= 0.0:
            raise SemanticConfigurationError(
                "visualization figure dimensions must be positive."
            )
        if result.dpi <= 0:
            raise SemanticConfigurationError("visualization.dpi must be positive.")
        if len(result.court_kp_indices) != num_court_kp:
            raise SemanticConfigurationError(
                "visualization.court_kp_indices must contain exactly "
                f"data.num_court_kp={num_court_kp} entries."
            )
        if len(set(result.court_kp_indices)) != len(result.court_kp_indices) or any(
            not 0 <= index < 20 for index in result.court_kp_indices
        ):
            raise SemanticConfigurationError(
                "visualization.court_kp_indices must be unique CourtKP20 indices."
            )
        if not 4 <= result.homography_min_points <= num_court_kp:
            raise SemanticConfigurationError(
                "visualization.homography_min_points must be between 4 and "
                "data.num_court_kp."
            )
        if not 0.0 <= result.court_visibility_threshold <= 1.0:
            raise SemanticConfigurationError(
                "visualization.court_visibility_threshold must be in [0, 1]."
            )
        return result


@dataclass(frozen=True, slots=True)
class SLCSPredictConfig:
    resolver: PathResolver
    data: SLCSDataRuntimeConfig
    checkpoint: Path
    clip_id: str
    camera_id: str
    device: str
    batch_size: int
    checkpoint_strict: bool
    checkpoint_weights_only: bool
    frame_step: int
    render_3d: bool
    render_overlay: bool
    output_dir: Path
    visualization: SLCSVisualizationConfig

    @classmethod
    def from_config(cls, config: DictConfig) -> SLCSPredictConfig:
        raw = _validate_boundary(config, SLCS_PREDICTION_BOUNDARY_SCHEMA)
        resolver = _resolver(raw)
        values = cast(dict[str, object], raw["predict"])
        if cast(int, values["batch_size"]) <= 0 or cast(int, values["frame_step"]) <= 0:
            raise SemanticConfigurationError(
                "predict.batch_size and predict.frame_step must be positive."
            )
        clip_id = _canonical_clip_id(values["clip_id"])
        camera_id = _canonical_camera_id(values["camera_id"])
        device = _device_string(values["device"], path="predict.device")
        data = SLCSDataRuntimeConfig.from_mapping(
            cast(dict[str, object], raw["data"]), resolver
        )
        return cls(
            resolver,
            data,
            _resolve_nonempty(
                resolver,
                PathRole.CHECKPOINT,
                values["checkpoint"],
                path="predict.checkpoint",
            ),
            clip_id,
            camera_id,
            device,
            cast(int, values["batch_size"]),
            cast(bool, values["checkpoint_strict"]),
            cast(bool, values["checkpoint_weights_only"]),
            cast(int, values["frame_step"]),
            cast(bool, values["render_3d"]),
            cast(bool, values["render_overlay"]),
            _resolve_nonempty(
                resolver,
                PathRole.OUTPUT,
                values["output_dir"],
                path="predict.output_dir",
            ),
            SLCSVisualizationConfig.from_mapping(
                cast(dict[str, object], raw["visualization"]),
                num_court_kp=data.pipeline.num_court_kp,
            ),
        )


@dataclass(frozen=True, slots=True)
class SLCSPrecomputeConfig:
    resolver: PathResolver
    data: SLCSDataRuntimeConfig
    device: str
    batch_size: int
    overwrite: bool
    strict: bool
    repository_path: Path
    checkpoint_path: Path

    @classmethod
    def from_config(cls, config: DictConfig) -> SLCSPrecomputeConfig:
        raw = _validate_boundary(config, SLCS_PRECOMPUTE_BOUNDARY_SCHEMA)
        resolver = _resolver(raw)
        values = cast(dict[str, object], raw["precompute"])
        if cast(int, values["batch_size"]) <= 0:
            raise SemanticConfigurationError("precompute.batch_size must be positive.")
        device = _device_string(values["device"], path="precompute.device")
        return cls(
            resolver,
            SLCSDataRuntimeConfig.from_mapping(
                cast(dict[str, object], raw["data"]), resolver
            ),
            device,
            cast(int, values["batch_size"]),
            cast(bool, values["overwrite"]),
            cast(bool, values["strict"]),
            _resolve_nonempty(
                resolver,
                PathRole.EXTERNAL_ASSET,
                values["repository_path"],
                path="precompute.repository_path",
            ),
            _resolve_nonempty(
                resolver,
                PathRole.CHECKPOINT,
                values["checkpoint_path"],
                path="precompute.checkpoint_path",
            ),
        )


@dataclass(frozen=True, slots=True)
class SLCSSplitConfig:
    data: SLCSDataRuntimeConfig
    val_ratio: float
    test_ratio: float
    seed: int
    overwrite: bool
    overfit: bool

    @classmethod
    def from_config(cls, config: DictConfig) -> SLCSSplitConfig:
        raw = _validate_boundary(config, SLCS_SPLITS_BOUNDARY_SCHEMA)
        resolver = _resolver(raw)
        values = cast(dict[str, object], raw["splits"])
        val_ratio, test_ratio = (
            _finite_number(values["val_ratio"], path="splits.val_ratio"),
            _finite_number(values["test_ratio"], path="splits.test_ratio"),
        )
        if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
            raise SemanticConfigurationError(
                "splits ratios must be non-negative and sum to less than one."
            )
        return cls(
            SLCSDataRuntimeConfig.from_mapping(
                cast(dict[str, object], raw["data"]), resolver
            ),
            val_ratio,
            test_ratio,
            cast(int, values["seed"]),
            cast(bool, values["overwrite"]),
            cast(bool, values["overfit"]),
        )


@dataclass(frozen=True, slots=True)
class SLCSAnalysisConfig:
    arrays: Path
    calibration_bins: int
    output_dir: Path

    @classmethod
    def from_config(cls, config: DictConfig) -> SLCSAnalysisConfig:
        raw = _validate_boundary(config, SLCS_ANALYSIS_BOUNDARY_SCHEMA)
        resolver = _resolver(raw)
        values = cast(dict[str, object], raw["analysis"])
        if cast(int, values["calibration_bins"]) <= 0:
            raise SemanticConfigurationError(
                "analysis.calibration_bins must be positive."
            )
        return cls(
            _resolve_nonempty(
                resolver,
                PathRole.OUTPUT,
                values["arrays"],
                path="analysis.arrays",
            ),
            cast(int, values["calibration_bins"]),
            _resolve_nonempty(
                resolver,
                PathRole.OUTPUT,
                values["output_dir"],
                path="analysis.output_dir",
            ),
        )


def validate_training_boundary(config: DictConfig) -> None:
    """Validate the complete SLCS training contract before runner side effects."""
    SLCSTrainingRuntimeConfig.from_config(config)


def validate_evaluation_boundary(config: DictConfig) -> None:
    """Validate the complete SLCS evaluation contract before checkpoint I/O."""
    SLCSEvaluationConfig.from_config(config)


def validate_prediction_boundary(config: DictConfig) -> None:
    """Validate the complete SLCS prediction/visualization contract before I/O."""
    SLCSPredictConfig.from_config(config)


def validate_precompute_boundary(config: DictConfig) -> None:
    """Validate the complete SLCS DINO precompute contract before model I/O."""
    SLCSPrecomputeConfig.from_config(config)


def validate_split_boundary(config: DictConfig) -> None:
    """Validate the complete SLCS split-generation contract before dataset I/O."""
    SLCSSplitConfig.from_config(config)


def validate_analysis_boundary(config: DictConfig) -> None:
    """Validate the complete SLCS analysis contract before artifact I/O."""
    SLCSAnalysisConfig.from_config(config)


register_boundary_validator("slcs.train", validate_training_boundary)
register_boundary_validator("slcs.evaluate", validate_evaluation_boundary)
register_boundary_validator("slcs.predict_clip", validate_prediction_boundary)
register_boundary_validator("slcs.precompute_dino_tokens", validate_precompute_boundary)
register_boundary_validator("slcs.make_splits", validate_split_boundary)
register_boundary_validator("slcs.analyze_predictions", validate_analysis_boundary)


__all__ = [
    "SLCS_ANALYSIS_BOUNDARY_SCHEMA",
    "SLCS_ANALYSIS_SCHEMA",
    "SLCS_DATA_SCHEMA",
    "SLCS_DINO_SCHEMA",
    "SLCS_EVALUATION_BOUNDARY_SCHEMA",
    "SLCS_EVALUATION_SCHEMA",
    "SLCS_LOSS_SCHEMA",
    "SLCS_MODEL_SCHEMA",
    "SLCS_PATHS_SCHEMA",
    "SLCS_PRECOMPUTE_BOUNDARY_SCHEMA",
    "SLCS_PRECOMPUTE_SCHEMA",
    "SLCS_PREDICTION_BOUNDARY_SCHEMA",
    "SLCS_PREDICTION_SCHEMA",
    "SLCS_QUALITY_SCHEMA",
    "SLCS_RUN_SCHEMA",
    "SLCS_SPLITS_BOUNDARY_SCHEMA",
    "SLCS_SPLITS_SCHEMA",
    "SLCS_TRAINER_SCHEMA",
    "SLCS_TRAINING_BOUNDARY_SCHEMA",
    "SLCS_TRAINING_SCHEMA",
    "SLCS_VISUALIZATION_SCHEMA",
    "SLCSAnalysisConfig",
    "SLCSDataRuntimeConfig",
    "SLCSEvaluationConfig",
    "SLCSModelConfig",
    "SLCSPrecomputeConfig",
    "SLCSPredictConfig",
    "SLCSSplitConfig",
    "SLCSTrainingRuntimeConfig",
    "SLCSVisualizationConfig",
    "validate_analysis_boundary",
    "validate_evaluation_boundary",
    "validate_precompute_boundary",
    "validate_prediction_boundary",
    "validate_split_boundary",
    "validate_training_boundary",
]

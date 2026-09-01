"""Typed Hydra boundary contracts for Court Alignment training and evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from omegaconf import DictConfig

from src.tasks.base.configuration import (
    TrainingRuntimeConfig,
    as_config_mapping,
    exact_config_mapping,
    require_config_mapping,
)
from src.utils.configuration import (
    ConfigField,
    ConfigurationTypeError,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    SemanticConfigurationError,
    StrictConfigSchema,
)
from src.utils.hydra import register_boundary_validator
from src.utils.paths import PROJECT_ROOT

ConfigMapping = Mapping[str, object]


def _schema(
    name: str,
    fields: Mapping[str, ConfigField],
    *,
    semantic_checks: Sequence[Callable[[Mapping[str, object]], None]] = (),
) -> StrictConfigSchema:
    return StrictConfigSchema(
        name=name,
        fields=fields,
        semantic_checks=tuple(semantic_checks),
    )


def _number() -> ConfigField:
    return ConfigField.of(float, int)


def _target(expected: str) -> Callable[[Mapping[str, object]], None]:
    def check(value: Mapping[str, object]) -> None:
        if value.get("_target_") != expected:
            raise SemanticConfigurationError(
                f"{value.get('_target_')!r} is not the supported target {expected!r}."
            )

    return check


def _positive(value: object, *, path: str, allow_zero: bool = False) -> None:
    if type(value) not in (int, float):
        raise SemanticConfigurationError(f"{path} must be a finite number.")
    numeric = float(cast(int | float, value))
    if not math.isfinite(numeric):
        raise SemanticConfigurationError(f"{path} must be a finite number.")
    if numeric < 0.0 or (not allow_zero and numeric <= 0.0):
        operator = ">= 0" if allow_zero else "> 0"
        raise SemanticConfigurationError(f"{path} must be {operator}.")


def _data_semantics(value: Mapping[str, object]) -> None:
    if (
        value["_target_"]
        != "src.tasks.court_alignment.data.datamodule.GroundCourtDataModule"
    ):
        raise SemanticConfigurationError(
            "data._target_ must construct GroundCourtDataModule."
        )
    for key in (
        "sigma_px",
        "line_width_px",
        "vote_radius_px",
        "rotation_seam_margin_rad",
    ):
        _positive(value[key], path=f"data.{key}")
    for key in (
        "min_center_distance_px",
        "footprint_overlap_tolerance_px",
        "court_margin_px",
    ):
        _positive(value[key], path=f"data.{key}", allow_zero=True)
    for key in (
        "train_samples",
        "val_samples",
        "test_samples",
        "batch_size",
        "max_sampling_attempts",
    ):
        item = value[key]
        if type(item) is not int or item <= 0:
            raise SemanticConfigurationError(f"data.{key} must be a positive integer.")
    if type(value["num_workers"]) is not int or value["num_workers"] < 0:
        raise SemanticConfigurationError(
            "data.num_workers must be a non-negative integer."
        )
    for key in ("min_courts", "max_courts"):
        item = value[key]
        if type(item) is not int or item <= 0:
            raise SemanticConfigurationError(f"data.{key} must be a positive integer.")
    if cast(int, value["min_courts"]) > cast(int, value["max_courts"]):
        raise SemanticConfigurationError(
            "data.min_courts must not exceed data.max_courts."
        )
    scale_low = float(cast(int | float, value["min_scale_px_per_metre"]))
    scale_high = float(cast(int | float, value["max_scale_px_per_metre"]))
    if (
        not math.isfinite(scale_low)
        or not math.isfinite(scale_high)
        or scale_low <= 0.0
        or scale_high <= 0.0
        or scale_low > scale_high
    ):
        raise SemanticConfigurationError(
            "data.scale_px_per_metre_range must be finite, positive, and ordered."
        )
    rotation = cast(Sequence[float | int], value["rotation_rad_range"])
    if len(rotation) != 2 or any(not math.isfinite(float(item)) for item in rotation):
        raise SemanticConfigurationError(
            "data.rotation_rad_range must contain two finite values."
        )
    margin = float(cast(int | float, value["rotation_seam_margin_rad"]))
    low, high = float(rotation[0]), float(rotation[1])
    if (
        margin >= math.pi / 2.0
        or low < margin
        or high > math.pi - margin
        or high <= low
    ):
        raise SemanticConfigurationError(
            "data.rotation_rad_range must stay inside the configured axial seam margin."
        )
    image_size = value["image_size"]
    if type(image_size) is int:
        if image_size <= 0:
            raise SemanticConfigurationError("data.image_size must be positive.")
    elif isinstance(image_size, (list, tuple)):
        if len(image_size) != 2 or any(
            type(item) is not int or item <= 0 for item in image_size
        ):
            raise SemanticConfigurationError(
                "data.image_size must be a positive integer pair."
            )
    else:
        raise ConfigurationTypeError(
            "data.image_size must be an integer or pair of integers."
        )
    augmentations = cast(Sequence[object], value["augmentations"])
    for index, item in enumerate(augmentations):
        if not isinstance(item, Mapping):
            raise ConfigurationTypeError(
                f"data.augmentations[{index}] must be a mapping."
            )
        if type(item.get("name")) is not str or not cast(str, item["name"]).strip():
            raise SemanticConfigurationError(
                f"data.augmentations[{index}].name must be a non-empty string."
            )
        if type(item.get("params")) is not dict:
            raise ConfigurationTypeError(
                f"data.augmentations[{index}].params must be a mapping."
            )


COURT_ALIGNMENT_PATHS_SCHEMA = _schema(
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

COURT_ALIGNMENT_DATA_SCHEMA = _schema(
    "data",
    {
        "_target_": ConfigField.of(str),
        "image_size": ConfigField.of(int, list, tuple),
        "train_samples": ConfigField.of(int),
        "val_samples": ConfigField.of(int),
        "test_samples": ConfigField.of(int),
        "batch_size": ConfigField.of(int),
        "num_workers": ConfigField.of(int),
        "pin_memory": ConfigField.of(bool),
        "min_courts": ConfigField.of(int),
        "max_courts": ConfigField.of(int),
        "sigma_px": _number(),
        "line_width_px": _number(),
        "vote_radius_px": _number(),
        "min_scale_px_per_metre": _number(),
        "max_scale_px_per_metre": _number(),
        "rotation_seam_margin_rad": _number(),
        "rotation_rad_range": ConfigField.sequence(_number()),
        "min_center_distance_px": _number(),
        "footprint_overlap_tolerance_px": _number(),
        "max_sampling_attempts": ConfigField.of(int),
        "court_margin_px": _number(),
        "seed": ConfigField.of(int),
        "augmentations": ConfigField.of(list, tuple),
    },
    semantic_checks=(_data_semantics,),
)

COURT_ALIGNMENT_MODEL_SCHEMA = _schema(
    "model",
    {
        "_target_": ConfigField.of(str),
        "base_channels": ConfigField.of(int),
        "group_norm_groups": ConfigField.of(int),
        "num_keypoints": ConfigField.of(int),
        "heatmap_prior_probability": _number(),
    },
    semantic_checks=(
        _target("src.tasks.court_alignment.models.cnn.CourtAlignmentCNN"),
    ),
)
COURT_ALIGNMENT_LOSS_SCHEMA = _schema(
    "loss",
    {
        "_target_": ConfigField.of(str),
        "heatmap_weight": _number(),
        "center_vote_weight": _number(),
        "focal_alpha": _number(),
        "focal_beta": _number(),
        "vote_beta": _number(),
    },
    semantic_checks=(
        _target("src.tasks.court_alignment.training.losses.CourtAlignmentLoss"),
    ),
)
COURT_ALIGNMENT_DECODER_SCHEMA = _schema(
    "decoder",
    {
        "threshold": _number(),
        "nms_kernel": ConfigField.of(int),
        "max_peaks": ConfigField.of(int),
        "subpixel_refine": ConfigField.of(bool),
        "cluster_distance_px": _number(),
        "max_instances": ConfigField.of(int),
    },
)
COURT_ALIGNMENT_METRICS_SCHEMA = _schema(
    "metrics",
    {
        "_target_": ConfigField.of(str),
        "threshold": _number(),
        "nms_kernel": ConfigField.of(int),
        "max_peaks": ConfigField.of(int),
        "match_max_error_px": _number(),
        "minimum_common_keypoints": ConfigField.of(int),
        "minimum_visible_keypoints": ConfigField.of(int),
        "minimum_visible_fraction": _number(),
        "minimum_sim2_keypoints": ConfigField.of(int),
    },
    semantic_checks=(
        _target("src.tasks.court_alignment.training.metrics.CourtAlignmentMetrics"),
    ),
)

_TRAINER_SCHEMA = _schema(
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
_CHECKPOINT_SCHEMA = _schema(
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
_EARLY_STOPPING_SCHEMA = _schema(
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
_QUALITATIVE_SCHEMA = _schema(
    "training.qualitative_logging",
    {
        "enabled": ConfigField.of(bool),
        "every_n_epochs": ConfigField.of(int),
        "num_samples": ConfigField.of(int),
        "selection_mode": ConfigField.of(str),
        "selected_indices": ConfigField.of(list, tuple, type(None)),
    },
)
_GAN_SCHEMA = _schema(
    "training.gan",
    {
        "enabled": ConfigField.of(bool),
        "target_weight": _number(),
        "warmup_epochs": ConfigField.of(int),
        "generator_gradient_clip_val": ConfigField.of(float, int, type(None)),
        "discriminator_gradient_clip_val": ConfigField.of(float, int, type(None)),
        "transition": ConfigField.mapping(
            _schema("training.gan.transition", {"start_epoch": ConfigField.of(int)})
        ),
    },
)
COURT_ALIGNMENT_TRAINING_SCHEMA = _schema(
    "training",
    {
        "trainer": ConfigField.mapping(_TRAINER_SCHEMA),
        "learning_rate": _number(),
        "weight_decay": _number(),
        "warmup_epochs": ConfigField.of(int, type(None)),
        "warmup_steps": ConfigField.of(int, type(None)),
        "min_lr": _number(),
        "steps_per_epoch": ConfigField.of(int, type(None)),
        "optimizer": ConfigField.mapping(
            _schema("training.optimizer", {"betas": ConfigField.sequence(_number())})
        ),
        "compile": ConfigField.mapping(
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
        "matmul_precision": ConfigField.of(str),
        "allow_tf32": ConfigField.of(bool),
        "checkpoint": ConfigField.mapping(_CHECKPOINT_SCHEMA),
        "early_stopping": ConfigField.mapping(_EARLY_STOPPING_SCHEMA),
        "lr_monitor": ConfigField.mapping(
            _schema(
                "training.lr_monitor",
                {"enabled": ConfigField.of(bool), "interval": ConfigField.of(str)},
            )
        ),
        "qualitative_logging": ConfigField.mapping(_QUALITATIVE_SCHEMA),
        "gan": ConfigField.mapping(_GAN_SCHEMA),
    },
)

COURT_ALIGNMENT_RUN_SCHEMA = _schema(
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
COURT_ALIGNMENT_EVALUATION_SCHEMA = _schema(
    "evaluation", {"checkpoint_path": ConfigField.of(str, type(None))}
)


@dataclass(frozen=True, slots=True)
class CourtAlignmentRuntimeConfig:
    """Fully composed and validated task boundary configuration."""

    runtime: TrainingRuntimeConfig
    sections: Mapping[str, Mapping[str, object]]
    evaluation_checkpoint: Path | None

    @classmethod
    def from_config(
        cls, config: DictConfig | Mapping[str, object], *, evaluation: bool = False
    ) -> CourtAlignmentRuntimeConfig:
        resolved = as_config_mapping(config, path="configuration")
        required = {
            "paths",
            "data",
            "model",
            "loss",
            "decoder",
            "metrics",
            "training",
            "run",
        }
        optional = {"evaluation"} if evaluation else set()
        root = exact_config_mapping(
            resolved,
            path="configuration",
            required_keys=required,
            optional_keys=optional,
        )
        runtime = TrainingRuntimeConfig.from_config(root, repository_root=PROJECT_ROOT)
        COURT_ALIGNMENT_PATHS_SCHEMA.validate(
            require_config_mapping(root, "paths", path="configuration"),
            path="paths",
        )
        sections: dict[str, Mapping[str, object]] = {}
        for name, schema in (
            ("data", COURT_ALIGNMENT_DATA_SCHEMA),
            ("model", COURT_ALIGNMENT_MODEL_SCHEMA),
            ("loss", COURT_ALIGNMENT_LOSS_SCHEMA),
            ("decoder", COURT_ALIGNMENT_DECODER_SCHEMA),
            ("metrics", COURT_ALIGNMENT_METRICS_SCHEMA),
            ("training", COURT_ALIGNMENT_TRAINING_SCHEMA),
            ("run", COURT_ALIGNMENT_RUN_SCHEMA),
        ):
            sections[name] = schema.validate(
                require_config_mapping(root, name, path="configuration"), path=name
            )
        data = sections["data"]
        if cast(int, data["seed"]) != cast(int, sections["run"]["seed"]):
            raise SemanticConfigurationError(
                "data.seed must equal run.seed for reproducible splits."
            )
        if cast(int, data["max_courts"]) > cast(
            int, sections["decoder"]["max_instances"]
        ):
            raise SemanticConfigurationError(
                "decoder.max_instances must cover data.max_courts."
            )
        if cast(int, data["max_courts"]) > cast(int, sections["decoder"]["max_peaks"]):
            raise SemanticConfigurationError(
                "decoder.max_peaks must cover data.max_courts."
            )
        if cast(int, sections["model"]["num_keypoints"]) != 14:
            raise SemanticConfigurationError(
                "model.num_keypoints must preserve the KP14 contract."
            )
        expected_steps = math.ceil(
            cast(int, data["train_samples"]) / cast(int, data["batch_size"])
        )
        if sections["training"]["steps_per_epoch"] != expected_steps:
            raise SemanticConfigurationError(
                "training.steps_per_epoch must equal ceil(data.train_samples / "
                f"data.batch_size)={expected_steps}; got "
                f"{sections['training']['steps_per_epoch']!r}."
            )
        checkpoint_config = cast(
            Mapping[str, object], sections["training"]["checkpoint"]
        )
        if (
            checkpoint_config["monitor"] != "val/loss"
            or checkpoint_config["mode"] != "min"
        ):
            raise SemanticConfigurationError(
                "training.checkpoint must select the minimum val/loss metric."
            )
        if (
            not evaluation
            and sections["run"]["test_after_fit"]
            and not checkpoint_config["enabled"]
        ):
            raise SemanticConfigurationError(
                "Court alignment training requires checkpointing for best-checkpoint testing."
            )
        if (
            checkpoint_config["enabled"]
            and cast(int, checkpoint_config["save_top_k"]) == 0
        ):
            raise SemanticConfigurationError(
                "Court alignment checkpointing must retain at least one val/loss checkpoint."
            )
        checkpoint = None
        if evaluation:
            evaluation_mapping = COURT_ALIGNMENT_EVALUATION_SCHEMA.validate(
                require_config_mapping(root, "evaluation", path="configuration"),
                path="evaluation",
            )
            raw_checkpoint = evaluation_mapping["checkpoint_path"]
            if raw_checkpoint is not None:
                checkpoint_text = cast(str, raw_checkpoint)
                if not checkpoint_text.strip():
                    raise SemanticConfigurationError(
                        "evaluation.checkpoint_path must not be empty."
                    )
                roots = RuntimePathRoots.from_mapping(
                    cast(ConfigMapping, root["paths"]), repository_root=PROJECT_ROOT
                )
                resolver = PathResolver(roots)
                checkpoint_candidate = Path(checkpoint_text).expanduser()
                checkpoint = (
                    resolver.validate(PathRole.CHECKPOINT, checkpoint_candidate)
                    if checkpoint_candidate.is_absolute()
                    else resolver.resolve(PathRole.CHECKPOINT, checkpoint_text)
                )
        return cls(runtime=runtime, sections=sections, evaluation_checkpoint=checkpoint)


def validate_training_boundary(config: DictConfig) -> None:
    """Validate composed Court Alignment training config before side effects."""
    CourtAlignmentRuntimeConfig.from_config(config)


def validate_evaluation_boundary(config: DictConfig) -> None:
    """Validate composed Court Alignment evaluation config before checkpoint I/O."""
    CourtAlignmentRuntimeConfig.from_config(config, evaluation=True)


register_boundary_validator("court_alignment.train", validate_training_boundary)
register_boundary_validator("court_alignment.evaluate", validate_evaluation_boundary)


__all__ = [
    "COURT_ALIGNMENT_DATA_SCHEMA",
    "COURT_ALIGNMENT_DECODER_SCHEMA",
    "COURT_ALIGNMENT_EVALUATION_SCHEMA",
    "COURT_ALIGNMENT_LOSS_SCHEMA",
    "COURT_ALIGNMENT_METRICS_SCHEMA",
    "COURT_ALIGNMENT_MODEL_SCHEMA",
    "COURT_ALIGNMENT_PATHS_SCHEMA",
    "COURT_ALIGNMENT_RUN_SCHEMA",
    "COURT_ALIGNMENT_TRAINING_SCHEMA",
    "CourtAlignmentRuntimeConfig",
    "validate_evaluation_boundary",
    "validate_training_boundary",
]

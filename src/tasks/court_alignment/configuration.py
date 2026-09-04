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
from src.tasks.court_alignment.evaluation.real_heatmap import (
    DecoderOptions,
    MetricOptions,
    PreprocessOptions,
    RealHeatmapEvaluationRequest,
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

CNN_MODEL_TARGET = "src.tasks.court_alignment.models.cnn.CourtAlignmentCNN"
DINO_MODEL_TARGET = (
    "src.tasks.court_alignment.models.dino_detector."
    "load_pretrained_dino_court_detector"
)


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
        _target(CNN_MODEL_TARGET),
    ),
)


def _dino_model_semantics(value: Mapping[str, object]) -> None:
    _target(DINO_MODEL_TARGET)(value)
    if value["input_mode"] not in {"repeat_rgb", "learnable_1x1", "red_only"}:
        raise SemanticConfigurationError(
            "model.input_mode must be repeat_rgb, learnable_1x1, or red_only."
        )
    for name in ("repository", "checkpoint_path", "device"):
        if not cast(str, value[name]).strip():
            raise SemanticConfigurationError(f"model.{name} must not be empty.")
    short_side = value["short_side"]
    max_long_side = value["max_long_side"]
    if type(short_side) is not int or short_side <= 0:
        raise SemanticConfigurationError("model.short_side must be a positive integer.")
    if type(max_long_side) is not int or max_long_side < short_side:
        raise SemanticConfigurationError(
            "model.max_long_side must be an integer no smaller than short_side."
        )
    if short_side != 800 or max_long_side != 1333:
        raise SemanticConfigurationError(
            "DINO ablations must retain the released 800/1333 evaluation resize."
        )
    rank = value["lora_rank"]
    if type(rank) is not int or rank <= 0:
        raise SemanticConfigurationError("model.lora_rank must be a positive integer.")
    _positive(value["lora_alpha"], path="model.lora_alpha")
    dropout = value["lora_dropout"]
    if type(dropout) not in (int, float):
        raise SemanticConfigurationError("model.lora_dropout must lie in [0,1).")
    if not 0.0 <= float(cast(int | float, dropout)) < 1.0:
        raise SemanticConfigurationError("model.lora_dropout must lie in [0,1).")
    targets = cast(Sequence[object], value["lora_target_modules"])
    if not targets or any(type(name) is not str or not name for name in targets):
        raise SemanticConfigurationError(
            "model.lora_target_modules must contain non-empty strings."
        )


COURT_ALIGNMENT_DINO_MODEL_SCHEMA = _schema(
    "model",
    {
        "_target_": ConfigField.of(str),
        "repository": ConfigField.of(str),
        "checkpoint_path": ConfigField.of(str),
        "device": ConfigField.of(str),
        "input_mode": ConfigField.of(str),
        "short_side": ConfigField.of(int),
        "max_long_side": ConfigField.of(int),
        "lora_rank": ConfigField.of(int),
        "lora_alpha": _number(),
        "lora_dropout": _number(),
        "lora_target_modules": ConfigField.sequence(ConfigField.of(str)),
    },
    semantic_checks=(_dino_model_semantics,),
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

_DINO_LOSS_WEIGHTS_SCHEMA = _schema(
    "loss.weights",
    {
        "_target_": ConfigField.of(str),
        "classification": _number(),
        "bbox": _number(),
        "giou": _number(),
        "scale": _number(),
        "axis": _number(),
    },
    semantic_checks=(
        _target(
            "src.tasks.court_alignment.training.detr_losses.CourtDetrLossWeights"
        ),
    ),
)


def _dino_loss_semantics(value: Mapping[str, object]) -> None:
    if value["num_classes"] != 1:
        raise SemanticConfigurationError("loss.num_classes must equal one.")
    alpha = float(cast(int | float, value["focal_alpha"]))
    gamma = float(cast(int | float, value["focal_gamma"]))
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise SemanticConfigurationError("loss.focal_alpha must lie in [0,1].")
    if not math.isfinite(gamma) or gamma < 0.0:
        raise SemanticConfigurationError(
            "loss.focal_gamma must be finite and non-negative."
        )
    weights = cast(Mapping[str, object], value["weights"])
    names = ("classification", "bbox", "giou", "scale", "axis")
    for name in names:
        _positive(weights[name], path=f"loss.weights.{name}", allow_zero=True)
    if not any(float(cast(int | float, weights[name])) > 0.0 for name in names):
        raise SemanticConfigurationError(
            "At least one DINO loss weight must be positive."
        )


COURT_ALIGNMENT_DINO_LOSS_SCHEMA = _schema(
    "loss",
    {
        "_target_": ConfigField.of(str),
        "num_classes": ConfigField.of(int),
        "weights": ConfigField.mapping(_DINO_LOSS_WEIGHTS_SCHEMA),
        "focal_alpha": _number(),
        "focal_gamma": _number(),
        "auxiliary_loss": ConfigField.of(bool),
    },
    semantic_checks=(
        _target("src.tasks.court_alignment.training.detr_losses.CourtDetrCriterion"),
        _dino_loss_semantics,
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
COURT_ALIGNMENT_DINO_DECODER_SCHEMA = _schema(
    "decoder",
    {
        "threshold": _number(),
        "class_index": ConfigField.of(int),
        "top_k": ConfigField.of(int),
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
COURT_ALIGNMENT_DINO_METRICS_SCHEMA = _schema(
    "metrics",
    {
        "_target_": ConfigField.of(str),
        "match_max_corner_error_px": _number(),
    },
    semantic_checks=(
        _target("src.tasks.court_alignment.training.detr_metrics.CourtDetrMetrics"),
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

_REAL_HEATMAP_PREPROCESS_SCHEMA = _schema(
    "real_evaluation.preprocess",
    {
        "method": ConfigField.of(str),
        "output_size": ConfigField.of(list, tuple),
        "padding_value": _number(),
        "content_fraction": _number(),
    },
)
COURT_ALIGNMENT_REAL_HEATMAP_SCHEMA = _schema(
    "real_evaluation",
    {
        "archive_path": ConfigField.of(str, type(None)),
        "manifest_path": ConfigField.of(str, type(None)),
        "alignment_path": ConfigField.of(str, type(None)),
        "checkpoint_path": ConfigField.of(str, type(None)),
        "output_dir": ConfigField.of(str),
        "device": ConfigField.of(str),
        "preprocess": ConfigField.mapping(_REAL_HEATMAP_PREPROCESS_SCHEMA),
        "training_scale_range_px_per_metre": ConfigField.of(list, tuple),
    },
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
        sections["data"] = COURT_ALIGNMENT_DATA_SCHEMA.validate(
            require_config_mapping(root, "data", path="configuration"),
            path="data",
        )
        raw_model = require_config_mapping(root, "model", path="configuration")
        model_target = raw_model.get("_target_")
        is_dino = model_target == DINO_MODEL_TARGET
        model_schema = (
            COURT_ALIGNMENT_DINO_MODEL_SCHEMA
            if is_dino
            else COURT_ALIGNMENT_MODEL_SCHEMA
        )
        loss_schema = (
            COURT_ALIGNMENT_DINO_LOSS_SCHEMA
            if is_dino
            else COURT_ALIGNMENT_LOSS_SCHEMA
        )
        decoder_schema = (
            COURT_ALIGNMENT_DINO_DECODER_SCHEMA
            if is_dino
            else COURT_ALIGNMENT_DECODER_SCHEMA
        )
        metrics_schema = (
            COURT_ALIGNMENT_DINO_METRICS_SCHEMA
            if is_dino
            else COURT_ALIGNMENT_METRICS_SCHEMA
        )
        for name, schema in (
            ("model", model_schema),
            ("loss", loss_schema),
            ("decoder", decoder_schema),
            ("metrics", metrics_schema),
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
        if is_dino:
            image_size = data["image_size"]
            resolved_image_size = (
                (image_size, image_size)
                if type(image_size) is int
                else tuple(cast(Sequence[int], image_size))
            )
            if resolved_image_size != (800, 800):
                raise SemanticConfigurationError(
                    "DINO court alignment requires data.image_size=800x800."
                )
            top_k = cast(int, sections["decoder"]["top_k"])
            if top_k <= 0:
                raise SemanticConfigurationError(
                    "decoder.top_k must be a positive integer."
                )
            if cast(int, data["max_courts"]) > top_k:
                raise SemanticConfigurationError(
                    "decoder.top_k must cover data.max_courts."
                )
            if cast(int, sections["decoder"]["class_index"]) != 0:
                raise SemanticConfigurationError(
                    "The one-class DINO court head requires decoder.class_index=0."
                )
            if cast(int, sections["loss"]["num_classes"]) != 1:
                raise SemanticConfigurationError(
                    "The DINO court criterion requires loss.num_classes=1."
                )
            threshold = float(
                cast(int | float, sections["decoder"]["threshold"])
            )
            if not 0.0 <= threshold <= 1.0:
                raise SemanticConfigurationError(
                    "decoder.threshold must lie in [0,1]."
                )
            _positive(
                sections["metrics"]["match_max_corner_error_px"],
                path="metrics.match_max_corner_error_px",
            )
        else:
            if cast(int, data["max_courts"]) > cast(
                int, sections["decoder"]["max_instances"]
            ):
                raise SemanticConfigurationError(
                    "decoder.max_instances must cover data.max_courts."
                )
            if cast(int, data["max_courts"]) > cast(
                int, sections["decoder"]["max_peaks"]
            ):
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


def _resolve_explicit_path(
    resolver: PathResolver,
    *,
    role: PathRole,
    value: str,
    name: str,
) -> Path:
    if not value.strip() or value != value.strip():
        raise SemanticConfigurationError(f"{name} must be a non-empty trimmed path.")
    candidate = Path(value).expanduser()
    resolved: Path = (
        resolver.validate(role, candidate)
        if candidate.is_absolute()
        else resolver.resolve(role, value)
    )
    return resolved


@dataclass(frozen=True, slots=True)
class CourtAlignmentRealHeatmapRuntimeConfig:
    """Typed boundary for measured line-heatmap checkpoint evaluation."""

    sections: Mapping[str, Mapping[str, object]]
    request: RealHeatmapEvaluationRequest | None

    @classmethod
    def from_config(
        cls, config: DictConfig | Mapping[str, object]
    ) -> CourtAlignmentRealHeatmapRuntimeConfig:
        resolved = as_config_mapping(config, path="configuration")
        root = exact_config_mapping(
            resolved,
            path="configuration",
            required_keys={"paths", "model", "decoder", "metrics", "real_evaluation"},
        )
        paths = COURT_ALIGNMENT_PATHS_SCHEMA.validate(
            require_config_mapping(root, "paths", path="configuration"),
            path="paths",
        )
        sections: dict[str, Mapping[str, object]] = {}
        for name, schema in (
            ("model", COURT_ALIGNMENT_MODEL_SCHEMA),
            ("decoder", COURT_ALIGNMENT_DECODER_SCHEMA),
            ("metrics", COURT_ALIGNMENT_METRICS_SCHEMA),
            ("real_evaluation", COURT_ALIGNMENT_REAL_HEATMAP_SCHEMA),
        ):
            sections[name] = schema.validate(
                require_config_mapping(root, name, path="configuration"), path=name
            )
        if cast(int, sections["model"]["num_keypoints"]) != 14:
            raise SemanticConfigurationError(
                "model.num_keypoints must preserve the KP14 contract."
            )
        decoder_mapping = sections["decoder"]
        decoder = DecoderOptions(
            threshold=float(cast(int | float, decoder_mapping["threshold"])),
            nms_kernel=cast(int, decoder_mapping["nms_kernel"]),
            max_peaks=cast(int, decoder_mapping["max_peaks"]),
            subpixel_refine=cast(bool, decoder_mapping["subpixel_refine"]),
            cluster_distance_px=float(
                cast(int | float, decoder_mapping["cluster_distance_px"])
            ),
            max_instances=cast(int, decoder_mapping["max_instances"]),
        )
        metrics_mapping = sections["metrics"]
        if (
            float(cast(int | float, metrics_mapping["threshold"])) != decoder.threshold
            or cast(int, metrics_mapping["nms_kernel"]) != decoder.nms_kernel
            or cast(int, metrics_mapping["max_peaks"]) != decoder.max_peaks
        ):
            raise SemanticConfigurationError(
                "metrics decoder fields must resolve to decoder.threshold/nms_kernel/max_peaks."
            )
        metrics = MetricOptions(
            match_max_error_px=float(
                cast(int | float, metrics_mapping["match_max_error_px"])
            ),
            minimum_common_keypoints=cast(
                int, metrics_mapping["minimum_common_keypoints"]
            ),
            minimum_visible_keypoints=cast(
                int, metrics_mapping["minimum_visible_keypoints"]
            ),
            minimum_visible_fraction=float(
                cast(int | float, metrics_mapping["minimum_visible_fraction"])
            ),
            minimum_sim2_keypoints=cast(int, metrics_mapping["minimum_sim2_keypoints"]),
        )
        evaluation = sections["real_evaluation"]
        preprocess_value = evaluation["preprocess"]
        if not isinstance(preprocess_value, Mapping):
            raise ConfigurationTypeError(
                "real_evaluation.preprocess must be a mapping."
            )
        preprocess_mapping = cast(Mapping[str, object], preprocess_value)
        output_size_raw = cast(Sequence[object], preprocess_mapping["output_size"])
        if len(output_size_raw) != 2 or any(
            type(item) is not int for item in output_size_raw
        ):
            raise SemanticConfigurationError(
                "real_evaluation.preprocess.output_size must contain two integers."
            )
        preprocess = PreprocessOptions(
            method=cast(str, preprocess_mapping["method"]),
            output_size=(cast(int, output_size_raw[0]), cast(int, output_size_raw[1])),
            padding_value=float(cast(int | float, preprocess_mapping["padding_value"])),
            content_fraction=float(
                cast(int | float, preprocess_mapping["content_fraction"])
            ),
        )
        if preprocess.output_size != (256, 256):
            raise SemanticConfigurationError(
                "real_evaluation.preprocess.output_size must be [256,256]."
            )
        scale_range_raw = cast(
            Sequence[object], evaluation["training_scale_range_px_per_metre"]
        )
        if len(scale_range_raw) != 2 or any(
            type(item) not in {float, int} for item in scale_range_raw
        ):
            raise SemanticConfigurationError(
                "real_evaluation.training_scale_range_px_per_metre must contain two numbers."
            )
        scale_range = (
            float(cast(float | int, scale_range_raw[0])),
            float(cast(float | int, scale_range_raw[1])),
        )
        path_keys = (
            "archive_path",
            "manifest_path",
            "alignment_path",
            "checkpoint_path",
        )
        raw_paths = [evaluation[key] for key in path_keys]
        if any(value is None for value in raw_paths):
            if not all(value is None for value in raw_paths):
                raise SemanticConfigurationError(
                    "real_evaluation input paths are all-or-none explicit values."
                )
            return cls(sections=sections, request=None)
        roots = RuntimePathRoots.from_mapping(paths, repository_root=PROJECT_ROOT)
        resolver = PathResolver(roots)
        output_text = cast(str, evaluation["output_dir"])
        output_path = _resolve_explicit_path(
            resolver,
            role=PathRole.OUTPUT,
            value=output_text,
            name="real_evaluation.output_dir",
        )
        request = RealHeatmapEvaluationRequest(
            archive_path=_resolve_explicit_path(
                resolver,
                role=PathRole.DATA,
                value=cast(str, evaluation["archive_path"]),
                name="real_evaluation.archive_path",
            ),
            manifest_path=_resolve_explicit_path(
                resolver,
                role=PathRole.DATA,
                value=cast(str, evaluation["manifest_path"]),
                name="real_evaluation.manifest_path",
            ),
            alignment_path=_resolve_explicit_path(
                resolver,
                role=PathRole.DATA,
                value=cast(str, evaluation["alignment_path"]),
                name="real_evaluation.alignment_path",
            ),
            checkpoint_path=_resolve_explicit_path(
                resolver,
                role=PathRole.CHECKPOINT,
                value=cast(str, evaluation["checkpoint_path"]),
                name="real_evaluation.checkpoint_path",
            ),
            output_dir=output_path,
            device=cast(str, evaluation["device"]),
            preprocess=preprocess,
            decoder=decoder,
            metrics=metrics,
            training_scale_range_px_per_metre=scale_range,
        )
        return cls(sections=sections, request=request)

    def require_request(self) -> RealHeatmapEvaluationRequest:
        """Return execution paths or fail before any filesystem/model side effect."""
        if self.request is None:
            raise ValueError(
                "Measured evaluation requires archive, manifest, alignment, and checkpoint paths."
            )
        return self.request


def validate_training_boundary(config: DictConfig) -> None:
    """Validate composed Court Alignment training config before side effects."""
    CourtAlignmentRuntimeConfig.from_config(config)


def validate_evaluation_boundary(config: DictConfig) -> None:
    """Validate composed Court Alignment evaluation config before checkpoint I/O."""
    CourtAlignmentRuntimeConfig.from_config(config, evaluation=True)


def validate_real_heatmap_evaluation_boundary(config: DictConfig) -> None:
    """Validate the measured line-heatmap evaluation boundary."""
    CourtAlignmentRealHeatmapRuntimeConfig.from_config(config)


register_boundary_validator("court_alignment.train", validate_training_boundary)
register_boundary_validator("court_alignment.evaluate", validate_evaluation_boundary)
register_boundary_validator(
    "court_alignment.evaluate_real_heatmap", validate_real_heatmap_evaluation_boundary
)


__all__ = [
    "CNN_MODEL_TARGET",
    "COURT_ALIGNMENT_DATA_SCHEMA",
    "COURT_ALIGNMENT_DECODER_SCHEMA",
    "COURT_ALIGNMENT_DINO_DECODER_SCHEMA",
    "COURT_ALIGNMENT_DINO_LOSS_SCHEMA",
    "COURT_ALIGNMENT_DINO_METRICS_SCHEMA",
    "COURT_ALIGNMENT_DINO_MODEL_SCHEMA",
    "COURT_ALIGNMENT_EVALUATION_SCHEMA",
    "COURT_ALIGNMENT_LOSS_SCHEMA",
    "COURT_ALIGNMENT_METRICS_SCHEMA",
    "COURT_ALIGNMENT_MODEL_SCHEMA",
    "COURT_ALIGNMENT_PATHS_SCHEMA",
    "COURT_ALIGNMENT_REAL_HEATMAP_SCHEMA",
    "COURT_ALIGNMENT_RUN_SCHEMA",
    "COURT_ALIGNMENT_TRAINING_SCHEMA",
    "CourtAlignmentRuntimeConfig",
    "CourtAlignmentRealHeatmapRuntimeConfig",
    "DINO_MODEL_TARGET",
    "validate_evaluation_boundary",
    "validate_real_heatmap_evaluation_boundary",
    "validate_training_boundary",
]

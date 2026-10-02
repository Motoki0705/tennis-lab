"""Strict configuration and path contracts for ball-detection runtimes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Literal, cast

from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import (
    BaseRunConfig,
    BaseTrainingConfig,
    CheckpointInput,
    resolve_checkpoint_input,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.hydra import register_boundary_validator
from src.utils.models.components.ffn_layers import SUPPORTED_FFN_TYPES
from src.utils.paths import PROJECT_ROOT

ConfigMapping = Mapping[str, Any]


def as_mapping(value: object, *, path: str) -> ConfigMapping:
    """Return a resolved mapping with string keys only."""
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise ConfigurationTypeError(
            f"{path}: expected mapping, got {type(value).__name__}."
        )
    if any(not isinstance(key, str) for key in value):
        raise ConfigurationTypeError(f"{path}: all keys must be strings.")
    return cast(ConfigMapping, value)


def exact_mapping(
    value: object,
    *,
    path: str,
    required: set[str],
    optional: frozenset[str] | set[str] = frozenset(),
) -> ConfigMapping:
    """Reject missing and unknown keys and return a plain mapping."""
    mapping = as_mapping(value, path=path)
    missing = sorted(required - set(mapping))
    if missing:
        raise MissingConfigurationKeyError(
            "Missing required configuration key(s): "
            + ", ".join(f"{path}.{key}" for key in missing)
            + "."
        )
    unknown = sorted(set(mapping) - required - optional)
    if unknown:
        raise UnknownConfigurationKeyError(
            "Unknown configuration key(s): "
            + ", ".join(f"{path}.{key}" for key in unknown)
            + "."
        )
    return mapping


def typed(
    mapping: ConfigMapping,
    key: str,
    expected: type[object] | tuple[type[object], ...],
    *,
    path: str,
) -> object:
    """Read a required exact-typed value."""
    if key not in mapping:
        raise MissingConfigurationKeyError(
            f"Missing required configuration key: {path}.{key}."
        )
    accepted = expected if isinstance(expected, tuple) else (expected,)
    value = mapping[key]
    if type(value) not in accepted:
        names = " | ".join(item.__name__ for item in accepted)
        raise ConfigurationTypeError(
            f"{path}.{key}: expected {names}, got {type(value).__name__}."
        )
    return value


def sequence(
    value: object, *, path: str, length: int | None = None
) -> tuple[object, ...]:
    """Validate a non-string sequence."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationTypeError(
            f"{path}: expected sequence, got {type(value).__name__}."
        )
    result = tuple(value)
    if length is not None and len(result) != length:
        raise SemanticConfigurationError(
            f"{path}: expected {length} items, got {len(result)}."
        )
    return result


def typed_sequence(
    value: object,
    *,
    path: str,
    item_type: type[object] | tuple[type[object], ...],
    length: int | None = None,
) -> tuple[object, ...]:
    """Validate a fixed or variable-length sequence with exact item types."""
    result = sequence(value, path=path, length=length)
    accepted = item_type if isinstance(item_type, tuple) else (item_type,)
    for index, item in enumerate(result):
        if type(item) not in accepted:
            names = " | ".join(candidate.__name__ for candidate in accepted)
            raise ConfigurationTypeError(
                f"{path}[{index}]: expected {names}, got {type(item).__name__}."
            )
    return result


def _required_sequence(
    mapping: ConfigMapping,
    key: str,
    *,
    path: str,
    item_type: type[object] | tuple[type[object], ...],
    length: int | None = None,
) -> tuple[object, ...]:
    return typed_sequence(
        typed(mapping, key, (list, tuple), path=path),
        path=f"{path}.{key}",
        item_type=item_type,
        length=length,
    )


def _required_number(mapping: ConfigMapping, key: str, *, path: str) -> float:
    return float(cast(float | int, typed(mapping, key, (float, int), path=path)))


def _optional_number(mapping: ConfigMapping, key: str, *, path: str) -> float | None:
    value = typed(mapping, key, (float, int, type(None)), path=path)
    return None if value is None else float(cast(float | int, value))


def _positive(value: float | int, *, path: str, allow_zero: bool = False) -> None:
    invalid = not math.isfinite(value) or (value < 0 if allow_zero else value <= 0)
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise SemanticConfigurationError(f"{path} must be {qualifier}.")


def _validate_rgb(mapping: ConfigMapping, key: str, *, path: str) -> None:
    values = _required_sequence(
        mapping,
        key,
        path=path,
        item_type=int,
        length=3,
    )
    if any(cast(int, value) < 0 or cast(int, value) > 255 for value in values):
        raise SemanticConfigurationError(
            f"{path}.{key} must contain RGB integers in [0, 255]."
        )


def _validate_relative_child(value: object, *, path: str) -> str:
    relative = cast(str, value)
    if type(value) is not str or not relative.strip() or relative != relative.strip():
        raise ConfigurationTypeError(
            f"{path}: expected non-empty trimmed str, got {type(value).__name__}."
        )
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SemanticConfigurationError(
            f"{path} must be a non-escaping relative path, got {relative!r}."
        )
    return relative


def _validate_string_sequence(
    value: object, *, path: str, allow_empty: bool = False
) -> tuple[str, ...]:
    result = cast(
        tuple[str, ...],
        typed_sequence(value, path=path, item_type=str),
    )
    if (not result and not allow_empty) or any(
        not item.strip() or item != item.strip() for item in result
    ):
        qualifier = "a sequence" if allow_empty else "a non-empty sequence"
        raise SemanticConfigurationError(
            f"{path} must be {qualifier} of non-empty trimmed strings."
        )
    return result


@dataclass(frozen=True, slots=True)
class BallRuntimePaths:
    """All shared root roles and the derived-path resolver."""

    resolver: PathResolver

    @classmethod
    def from_config(cls, config: object) -> BallRuntimePaths:
        root = as_mapping(config, path="configuration")
        paths = typed(root, "paths", (dict, DictConfig), path="configuration")
        roots = RuntimePathRoots.from_mapping(
            as_mapping(paths, path="paths"), repository_root=PROJECT_ROOT
        )
        return cls(PathResolver(roots))

    def data(self, relative: str) -> Path:
        resolved: Path = self.resolver.resolve(PathRole.DATA, relative)
        return resolved

    def project(self, relative: str) -> Path:
        resolved: Path = self.resolver.resolve(PathRole.PROJECT, relative)
        return resolved

    def checkpoint(self, relative: str) -> Path:
        resolved: Path = self.resolver.resolve(PathRole.CHECKPOINT, relative)
        return resolved

    def checkpoint_input(self, mapping: ConfigMapping, key: str, *, path: str) -> CheckpointInput:
        declared = resolve_checkpoint_input(mapping, key, path=path, resolver=self.resolver)
        if declared is None:
            raise SemanticConfigurationError(f"{path}.{key}: checkpoint input is required.")
        return declared

    def output(self, relative: str) -> Path:
        resolved: Path = self.resolver.resolve(PathRole.OUTPUT, relative)
        return resolved

    def artifact(self, relative: str) -> Path:
        resolved: Path = self.resolver.resolve(PathRole.ARTIFACT, relative)
        return resolved

    def cache(self, relative: str) -> Path:
        resolved: Path = self.resolver.resolve(PathRole.CACHE, relative)
        return resolved

    def external_asset(self, relative: str) -> Path:
        resolved: Path = self.resolver.resolve(PathRole.EXTERNAL_ASSET, relative)
        return resolved


@dataclass(frozen=True, slots=True)
class DetailedEvaluationConfig:
    """Typed values consumed by the detailed checkpoint evaluator."""

    splits: tuple[Literal["val", "test"], ...]
    output_json_name: str
    max_batches_per_split: int | None
    edge_threshold_ratio: float

    @classmethod
    def from_config(cls, config: object) -> DetailedEvaluationConfig:
        root = as_mapping(config, path="configuration")
        evaluation = exact_mapping(
            typed(root, "evaluation", (dict, DictConfig), path="configuration"),
            path="evaluation",
            required={
                "splits",
                "output_json_name",
                "max_batches_per_split",
                "analysis",
            },
        )
        raw_splits = _required_sequence(
            evaluation, "splits", path="evaluation", item_type=str
        )
        splits = tuple(cast(str, split) for split in raw_splits)
        if not splits or len(set(splits)) != len(splits):
            raise SemanticConfigurationError(
                "evaluation.splits must be a non-empty sequence without duplicates."
            )
        if any(split not in {"val", "test"} for split in splits):
            raise SemanticConfigurationError(
                "evaluation.splits may contain only 'val' and 'test'."
            )
        output_json_name = cast(
            str,
            typed(evaluation, "output_json_name", str, path="evaluation"),
        )
        _validate_relative_child(output_json_name, path="evaluation.output_json_name")
        raw_max_batches = typed(
            evaluation,
            "max_batches_per_split",
            (int, type(None)),
            path="evaluation",
        )
        max_batches = None if raw_max_batches is None else cast(int, raw_max_batches)
        if max_batches is not None:
            _positive(max_batches, path="evaluation.max_batches_per_split")
        analysis = exact_mapping(
            evaluation["analysis"],
            path="evaluation.analysis",
            required={"edge_threshold_ratio"},
        )
        edge_threshold = _required_number(
            analysis, "edge_threshold_ratio", path="evaluation.analysis"
        )
        if not 0.0 <= edge_threshold <= 0.5:
            raise SemanticConfigurationError(
                "evaluation.analysis.edge_threshold_ratio must be in [0, 0.5]."
            )
        return cls(
            splits=cast(tuple[Literal["val", "test"], ...], splits),
            output_json_name=output_json_name,
            max_batches_per_split=max_batches,
            edge_threshold_ratio=edge_threshold,
        )


_COMMON_MODEL = {
    "name",
    "input_mode",
    "in_channels",
    "num_classes",
    "num_frames",
    "input_layout",
}


def validate_model(
    config: object, *, paths: BallRuntimePaths | None = None
) -> ConfigMapping:
    """Validate the selected model variant without model construction."""
    root = as_mapping(config, path="configuration")
    model = as_mapping(
        typed(root, "model", (dict, DictConfig), path="configuration"), path="model"
    )
    name = typed(model, "name", str, path="model")
    if name == "stunet":
        allowed = _COMMON_MODEL | {"mdd_a", "mdd_b"}
    elif name == "conv_next_unet":
        allowed = _COMMON_MODEL | {"dims", "depth", "drop_path_prob", "mdd_a", "mdd_b"}
    elif name == "dinov3_rope":
        allowed = _COMMON_MODEL | {"image_size", "backbone", "decoder", "heatmap_head"}
    else:
        raise SemanticConfigurationError(f"model.name: unsupported value {name!r}.")
    model = exact_mapping(model, path="model", required=allowed)
    for key in ("in_channels", "num_classes", "num_frames"):
        value = cast(int, typed(model, key, int, path="model"))
        _positive(value, path=f"model.{key}")
    typed(model, "input_mode", str, path="model")
    typed(model, "input_layout", str, path="model")
    if name in {"stunet", "conv_next_unet"}:
        _required_number(model, "mdd_a", path="model")
        _required_number(model, "mdd_b", path="model")
    if name == "conv_next_unet":
        _required_sequence(model, "dims", path="model", item_type=int, length=4)
        _positive(
            cast(int, typed(model, "depth", int, path="model")), path="model.depth"
        )
        drop_path_prob = _required_number(model, "drop_path_prob", path="model")
        if not 0.0 <= drop_path_prob < 1.0:
            raise SemanticConfigurationError("model.drop_path_prob must be in [0, 1).")
    if name == "dinov3_rope":
        _required_sequence(model, "image_size", path="model", item_type=int, length=2)
        backbone = exact_mapping(
            typed(model, "backbone", (dict, DictConfig), path="model"),
            path="model.backbone",
            required={
                "name",
                "repository_path",
                "checkpoint_path",
                "strict",
                "train_mode",
                "last_n_blocks",
                "lora",
            },
        )
        typed(backbone, "name", str, path="model.backbone")
        repository_path = cast(
            str,
            typed(backbone, "repository_path", str, path="model.backbone"),
        )
        checkpoint_path = cast(
            str,
            typed(backbone, "checkpoint_path", str, path="model.backbone"),
        )
        typed(backbone, "strict", bool, path="model.backbone")
        train_mode = cast(
            str, typed(backbone, "train_mode", str, path="model.backbone")
        )
        if train_mode not in {"frozen", "last_n_blocks", "full"}:
            raise SemanticConfigurationError(
                "model.backbone.train_mode must be frozen, last_n_blocks, or full."
            )
        _positive(
            cast(int, typed(backbone, "last_n_blocks", int, path="model.backbone")),
            path="model.backbone.last_n_blocks",
            allow_zero=True,
        )
        lora = exact_mapping(
            typed(backbone, "lora", (dict, DictConfig), path="model.backbone"),
            path="model.backbone.lora",
            required={"enabled", "rank", "alpha", "dropout", "target_modules"},
        )
        typed(lora, "enabled", bool, path="model.backbone.lora")
        _positive(
            cast(int, typed(lora, "rank", int, path="model.backbone.lora")),
            path="model.backbone.lora.rank",
        )
        _required_number(lora, "alpha", path="model.backbone.lora")
        lora_dropout = _required_number(lora, "dropout", path="model.backbone.lora")
        if not 0.0 <= lora_dropout < 1.0:
            raise SemanticConfigurationError(
                "model.backbone.lora.dropout must be in [0, 1)."
            )
        _required_sequence(
            lora, "target_modules", path="model.backbone.lora", item_type=str
        )
        decoder = exact_mapping(
            typed(model, "decoder", (dict, DictConfig), path="model"),
            path="model.decoder",
            required={
                "dim",
                "num_layers",
                "num_heads",
                "head_dim",
                "ffn_dim",
                "rope_dim",
                "rope_base",
                "dropout",
                "attention_type",
                "n_kv_heads",
                "ffn_type",
                "gradient_checkpointing",
            },
        )
        for key in (
            "dim",
            "num_layers",
            "num_heads",
            "head_dim",
            "ffn_dim",
            "rope_dim",
        ):
            _positive(
                cast(int, typed(decoder, key, int, path="model.decoder")),
                path=f"model.decoder.{key}",
            )
        rope_base = typed(
            decoder, "rope_base", (float, int, list, tuple), path="model.decoder"
        )
        if isinstance(rope_base, (list, tuple)):
            _required_sequence(
                decoder,
                "rope_base",
                path="model.decoder",
                item_type=(float, int),
                length=3,
            )
        decoder_dropout = _required_number(decoder, "dropout", path="model.decoder")
        if not 0.0 <= decoder_dropout < 1.0:
            raise SemanticConfigurationError("model.decoder.dropout must be in [0, 1).")
        typed(
            decoder,
            "gradient_checkpointing",
            bool,
            path="model.decoder",
        )
        attention_type = cast(
            str, typed(decoder, "attention_type", str, path="model.decoder")
        )
        if attention_type != "mha":
            raise SemanticConfigurationError(
                "model.decoder.attention_type must be 'mha'."
            )
        typed(decoder, "n_kv_heads", type(None), path="model.decoder")
        ffn_type = cast(str, typed(decoder, "ffn_type", str, path="model.decoder"))
        if ffn_type not in SUPPORTED_FFN_TYPES:
            raise SemanticConfigurationError(
                "model.decoder.ffn_type must be one of "
                f"{sorted(SUPPORTED_FFN_TYPES)!r}."
            )
        heatmap_head = exact_mapping(
            typed(model, "heatmap_head", (dict, DictConfig), path="model"),
            path="model.heatmap_head",
            required={"min_channels"},
        )
        _positive(
            cast(
                int,
                typed(heatmap_head, "min_channels", int, path="model.heatmap_head"),
            ),
            path="model.heatmap_head.min_channels",
        )
        if paths is not None:
            paths.external_asset(repository_path)
            paths.checkpoint(checkpoint_path)
    return model


_AUGMENTATION_FIELDS: Mapping[str, set[str]] = {
    "camera_rotation": {
        "enabled",
        "prob",
        "max_center_angle_deg",
        "max_angular_velocity_deg_per_frame",
        "border_mode",
    },
    "horizontal_flip": {"enabled", "prob"},
    "affine": {
        "enabled",
        "prob",
        "rotation_deg_range",
        "scale_range",
        "translate_x_ratio_range",
        "translate_y_ratio_range",
        "shear_x_deg_range",
        "shear_y_deg_range",
        "border_mode",
    },
    "scale_and_crop": {"enabled", "prob", "scale_range", "border_mode"},
    "ball_area_zero_mask": {
        "enabled",
        "prob",
        "mask_width_ratio_range",
        "mask_height_ratio_range",
        "num_frames_range",
    },
    "brightness_gain": {"enabled", "jitter"},
    "contrast": {"enabled", "jitter"},
    "gamma": {"enabled", "jitter"},
    "gaussian_noise": {"enabled", "std"},
    "gaussian_blur": {"enabled", "prob", "kernel_size"},
    "normalize_imagenet": {"enabled", "mean", "std"},
}


def validate_augmentation(value: object) -> ConfigMapping:
    augmentation = exact_mapping(
        value, path="data.augmentation", required=set(_AUGMENTATION_FIELDS)
    )
    for name, fields in _AUGMENTATION_FIELDS.items():
        section = exact_mapping(
            augmentation[name], path=f"data.augmentation.{name}", required=fields
        )
        typed(section, "enabled", bool, path=f"data.augmentation.{name}")
    probability_sections = (
        "camera_rotation",
        "horizontal_flip",
        "affine",
        "scale_and_crop",
        "ball_area_zero_mask",
        "gaussian_blur",
    )
    for name in probability_sections:
        section = as_mapping(augmentation[name], path=f"data.augmentation.{name}")
        probability = _required_number(
            section, "prob", path=f"data.augmentation.{name}"
        )
        if not 0.0 <= probability <= 1.0:
            raise SemanticConfigurationError(
                f"data.augmentation.{name}.prob must be in [0, 1]."
            )
    for name in ("camera_rotation", "affine", "scale_and_crop"):
        section = as_mapping(augmentation[name], path=f"data.augmentation.{name}")
        typed(section, "border_mode", str, path=f"data.augmentation.{name}")
    camera_rotation = as_mapping(
        augmentation["camera_rotation"], path="data.augmentation.camera_rotation"
    )
    _required_number(
        camera_rotation,
        "max_center_angle_deg",
        path="data.augmentation.camera_rotation",
    )
    _required_number(
        camera_rotation,
        "max_angular_velocity_deg_per_frame",
        path="data.augmentation.camera_rotation",
    )
    affine = as_mapping(augmentation["affine"], path="data.augmentation.affine")
    for key in (
        "rotation_deg_range",
        "scale_range",
        "translate_x_ratio_range",
        "translate_y_ratio_range",
        "shear_x_deg_range",
        "shear_y_deg_range",
    ):
        _required_sequence(
            affine,
            key,
            path="data.augmentation.affine",
            item_type=(float, int),
            length=2,
        )
    scale_crop = as_mapping(
        augmentation["scale_and_crop"], path="data.augmentation.scale_and_crop"
    )
    _required_sequence(
        scale_crop,
        "scale_range",
        path="data.augmentation.scale_and_crop",
        item_type=(float, int),
        length=2,
    )
    zero_mask = as_mapping(
        augmentation["ball_area_zero_mask"],
        path="data.augmentation.ball_area_zero_mask",
    )
    for key in ("mask_width_ratio_range", "mask_height_ratio_range"):
        _required_sequence(
            zero_mask,
            key,
            path="data.augmentation.ball_area_zero_mask",
            item_type=(float, int),
            length=2,
        )
    _required_sequence(
        zero_mask,
        "num_frames_range",
        path="data.augmentation.ball_area_zero_mask",
        item_type=int,
        length=2,
    )
    for name in ("brightness_gain", "contrast", "gamma"):
        section = as_mapping(augmentation[name], path=f"data.augmentation.{name}")
        _required_number(section, "jitter", path=f"data.augmentation.{name}")
    noise = as_mapping(
        augmentation["gaussian_noise"], path="data.augmentation.gaussian_noise"
    )
    _required_number(noise, "std", path="data.augmentation.gaussian_noise")
    blur = as_mapping(
        augmentation["gaussian_blur"], path="data.augmentation.gaussian_blur"
    )
    _positive(
        cast(
            int, typed(blur, "kernel_size", int, path="data.augmentation.gaussian_blur")
        ),
        path="data.augmentation.gaussian_blur.kernel_size",
    )
    normalize = as_mapping(
        augmentation["normalize_imagenet"],
        path="data.augmentation.normalize_imagenet",
    )
    for key in ("mean", "std"):
        _required_sequence(
            normalize,
            key,
            path="data.augmentation.normalize_imagenet",
            item_type=(float, int),
            length=3,
        )
    return augmentation


STORE_DATA_KEYS = frozenset(
    {"data_dir", "sources", "train_sampling", "train_stride", "eval_stride", "supervision"}
)
_STORE_SOURCES = frozenset({"tracknet", "meiji", "chat_annotation"})


def _validate_store_fields(
    data: ConfigMapping,
    *,
    path: str,
    paths: BallRuntimePaths | None,
) -> None:
    """Validate the ball frame store and source sampling settings."""
    typed(data, "data_dir", str, path=path)
    sources = _validate_string_sequence(data["sources"], path=f"{path}.sources")
    if not sources:
        raise SemanticConfigurationError(f"{path}.sources must not be empty.")
    unknown = sorted(set(sources) - _STORE_SOURCES)
    if unknown or len(set(sources)) != len(sources):
        raise SemanticConfigurationError(
            f"{path}.sources must be distinct names from {sorted(_STORE_SOURCES)}; got {list(sources)}."
        )
    _positive(cast(int, typed(data, "train_stride", int, path=path)), path=f"{path}.train_stride")
    eval_stride = typed(data, "eval_stride", (int, type(None)), path=path)
    if eval_stride is not None:
        _positive(cast(int, eval_stride), path=f"{path}.eval_stride")
    supervision = exact_mapping(
        data["supervision"], path=f"{path}.supervision", required={"positive", "absent", "ignore"}
    )
    for role in ("positive", "absent", "ignore"):
        _validate_string_sequence(supervision[role], path=f"{path}.supervision.{role}")
    from src.tasks.ball_detection.data.supervision import FrameSupervisionPolicy

    try:
        FrameSupervisionPolicy.from_mapping(cast(Mapping[str, Sequence[str]], supervision))
    except ValueError as error:
        raise SemanticConfigurationError(f"{path}.supervision: {error}") from error
    sampling = data["train_sampling"]
    if sampling is not None:
        sampling_path = f"{path}.train_sampling"
        sampling = exact_mapping(
            sampling, path=sampling_path, required={"windows_per_epoch", "seed", "source_weights"}
        )
        _positive(
            cast(int, typed(sampling, "windows_per_epoch", int, path=sampling_path)),
            path=f"{sampling_path}.windows_per_epoch",
        )
        typed(sampling, "seed", int, path=sampling_path)
        weights = as_mapping(sampling["source_weights"], path=f"{sampling_path}.source_weights")
        if set(weights) != set(sources):
            raise SemanticConfigurationError(
                f"{sampling_path}.source_weights must name exactly {path}.sources "
                f"({sorted(sources)}); got {sorted(weights)}."
            )
        for name in weights:
            weight = _required_number(weights, name, path=f"{sampling_path}.source_weights")
            _positive(weight, path=f"{sampling_path}.source_weights.{name}")
    if paths is not None:
        paths.data(cast(str, data["data_dir"]))


def validate_data(
    config: object, *, paths: BallRuntimePaths | None = None
) -> ConfigMapping:
    """Validate the selected data variant and its role-relative paths."""
    root = as_mapping(config, path="configuration")
    data = as_mapping(
        typed(root, "data", (dict, DictConfig), path="configuration"), path="data"
    )
    source = typed(data, "source", str, path="data")
    common = {
        "source",
        "batch_size",
        "num_workers",
        "pin_memory",
        "image_size",
        "heatmap_size",
        "sigma_ratio",
        "max_instances",
        "augmentation",
    }
    if source != "store":
        raise SemanticConfigurationError(
            f"data.source: unsupported value {source!r}; expected 'store'."
        )
    required = common | STORE_DATA_KEYS
    data = exact_mapping(data, path="data", required=required)
    if "augmentation" in data:
        validate_augmentation(data["augmentation"])
    for key in ("num_workers", "max_instances"):
        _positive(
            cast(int, typed(data, key, int, path="data")),
            path=f"data.{key}",
            allow_zero=key == "num_workers",
        )
    typed(data, "pin_memory", bool, path="data")
    for key in ("image_size", "heatmap_size"):
        values = _required_sequence(data, key, path="data", item_type=int, length=2)
        for index, value in enumerate(values):
            _positive(cast(int, value), path=f"data.{key}[{index}]")
    _positive(
        _required_number(data, "sigma_ratio", path="data"), path="data.sigma_ratio"
    )
    typed(data, "data_dir", str, path="data")
    _positive(
        cast(int, typed(data, "batch_size", int, path="data")),
        path="data.batch_size",
    )
    _validate_store_fields(data, path="data", paths=paths)
    return data


def validate_candidate_settings(value: object) -> ConfigMapping:
    """The campaign candidate metric is fixed; no compatibility defaults."""
    expected: dict[str, int | float | bool] = {
        "max_candidates": 8, "nms_kernel": 5, "patch_size": 5,
        "subpixel_refine": True, "radius_source_px": 20.0,
    }
    settings = exact_mapping(value, path="training.validation_candidates", required=set(expected))
    for key, required in expected.items():
        actual = typed(settings, key, (int, float) if type(required) is float else type(required),
                       path="training.validation_candidates")
        if actual != required:
            raise SemanticConfigurationError(f"training.validation_candidates.{key} must be {required!r}")
    return settings


def validate_epoch_candidate_policy(config: DictConfig) -> None:
    """Every detector training entry must validate and retain every epoch."""
    training = as_mapping(config.training, path="training")
    validate_candidate_settings(typed(training, "validation_candidates", (dict, DictConfig), path="training"))
    checkpoint = as_mapping(training["checkpoint"], path="training.checkpoint")
    trainer = as_mapping(training["trainer"], path="training.trainer")
    if checkpoint["enabled"] is not True or checkpoint["save_top_k"] != -1:
        raise SemanticConfigurationError("Ball training must keep every epoch: checkpoint.enabled=true, save_top_k=-1")
    if trainer["check_val_every_n_epoch"] != 1:
        raise SemanticConfigurationError("Ball candidate recall must be validated every epoch")
    fields = {field for _, field, _, _ in Formatter().parse(checkpoint["filename"])}
    if "epoch" not in fields:
        raise SemanticConfigurationError("Ball checkpoint.filename must include {epoch} to retain every epoch")


def validate_training(config: DictConfig) -> None:
    """Validate a complete ball store training composition."""
    root = exact_mapping(
        config,
        path="configuration",
        required={"paths", "model", "data", "loss", "metrics", "training", "run"},
    )
    paths = BallRuntimePaths.from_config(config)
    validate_model(config, paths=paths)
    validate_data(config, paths=paths)
    loss = exact_mapping(
        typed(root, "loss", (dict, DictConfig), path="configuration"),
        path="loss",
        required={"name", "gamma"},
    )
    typed(loss, "name", str, path="loss")
    _required_number(loss, "gamma", path="loss")
    metrics = exact_mapping(
        typed(root, "metrics", (dict, DictConfig), path="configuration"),
        path="metrics",
        required={
            "peak_threshold",
            "ball_distance_threshold",
            "nms_kernel",
            "max_predictions_per_frame",
            "subpixel_refine",
        },
    )
    _validate_metrics(metrics)
    run = exact_mapping(
        root["run"],
        path="run",
        required={
            "output_dir",
            "seed",
            "gpus",
            "resume",
            "init_weights",
            "fast_dev_run",
            "dry_run",
            "test_after_fit",
            "artifact_store",
        },
    )
    training = as_mapping(root["training"], path="training")
    training_fields = {
        "trainer",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "warmup_epochs",
        "min_lr",
        "steps_per_epoch",
        "optimizer",
        "compile",
        "matmul_precision",
        "allow_tf32",
        "checkpoint",
        "early_stopping",
        "lr_monitor",
        "qualitative_logging",
        "qualitative_rendering",
        "gan",
        "validation_candidates",
    }
    training = exact_mapping(training, path="training", required=training_fields)
    optimizer = exact_mapping(
        training["optimizer"], path="training.optimizer", required={"name", "betas"}
    )
    exact_mapping(
        training["compile"],
        path="training.compile",
        required={"enabled", "backend", "mode", "fullgraph", "dynamic"},
    )
    typed(optimizer, "name", str, path="training.optimizer")
    trainer = exact_mapping(
        training["trainer"],
        path="training.trainer",
        required={
            "max_epochs",
            "gradient_clip_val",
            "deterministic",
            "precision",
            "log_every_n_steps",
            "check_val_every_n_epoch",
            "accumulate_grad_batches",
            "reload_dataloaders_every_n_epochs",
            "enable_progress_bar",
            "enable_model_summary",
            "benchmark",
        },
    )
    typed(trainer, "benchmark", bool, path="training.trainer")
    exact_mapping(
        training["checkpoint"],
        path="training.checkpoint",
        required={"enabled", "filename", "monitor", "mode", "save_top_k", "save_last"},
    )
    early_stopping = exact_mapping(
        training["early_stopping"],
        path="training.early_stopping",
        required={
            "enabled",
            "monitor",
            "mode",
            "patience",
            "min_delta",
            "check_on_train_epoch_end",
        },
    )
    exact_mapping(
        training["lr_monitor"],
        path="training.lr_monitor",
        required={"enabled", "interval"},
    )
    exact_mapping(
        training["qualitative_logging"],
        path="training.qualitative_logging",
        required={
            "enabled",
            "every_n_epochs",
            "num_samples",
            "selection_mode",
            "selected_indices",
        },
    )
    qualitative_rendering = exact_mapping(
        training["qualitative_rendering"],
        path="training.qualitative_rendering",
        required={"fps", "draw", "layout"},
    )
    fps = _required_number(
        qualitative_rendering, "fps", path="training.qualitative_rendering"
    )
    _positive(fps, path="training.qualitative_rendering.fps")
    _validate_draw_style(
        qualitative_rendering["draw"], path="training.qualitative_rendering.draw"
    )
    _validate_layout_style(
        qualitative_rendering["layout"], path="training.qualitative_rendering.layout"
    )
    gan = exact_mapping(
        training["gan"],
        path="training.gan",
        required={
            "enabled",
            "target_weight",
            "warmup_epochs",
            "generator_gradient_clip_val",
            "discriminator_gradient_clip_val",
            "soft_argmax_temperature",
            "transition",
            "discriminator",
        },
    )
    gan_enabled = cast(bool, typed(gan, "enabled", bool, path="training.gan"))
    if gan_enabled:
        if trainer["gradient_clip_val"] is not None:
            raise SemanticConfigurationError(
                "training.trainer.gradient_clip_val must be null when "
                "training.gan.enabled=true."
            )
        early_stopping_enabled = cast(
            bool,
            typed(
                early_stopping,
                "enabled",
                bool,
                path="training.early_stopping",
            ),
        )
        if early_stopping_enabled:
            raise SemanticConfigurationError(
                "training.early_stopping.enabled must be false when "
                "training.gan.enabled=true."
            )
    exact_mapping(
        gan["transition"], path="training.gan.transition", required={"start_epoch"}
    )
    discriminator = exact_mapping(
        gan["discriminator"],
        path="training.gan.discriminator",
        required={
            "name",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "ffn_dim",
            "ffn_type",
            "dropout",
            "rope_dim",
            "rope_theta",
            "max_seq_len",
            "invalid_init_std",
            "cls_init_std",
        },
    )
    for key in (
        "hidden_dim",
        "num_layers",
        "num_heads",
        "ffn_dim",
        "rope_dim",
        "max_seq_len",
    ):
        _positive(
            cast(
                int, typed(discriminator, key, int, path="training.gan.discriminator")
            ),
            path=f"training.gan.discriminator.{key}",
        )
    typed(discriminator, "name", str, path="training.gan.discriminator")
    typed(discriminator, "ffn_type", str, path="training.gan.discriminator")
    for key in ("dropout", "rope_theta", "invalid_init_std", "cls_init_std"):
        _required_number(discriminator, key, path="training.gan.discriminator")
    soft_argmax_temperature = _required_number(
        gan, "soft_argmax_temperature", path="training.gan"
    )
    _positive(
        soft_argmax_temperature,
        path="training.gan.soft_argmax_temperature",
    )
    # Task-owned maps are exact-closed before the shared projections parse them.
    BaseRunConfig.from_mapping(run, resolver=paths.resolver)
    BaseTrainingConfig.from_validated_task_mapping(training)
    validate_epoch_candidate_policy(config)


def _validate_metrics(metrics: ConfigMapping) -> None:
    peak = _required_number(metrics, "peak_threshold", path="metrics")
    distance = _required_number(metrics, "ball_distance_threshold", path="metrics")
    if peak < 0.0 or distance < 0.0:
        raise SemanticConfigurationError("metrics thresholds must be non-negative.")
    nms_kernel = cast(int, typed(metrics, "nms_kernel", int, path="metrics"))
    if nms_kernel <= 0 or nms_kernel % 2 == 0:
        raise SemanticConfigurationError("metrics.nms_kernel must be positive and odd.")
    _positive(
        cast(
            int,
            typed(
                metrics,
                "max_predictions_per_frame",
                int,
                path="metrics",
            ),
        ),
        path="metrics.max_predictions_per_frame",
    )
    typed(metrics, "subpixel_refine", bool, path="metrics")


def _validate_draw_style(value: object, *, path: str) -> ConfigMapping:
    draw = exact_mapping(
        value,
        path=path,
        required={
            "gt_radius",
            "pred_radius",
            "thickness",
            "gt_color_rgb",
            "pred_color_rgb",
            "text_color_rgb",
            "muted_text_color_rgb",
        },
    )
    for key in ("gt_radius", "pred_radius", "thickness"):
        _positive(cast(int, typed(draw, key, int, path=path)), path=f"{path}.{key}")
    for key in (
        "gt_color_rgb",
        "pred_color_rgb",
        "text_color_rgb",
        "muted_text_color_rgb",
    ):
        _validate_rgb(draw, key, path=path)
    return draw


def _validate_layout_style(value: object, *, path: str) -> ConfigMapping:
    layout = exact_mapping(
        value,
        path=path,
        required={
            "header_height",
            "tile_gap",
            "text_scale",
            "text_thickness",
            "background_rgb",
            "panel_label_height",
        },
    )
    for key in (
        "header_height",
        "tile_gap",
        "text_thickness",
        "panel_label_height",
    ):
        _positive(
            cast(int, typed(layout, key, int, path=path)),
            path=f"{path}.{key}",
            allow_zero=key == "tile_gap",
        )
    _positive(
        _required_number(layout, "text_scale", path=path), path=f"{path}.text_scale"
    )
    _validate_rgb(layout, "background_rgb", path=path)
    return layout


def validate_visualization(config: DictConfig) -> None:
    """Validate visualization composition and all derived paths."""
    exact_mapping(
        config,
        path="configuration",
        required={"paths", "model", "data", "metrics", "visualization", "run"},
    )
    paths = BallRuntimePaths.from_config(config)
    validate_model(config, paths=paths)
    validate_data(config, paths=paths)
    root = as_mapping(config, path="configuration")
    run = exact_mapping(
        root["run"],
        path="run",
        required={"output_dir", "device"},
    )
    typed(run, "output_dir", str, path="run")
    typed(run, "device", str, path="run")
    metrics = exact_mapping(
        root["metrics"],
        path="metrics",
        required={
            "peak_threshold",
            "ball_distance_threshold",
            "nms_kernel",
            "max_predictions_per_frame",
            "subpixel_refine",
        },
    )
    _validate_metrics(metrics)
    vis = exact_mapping(
        root["visualization"],
        path="visualization",
        required={
            "store_dir",
            "clip_id",
            "checkpoint",
            "save",
            "fps",
            "window_stride",
            "inference_batch_size",
            "peak_threshold",
            "max_frames",
            "info",
            "strict",
            "weights_only",
            "draw",
            "layout",
            "gif",
        },
    )
    for key in ("store_dir", "clip_id", "save"):
        typed(vis, key, str, path="visualization")
    fps = _required_number(vis, "fps", path="visualization")
    _positive(fps, path="visualization.fps")
    for key in ("window_stride", "inference_batch_size"):
        _positive(
            cast(int, typed(vis, key, int, path="visualization")),
            path=f"visualization.{key}",
        )
    _required_number(vis, "peak_threshold", path="visualization")
    max_frames = typed(vis, "max_frames", (int, type(None)), path="visualization")
    if max_frames is not None:
        _positive(cast(int, max_frames), path="visualization.max_frames")
    typed(vis, "info", bool, path="visualization")
    typed(vis, "strict", bool, path="visualization")
    typed(vis, "weights_only", bool, path="visualization")
    _validate_draw_style(vis["draw"], path="visualization.draw")
    _validate_layout_style(vis["layout"], path="visualization.layout")
    gif = exact_mapping(vis["gif"], path="visualization.gif", required={"loop"})
    _positive(
        cast(int, typed(gif, "loop", int, path="visualization.gif")),
        path="visualization.gif.loop",
        allow_zero=True,
    )
    paths.output(cast(str, run["output_dir"]))
    paths.data(cast(str, vis["store_dir"]))
    paths.checkpoint_input(vis, "checkpoint", path="visualization")
    paths.artifact(cast(str, vis["save"]))


def validate_preview(config: DictConfig) -> None:
    """Validate augmentation/heatmap preview boundaries."""
    exact_mapping(
        config, path="configuration", required={"paths", "model", "data", "preview"}
    )
    paths = BallRuntimePaths.from_config(config)
    validate_model(config, paths=paths)
    validate_data(config, paths=paths)
    root = as_mapping(config, path="configuration")
    preview = as_mapping(root["preview"], path="preview")
    common = {"split", "sample_indices", "max_samples", "output_dir", "draw", "layout"}
    if "ratios" in preview:
        preview = exact_mapping(preview, path="preview", required=common | {"ratios"})
        ratios = _required_sequence(
            preview, "ratios", path="preview", item_type=(float, int)
        )
        if not ratios or any(cast(float | int, value) <= 0 for value in ratios):
            raise SemanticConfigurationError(
                "preview.ratios must contain positive values."
            )
        draw = exact_mapping(
            preview["draw"],
            path="preview.draw",
            required={"gt_radius", "argmax_radius", "thickness"},
        )
        layout = exact_mapping(
            preview["layout"],
            path="preview.layout",
            required={
                "tile_gap",
                "header_height",
                "text_scale",
                "text_thickness",
                "background_rgb",
            },
        )
    else:
        preview = exact_mapping(preview, path="preview", required=common | {"seed"})
        typed(preview, "seed", int, path="preview")
        draw = exact_mapping(
            preview["draw"],
            path="preview.draw",
            required={"radius", "thickness"},
        )
        layout = exact_mapping(
            preview["layout"],
            path="preview.layout",
            required={
                "tile_gap",
                "row_gap",
                "header_height",
                "text_scale",
                "text_thickness",
                "background_rgb",
            },
        )
    split_name = cast(str, typed(preview, "split", str, path="preview"))
    if split_name not in {"train", "val", "test"}:
        raise SemanticConfigurationError("preview.split must be train, val, or test.")
    _required_sequence(preview, "sample_indices", path="preview", item_type=int)
    _positive(
        cast(int, typed(preview, "max_samples", int, path="preview")),
        path="preview.max_samples",
    )
    for key in draw:
        _positive(
            cast(int, typed(draw, key, int, path="preview.draw")),
            path=f"preview.draw.{key}",
        )
    for key in ("tile_gap", "header_height", "text_thickness"):
        _positive(
            cast(int, typed(layout, key, int, path="preview.layout")),
            path=f"preview.layout.{key}",
            allow_zero=key == "tile_gap",
        )
    if "row_gap" in layout:
        _positive(
            cast(int, typed(layout, "row_gap", int, path="preview.layout")),
            path="preview.layout.row_gap",
            allow_zero=True,
        )
    _positive(
        _required_number(layout, "text_scale", path="preview.layout"),
        path="preview.layout.text_scale",
    )
    _validate_rgb(layout, "background_rgb", path="preview.layout")
    paths.output(cast(str, typed(preview, "output_dir", str, path="preview")))


def validate_eval(config: DictConfig) -> None:
    """Validate the detailed checkpoint evaluation boundary."""
    exact_mapping(
        config,
        path="configuration",
        required={
            "paths",
            "model",
            "data",
            "loss",
            "metrics",
            "training",
            "run",
            "evaluation",
        },
    )
    paths = BallRuntimePaths.from_config(config)
    validate_model(config, paths=paths)
    validate_data(config, paths=paths)
    root = as_mapping(config, path="configuration")
    training = _validate_eval_training_mapping(root["training"])
    BaseTrainingConfig.from_validated_task_mapping(training)
    loss = exact_mapping(root["loss"], path="loss", required={"name", "gamma"})
    typed(loss, "name", str, path="loss")
    _required_number(loss, "gamma", path="loss")
    metrics = exact_mapping(
        root["metrics"],
        path="metrics",
        required={
            "peak_threshold",
            "ball_distance_threshold",
            "nms_kernel",
            "max_predictions_per_frame",
            "subpixel_refine",
        },
    )
    _validate_metrics(metrics)
    run = exact_mapping(
        root["run"],
        path="run",
        required={
            "output_dir",
            "seed",
            "gpus",
            "checkpoint_path",
            "strict",
            "weights_only",
        },
    )
    typed(run, "output_dir", str, path="run")
    typed(run, "seed", int, path="run")
    _positive(
        cast(int, typed(run, "gpus", int, path="run")),
        path="run.gpus",
        allow_zero=True,
    )
    for key in ("strict", "weights_only"):
        typed(run, key, bool, path="run")
    paths.output(cast(str, run["output_dir"]))
    paths.checkpoint_input(run, "checkpoint_path", path="run")
    DetailedEvaluationConfig.from_config(config)


def _validate_eval_training_mapping(value: object) -> ConfigMapping:
    """Exact-close the normal ball training section embedded in eval config."""
    training = exact_mapping(
        value,
        path="training",
        required={
            "trainer",
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "warmup_epochs",
            "min_lr",
            "steps_per_epoch",
            "optimizer",
            "compile",
            "matmul_precision",
            "allow_tf32",
            "checkpoint",
            "early_stopping",
            "lr_monitor",
            "qualitative_logging",
            "qualitative_rendering",
            "gan",
            "validation_candidates",
        },
    )
    validate_candidate_settings(training["validation_candidates"])
    exact_mapping(
        training["trainer"],
        path="training.trainer",
        required={
            "max_epochs",
            "gradient_clip_val",
            "deterministic",
            "precision",
            "log_every_n_steps",
            "check_val_every_n_epoch",
            "accumulate_grad_batches",
            "reload_dataloaders_every_n_epochs",
            "enable_progress_bar",
            "enable_model_summary",
            "benchmark",
        },
    )
    exact_mapping(
        training["compile"],
        path="training.compile",
        required={"enabled", "backend", "mode", "fullgraph", "dynamic"},
    )
    optimizer = exact_mapping(
        training["optimizer"],
        path="training.optimizer",
        required={"name", "betas"},
    )
    typed(optimizer, "name", str, path="training.optimizer")
    exact_mapping(
        training["checkpoint"],
        path="training.checkpoint",
        required={"enabled", "filename", "monitor", "mode", "save_top_k", "save_last"},
    )
    exact_mapping(
        training["early_stopping"],
        path="training.early_stopping",
        required={
            "enabled",
            "monitor",
            "mode",
            "patience",
            "min_delta",
            "check_on_train_epoch_end",
        },
    )
    exact_mapping(
        training["lr_monitor"],
        path="training.lr_monitor",
        required={"enabled", "interval"},
    )
    exact_mapping(
        training["qualitative_logging"],
        path="training.qualitative_logging",
        required={
            "enabled",
            "every_n_epochs",
            "num_samples",
            "selection_mode",
            "selected_indices",
        },
    )
    qualitative_rendering = exact_mapping(
        training["qualitative_rendering"],
        path="training.qualitative_rendering",
        required={"fps", "draw", "layout"},
    )
    _positive(
        _required_number(
            qualitative_rendering, "fps", path="training.qualitative_rendering"
        ),
        path="training.qualitative_rendering.fps",
    )
    _validate_draw_style(
        qualitative_rendering["draw"], path="training.qualitative_rendering.draw"
    )
    _validate_layout_style(
        qualitative_rendering["layout"], path="training.qualitative_rendering.layout"
    )
    gan = exact_mapping(
        training["gan"],
        path="training.gan",
        required={
            "enabled",
            "target_weight",
            "warmup_epochs",
            "generator_gradient_clip_val",
            "discriminator_gradient_clip_val",
            "soft_argmax_temperature",
            "transition",
            "discriminator",
        },
    )
    exact_mapping(
        gan["transition"],
        path="training.gan.transition",
        required={"start_epoch"},
    )
    discriminator = exact_mapping(
        gan["discriminator"],
        path="training.gan.discriminator",
        required={
            "name",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "ffn_dim",
            "ffn_type",
            "dropout",
            "rope_dim",
            "rope_theta",
            "max_seq_len",
            "invalid_init_std",
            "cls_init_std",
        },
    )
    for key in ("ffn_dim", "rope_dim"):
        typed(discriminator, key, int, path="training.gan.discriminator")
    _positive(
        _required_number(gan, "soft_argmax_temperature", path="training.gan"),
        path="training.gan.soft_argmax_temperature",
    )
    return training


def validate_manifest_boundary(config: DictConfig) -> None:
    """Validate the single manifest-owned evaluation authority."""
    paths = BallRuntimePaths.from_config(config)
    root = exact_mapping(
        config,
        path="configuration",
        required={"paths", "manifest_path"},
    )
    manifest_path = cast(str, typed(root, "manifest_path", str, path="configuration"))
    resolved_manifest = paths.project(manifest_path)
    if not resolved_manifest.is_file():
        raise SemanticConfigurationError(
            f"configuration.manifest_path does not name an existing file: "
            f"{resolved_manifest}."
        )


def _register() -> None:
    register_boundary_validator("ball.train", validate_training)
    register_boundary_validator("ball.visualize", validate_visualization)
    register_boundary_validator("ball.eval", validate_eval)
    register_boundary_validator("ball.evaluate_manifest", validate_manifest_boundary)
    register_boundary_validator("ball.preview", validate_preview)


_register()


__all__ = [
    "BallRuntimePaths",
    "as_mapping",
    "exact_mapping",
    "sequence",
    "typed",
    "validate_augmentation",
    "validate_data",
    "validate_model",
    "validate_training",
    "validate_visualization",
]

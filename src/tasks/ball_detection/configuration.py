"""Strict configuration and path contracts for ball-detection runtimes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import BaseRunConfig, BaseTrainingConfig
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
    invalid = not math.isfinite(value) or (
        value < 0 if allow_zero else value <= 0
    )
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
    if (
        type(value) is not str
        or not relative.strip()
        or relative != relative.strip()
    ):
        raise ConfigurationTypeError(
            f"{path}: expected non-empty trimmed str, got {type(value).__name__}."
        )
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SemanticConfigurationError(
            f"{path} must be a non-escaping relative path, got {relative!r}."
        )
    return relative


def _validate_field_types(
    mapping: ConfigMapping,
    *,
    path: str,
    fields: Mapping[str, type[object] | tuple[type[object], ...]],
) -> None:
    for key, expected in fields.items():
        typed(mapping, key, expected, path=path)


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


def _validate_trimmed_string(value: object, *, path: str) -> str:
    text = cast(str, value)
    if type(value) is not str or not text.strip() or text != text.strip():
        raise SemanticConfigurationError(
            f"{path} must be a non-empty trimmed string."
        )
    return text


def _validate_optional_trimmed_string(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    return _validate_trimmed_string(value, path=path)


def _validate_http_url(value: object, *, path: str) -> str:
    text = _validate_trimmed_string(value, path=path)
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SemanticConfigurationError(
            f"{path} must be an absolute HTTP(S) URL."
        )
    return text


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
class BallYoutubePathContract:
    """Resolved role-tagged inputs shared by both Ball YouTube workflows."""

    paths: BallRuntimePaths
    workflow_root: Path
    download_archive: Path | None
    local_videos: Mapping[str, Path]

    @classmethod
    def from_config(cls, config: object) -> BallYoutubePathContract:
        """Resolve archive/manual-video fragments once and reject legacy paths."""
        paths = BallRuntimePaths.from_config(config)
        root = as_mapping(config, path="configuration")
        workflow = as_mapping(
            typed(root, "workflow", (dict, DictConfig), path="configuration"),
            path="workflow",
        )
        workflow_fragment = _validate_relative_child(
            typed(workflow, "root", str, path="workflow"),
            path="workflow.root",
        )
        workflow_root = paths.resolver.resolve(PathRole.DATA, workflow_fragment)

        archive: Path | None = None
        if "download" in workflow:
            download = as_mapping(workflow["download"], path="workflow.download")
            raw_archive = typed(
                download,
                "download_archive",
                (str, type(None)),
                path="workflow.download",
            )
            if raw_archive is not None:
                archive_fragment = _validate_relative_child(
                    raw_archive,
                    path="workflow.download.download_archive",
                )
                archive = paths.resolver.resolve(
                    PathRole.DATA,
                    workflow_fragment,
                    archive_fragment,
                )

        local_videos: dict[str, Path] = {}
        if "discovery" in workflow:
            raw_sources = typed(workflow, "sources", (list, tuple), path="workflow")
            for index, raw_source in enumerate(
                sequence(raw_sources, path="workflow.sources")
            ):
                source_path = f"workflow.sources[{index}]"
                source = as_mapping(raw_source, path=source_path)
                video_id = cast(str, typed(source, "video_id", str, path=source_path))
                raw_local_video = typed(
                    source,
                    "local_video",
                    (dict, DictConfig, type(None)),
                    path=source_path,
                )
                if raw_local_video is None:
                    continue
                if video_id in local_videos:
                    raise SemanticConfigurationError(
                        f"{source_path}.video_id duplicates {video_id!r}."
                    )
                declaration = exact_mapping(
                    raw_local_video,
                    path=f"{source_path}.local_video",
                    required={"role", "path"},
                )
                role_value = cast(
                    str,
                    typed(
                        declaration,
                        "role",
                        str,
                        path=f"{source_path}.local_video",
                    ),
                )
                try:
                    role = PathRole(role_value)
                except ValueError as error:
                    raise SemanticConfigurationError(
                        f"{source_path}.local_video.role is unsupported: "
                        f"{role_value!r}."
                    ) from error
                if role not in {PathRole.DATA, PathRole.EXTERNAL_ASSET}:
                    raise SemanticConfigurationError(
                        f"{source_path}.local_video.role must be 'data' or "
                        "'external_asset'."
                    )
                local_fragment = _validate_relative_child(
                    typed(
                        declaration,
                        "path",
                        str,
                        path=f"{source_path}.local_video",
                    ),
                    path=f"{source_path}.local_video.path",
                )
                local_path = paths.resolver.resolve(role, local_fragment)
                if not local_path.is_file():
                    raise SemanticConfigurationError(
                        f"{source_path}.local_video does not name an existing file: "
                        f"{local_path}."
                    )
                local_videos[video_id] = local_path
        return cls(
            paths=paths,
            workflow_root=workflow_root,
            download_archive=archive,
            local_videos=MappingProxyType(local_videos),
        )

    def local_video_for(self, video_id: str) -> Path | None:
        """Return the explicitly configured manual DATA input for one source."""
        return self.local_videos.get(video_id)


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
        _validate_relative_child(
            output_json_name, path="evaluation.output_json_name"
        )
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
        if ffn_type not in {"swiglu", "mlp"}:
            raise SemanticConfigurationError(
                "model.decoder.ffn_type must be swiglu or mlp."
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
            paths.external_asset(checkpoint_path)
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
        "data_dir",
        "batch_size",
        "num_workers",
        "pin_memory",
        "image_size",
        "heatmap_size",
        "sigma_ratio",
        "max_instances",
        "augmentation",
    }
    if source in {"tracknet", "youtube"}:
        required = common | {"split", "sample_stride"}
    elif source == "web":
        required = common | {"sources", "sampling"}
    elif source == "mixed_tracknet":
        required = common | {
            "split",
            "sample_stride",
            "synthetic",
            "synthetic_per_batch",
            "synthetic_batch_period",
            "steps_per_epoch",
            "sampling_seed",
        }
    elif source == "staged":
        required = {
            "source",
            "t_max",
            "t_distribution",
            "t1_prob",
            "val_num_frames",
            "effective_batch_size",
            "batch_size_by_t",
            "val_batch_size",
            "num_workers",
            "pin_memory",
            "seed",
            "image_size",
            "heatmap_size",
            "sigma_ratio",
            "max_instances",
            "sources",
            "augmentation",
        }
    else:
        raise SemanticConfigurationError(f"data.source: unsupported value {source!r}.")
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
    if source != "staged":
        typed(data, "data_dir", str, path="data")
        _positive(
            cast(int, typed(data, "batch_size", int, path="data")),
            path="data.batch_size",
        )
    if "split" in data:
        split = exact_mapping(
            data["split"],
            path="data.split",
            required={"root_role", "train_file", "val_file", "test_file"},
        )
        _validate_split_mapping(split, path="data.split", paths=paths)
    if source in {"tracknet", "youtube"}:
        _positive(
            cast(int, typed(data, "sample_stride", int, path="data")),
            path="data.sample_stride",
        )
    if source == "web":
        sources = typed(data, "sources", (str, list, tuple), path="data")
        if isinstance(sources, (list, tuple)):
            typed_sequence(sources, path="data.sources", item_type=str)
        sampling = exact_mapping(
            data["sampling"],
            path="data.sampling",
            required={"mode", "seed", "train_negative_fraction", "temporal"},
        )
        mode = typed(sampling, "mode", str, path="data.sampling")
        if mode not in {"static", "temporal"}:
            raise SemanticConfigurationError(
                "data.sampling.mode must be 'static' or 'temporal'."
            )
        exact_mapping(
            sampling["temporal"],
            path="data.sampling.temporal",
            required={"frame_step", "sample_stride", "max_frame_gap"},
        )
        typed(sampling, "seed", int, path="data.sampling")
        negative_fraction = _optional_number(
            sampling, "train_negative_fraction", path="data.sampling"
        )
        if negative_fraction is not None and not 0.0 <= negative_fraction < 1.0:
            raise SemanticConfigurationError(
                "data.sampling.train_negative_fraction must be null or in [0, 1)."
            )
        temporal = as_mapping(sampling["temporal"], path="data.sampling.temporal")
        for key in ("frame_step", "sample_stride"):
            _positive(
                cast(int, typed(temporal, key, int, path="data.sampling.temporal")),
                path=f"data.sampling.temporal.{key}",
            )
        max_gap = typed(
            temporal,
            "max_frame_gap",
            (int, type(None)),
            path="data.sampling.temporal",
        )
        if max_gap is not None:
            _positive(cast(int, max_gap), path="data.sampling.temporal.max_frame_gap")
    if source == "mixed_tracknet":
        for key in (
            "sample_stride",
            "synthetic_per_batch",
            "synthetic_batch_period",
            "steps_per_epoch",
            "sampling_seed",
        ):
            value = cast(int, typed(data, key, int, path="data"))
            _positive(
                value,
                path=f"data.{key}",
                allow_zero=key == "synthetic_per_batch",
            )
        synthetic = exact_mapping(
            data["synthetic"],
            path="data.synthetic",
            required={"data_dir", "split", "sample_stride"},
        )
        typed(synthetic, "data_dir", str, path="data.synthetic")
        _positive(
            cast(
                int,
                typed(synthetic, "sample_stride", int, path="data.synthetic"),
            ),
            path="data.synthetic.sample_stride",
        )
        synthetic_split = exact_mapping(
            synthetic["split"],
            path="data.synthetic.split",
            required={"root_role", "train_file"},
        )
        _validate_split_mapping(
            synthetic_split,
            path="data.synthetic.split",
            paths=paths,
            file_keys=("train_file",),
        )
        if paths is not None:
            paths.data(cast(str, synthetic["data_dir"]))
    if source == "staged":
        for key in (
            "t_max",
            "val_num_frames",
            "effective_batch_size",
            "val_batch_size",
            "seed",
        ):
            _positive(
                cast(int, typed(data, key, int, path="data")),
                path=f"data.{key}",
            )
        t_distribution = cast(str, typed(data, "t_distribution", str, path="data"))
        if t_distribution not in {"variable", "fixed"}:
            raise SemanticConfigurationError(
                "data.t_distribution must be variable or fixed."
            )
        t1_prob = _required_number(data, "t1_prob", path="data")
        if not 0.0 < t1_prob <= 1.0:
            raise SemanticConfigurationError("data.t1_prob must be in (0, 1].")
        batch_plan = data["batch_size_by_t"]
        if not isinstance(batch_plan, Mapping) or not batch_plan:
            raise ConfigurationTypeError(
                "data.batch_size_by_t must be a non-empty mapping."
            )
        for raw_t, raw_batch in batch_plan.items():
            if type(raw_t) is not int or type(raw_batch) is not int:
                raise ConfigurationTypeError(
                    "data.batch_size_by_t keys and values must be integers."
                )
            _positive(raw_t, path=f"data.batch_size_by_t.{raw_t}")
            _positive(raw_batch, path=f"data.batch_size_by_t.{raw_t}")
        sources = exact_mapping(
            data["sources"], path="data.sources", required={"tracknet", "web"}
        )
        tracknet = exact_mapping(
            sources["tracknet"],
            path="data.sources.tracknet",
            required={"enabled", "splits", "data_dir", "sample_stride", "split"},
        )
        exact_mapping(
            tracknet["split"],
            path="data.sources.tracknet.split",
            required={"root_role", "train_file", "val_file", "test_file"},
        )
        typed(tracknet, "enabled", bool, path="data.sources.tracknet")
        typed(tracknet, "data_dir", str, path="data.sources.tracknet")
        _required_sequence(
            tracknet,
            "splits",
            path="data.sources.tracknet",
            item_type=str,
        )
        _positive(
            cast(
                int,
                typed(tracknet, "sample_stride", int, path="data.sources.tracknet"),
            ),
            path="data.sources.tracknet.sample_stride",
        )
        _validate_split_mapping(
            as_mapping(tracknet["split"], path="data.sources.tracknet.split"),
            path="data.sources.tracknet.split",
            paths=paths,
        )
        web = exact_mapping(
            sources["web"],
            path="data.sources.web",
            required={"enabled", "splits", "data_dir", "sources", "sampling"},
        )
        web_sampling = exact_mapping(
            web["sampling"],
            path="data.sources.web.sampling",
            required={"mode", "seed", "train_negative_fraction", "temporal"},
        )
        exact_mapping(
            web_sampling["temporal"],
            path="data.sources.web.sampling.temporal",
            required={"frame_step", "sample_stride", "max_frame_gap"},
        )
        typed(web, "enabled", bool, path="data.sources.web")
        typed(web, "data_dir", str, path="data.sources.web")
        _required_sequence(web, "splits", path="data.sources.web", item_type=str)
        web_sources = typed(web, "sources", (str, list, tuple), path="data.sources.web")
        if isinstance(web_sources, (list, tuple)):
            typed_sequence(web_sources, path="data.sources.web.sources", item_type=str)
        web_mode = cast(
            str, typed(web_sampling, "mode", str, path="data.sources.web.sampling")
        )
        if web_mode not in {"static", "temporal"}:
            raise SemanticConfigurationError(
                "data.sources.web.sampling.mode must be static or temporal."
            )
        typed(web_sampling, "seed", int, path="data.sources.web.sampling")
        _optional_number(
            web_sampling,
            "train_negative_fraction",
            path="data.sources.web.sampling",
        )
        web_temporal = as_mapping(
            web_sampling["temporal"], path="data.sources.web.sampling.temporal"
        )
        for key in ("frame_step", "sample_stride"):
            _positive(
                cast(
                    int,
                    typed(
                        web_temporal,
                        key,
                        int,
                        path="data.sources.web.sampling.temporal",
                    ),
                ),
                path=f"data.sources.web.sampling.temporal.{key}",
            )
        typed(
            web_temporal,
            "max_frame_gap",
            (int, type(None)),
            path="data.sources.web.sampling.temporal",
        )
        if paths is not None:
            paths.data(cast(str, tracknet["data_dir"]))
            paths.data(cast(str, web["data_dir"]))
    if paths is not None and "data_dir" in data:
        paths.data(cast(str, typed(data, "data_dir", str, path="data")))
    return data


def _validate_split_mapping(
    split: ConfigMapping,
    *,
    path: str,
    paths: BallRuntimePaths | None,
    file_keys: tuple[str, ...] = ("train_file", "val_file", "test_file"),
) -> None:
    root_role = cast(str, typed(split, "root_role", str, path=path))
    if root_role not in {"project", "data"}:
        raise SemanticConfigurationError(
            f"{path}.root_role must be 'project' or 'data'."
        )
    for key in file_keys:
        relative = cast(str, typed(split, key, str, path=path))
        if paths is not None:
            if root_role == "project":
                paths.project(relative)
            else:
                paths.data(relative)


def validate_training(config: DictConfig) -> None:
    """Validate a complete normal/staged training composition."""
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
        "matmul_precision",
        "allow_tf32",
        "checkpoint",
        "early_stopping",
        "lr_monitor",
        "qualitative_logging",
        "qualitative_rendering",
        "gan",
    }
    if "staged" in training:
        training_fields.add("staged")
    training = exact_mapping(training, path="training", required=training_fields)
    optimizer = exact_mapping(
        training["optimizer"], path="training.optimizer", required={"name", "betas"}
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
    gan_enabled = cast(
        bool, typed(gan, "enabled", bool, path="training.gan")
    )
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
    if "staged" in training:
        staged = exact_mapping(
            training["staged"],
            path="training.staged",
            required={
                "effective_batch_size",
                "gradient_clip_val",
                "calibration_token_budget",
                "calibration_safety",
            },
        )
        for key in ("effective_batch_size", "calibration_token_budget"):
            _positive(
                cast(int, typed(staged, key, int, path="training.staged")),
                path=f"training.staged.{key}",
            )
        typed(
            staged,
            "gradient_clip_val",
            (float, int, type(None)),
            path="training.staged",
        )
        safety = _required_number(staged, "calibration_safety", path="training.staged")
        if not 0.0 < safety <= 1.0:
            raise SemanticConfigurationError(
                "training.staged.calibration_safety must be in (0, 1]."
            )

    # Task-owned maps are exact-closed before the shared projections parse them.
    BaseRunConfig.from_mapping(run, resolver=paths.resolver)
    BaseTrainingConfig.from_validated_task_mapping(training)


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
        required={"output_dir", "device", "allow_device_fallback"},
    )
    typed(run, "output_dir", str, path="run")
    typed(run, "device", str, path="run")
    typed(run, "allow_device_fallback", bool, path="run")
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
            "clip_dir",
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
    for key in ("clip_dir", "checkpoint", "save"):
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
    paths.data(cast(str, vis["clip_dir"]))
    paths.checkpoint(cast(str, vis["checkpoint"]))
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
            "allow_device_fallback",
        },
    )
    for key in ("output_dir", "checkpoint_path"):
        typed(run, key, str, path="run")
    typed(run, "seed", int, path="run")
    _positive(
        cast(int, typed(run, "gpus", int, path="run")),
        path="run.gpus",
        allow_zero=True,
    )
    for key in ("strict", "weights_only", "allow_device_fallback"):
        typed(run, key, bool, path="run")
    paths.output(cast(str, run["output_dir"]))
    paths.checkpoint(cast(str, run["checkpoint_path"]))
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
            "matmul_precision",
            "allow_tf32",
            "checkpoint",
            "early_stopping",
            "lr_monitor",
            "qualitative_logging",
            "qualitative_rendering",
            "gan",
        },
    )
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


def validate_web_tool(config: DictConfig) -> None:
    """Validate web conversion/analysis roots before filesystem mutation."""
    paths = BallRuntimePaths.from_config(config)
    root = as_mapping(config, path="configuration")
    if "convert" in root:
        exact_mapping(root, path="configuration", required={"paths", "convert"})
        section = exact_mapping(
            root["convert"],
            path="convert",
            required={
                "web_root",
                "output_dir",
                "overwrite",
                "jpeg_quality",
                "shard_size_bytes",
                "val_ratio",
                "test_ratio",
                "split_seed",
                "kaggle_corner_frac",
                "max_bbox_side_ratio",
                "limit_per_source",
                "sources",
            },
        )
        paths.data(cast(str, typed(section, "web_root", str, path="convert")))
        paths.data(cast(str, typed(section, "output_dir", str, path="convert")))
        typed(section, "overwrite", bool, path="convert")
        for key in ("shard_size_bytes",):
            _positive(
                cast(int, typed(section, key, int, path="convert")),
                path=f"convert.{key}",
            )
        jpeg_quality = cast(int, typed(section, "jpeg_quality", int, path="convert"))
        if not 1 <= jpeg_quality <= 100:
            raise SemanticConfigurationError(
                "convert.jpeg_quality must be in [1, 100]."
            )
        val_ratio = _required_number(section, "val_ratio", path="convert")
        test_ratio = _required_number(section, "test_ratio", path="convert")
        if (
            not math.isfinite(val_ratio)
            or not math.isfinite(test_ratio)
            or val_ratio < 0
            or test_ratio < 0
            or val_ratio + test_ratio >= 1
        ):
            raise SemanticConfigurationError(
                "convert.val_ratio and convert.test_ratio must be finite, "
                "non-negative, and sum to less than 1."
            )
        corner_fraction = _required_number(
            section, "kaggle_corner_frac", path="convert"
        )
        if not math.isfinite(corner_fraction) or not 0 <= corner_fraction <= 1:
            raise SemanticConfigurationError(
                "convert.kaggle_corner_frac must be in [0, 1]."
            )
        max_bbox_side_ratio = typed(
            section,
            "max_bbox_side_ratio",
            (float, int, type(None)),
            path="convert",
        )
        if max_bbox_side_ratio is not None and (
            not math.isfinite(cast(float | int, max_bbox_side_ratio))
            or not 0 < cast(float | int, max_bbox_side_ratio) <= 1
        ):
            raise SemanticConfigurationError(
                "convert.max_bbox_side_ratio must be null or in (0, 1]."
            )
        typed(section, "split_seed", int, path="convert")
        _positive(
            cast(int, typed(section, "limit_per_source", int, path="convert")),
            path="convert.limit_per_source",
            allow_zero=True,
        )
        sources = exact_mapping(
            section["sources"],
            path="convert.sources",
            required={"roboflow", "racketvision", "kaggle", "ball_yolo"},
        )
        for key in sources:
            typed(sources, key, bool, path="convert.sources")
        if not any(cast(bool, enabled) for enabled in sources.values()):
            raise SemanticConfigurationError(
                "convert.sources must enable at least one source."
            )
    else:
        exact_mapping(root, path="configuration", required={"paths", "analyze"})
        section = exact_mapping(
            root["analyze"],
            path="analyze",
            required={
                "web_root",
                "output_dir",
                "sources",
                "sweep_thresholds",
                "export_bin_edges",
                "samples_per_bin",
                "seed",
                "jpeg_quality",
            },
        )
        paths.data(cast(str, typed(section, "web_root", str, path="analyze")))
        paths.output(cast(str, typed(section, "output_dir", str, path="analyze")))
        sources = exact_mapping(
            section["sources"],
            path="analyze.sources",
            required={"roboflow", "ball_yolo"},
        )
        for key in sources:
            typed(sources, key, bool, path="analyze.sources")
        if not any(cast(bool, enabled) for enabled in sources.values()):
            raise SemanticConfigurationError(
                "analyze.sources must enable at least one source."
            )
        for key in ("sweep_thresholds", "export_bin_edges"):
            values = _required_sequence(
                section, key, path="analyze", item_type=(float, int)
            )
            numeric_values = tuple(cast(float | int, item) for item in values)
            if (
                len(numeric_values) < 2
                or any(
                    not math.isfinite(item) or item < 0 for item in numeric_values
                )
                or any(
                    left >= right
                    for left, right in zip(
                        numeric_values, numeric_values[1:], strict=False
                    )
                )
            ):
                raise SemanticConfigurationError(
                    f"analyze.{key} must contain at least two strictly increasing "
                    "finite non-negative numbers."
                )
        _positive(
            cast(int, typed(section, "samples_per_bin", int, path="analyze")),
            path="analyze.samples_per_bin",
        )
        jpeg_quality = cast(int, typed(section, "jpeg_quality", int, path="analyze"))
        if not 1 <= jpeg_quality <= 100:
            raise SemanticConfigurationError(
                "analyze.jpeg_quality must be in [1, 100]."
            )
        typed(section, "seed", int, path="analyze")


def _validate_youtube_paths(
    value: object,
    *,
    path: str,
    keys: set[str],
    workflow_root: str,
    paths: BallRuntimePaths,
) -> ConfigMapping:
    mapping = exact_mapping(value, path=path, required=keys)
    for key in mapping:
        child = _validate_relative_child(mapping[key], path=f"{path}.{key}")
        paths.resolver.resolve(PathRole.DATA, workflow_root, child)
    return mapping


def _validate_download(value: object, *, path: str, require_av1: bool) -> ConfigMapping:
    required = {
        "enabled",
        "format",
        "merge_output_format",
        "js_runtimes",
        "remote_components",
        "download_archive",
        "overwrite",
        "extra_args",
    }
    if require_av1:
        required |= {"require_av1", "strict_format"}
    mapping = exact_mapping(value, path=path, required=required)
    fields: dict[str, type[object] | tuple[type[object], ...]] = {
        "enabled": bool,
        "format": str,
        "merge_output_format": str,
        "js_runtimes": (str, type(None)),
        "remote_components": (str, type(None)),
        "download_archive": (str, type(None)),
        "overwrite": bool,
        "extra_args": (list, tuple),
    }
    if require_av1:
        fields["require_av1"] = bool
        fields["strict_format"] = str
    _validate_field_types(mapping, path=path, fields=fields)
    enabled = cast(bool, mapping["enabled"])
    extra_args = _validate_string_sequence(
        mapping["extra_args"], path=f"{path}.extra_args", allow_empty=True
    )
    if not enabled:
        return mapping

    _validate_trimmed_string(mapping["merge_output_format"], path=f"{path}.merge_output_format")
    if require_av1 and cast(bool, mapping["require_av1"]):
        _validate_trimmed_string(mapping["strict_format"], path=f"{path}.strict_format")
    else:
        _validate_trimmed_string(mapping["format"], path=f"{path}.format")
    for key in ("js_runtimes", "remote_components"):
        _validate_optional_trimmed_string(mapping[key], path=f"{path}.{key}")
    reserved_arguments = {
        "-f",
        "--format",
        "-o",
        "--output",
        "--merge-output-format",
        "--download-archive",
        "--force-overwrites",
        "--no-overwrites",
        "--js-runtimes",
        "--remote-components",
    }
    conflicting_arguments = tuple(
        argument
        for argument in extra_args
        if argument.split("=", 1)[0] in reserved_arguments
    )
    if conflicting_arguments:
        raise SemanticConfigurationError(
            f"{path}.extra_args must not redefine typed download settings: "
            + ", ".join(conflicting_arguments)
            + "."
        )
    return mapping


def _validate_transcode(
    value: object, *, path: str, fallback_on_decode_error: bool
) -> ConfigMapping:
    required = {
        "enabled",
        "ffmpeg_binary",
        "encoder",
        "hwaccel",
        "hwaccel_output_format",
        "preset",
        "tune",
        "rate_control",
        "cq",
        "bitrate",
        "maxrate",
        "bufsize",
        "profile",
        "pix_fmt",
        "crf",
        "overwrite",
    }
    if fallback_on_decode_error:
        required.add("fallback_on_decode_error")
    mapping = exact_mapping(value, path=path, required=required)
    fields: dict[str, type[object] | tuple[type[object], ...]] = {
        "enabled": bool,
        "ffmpeg_binary": str,
        "encoder": str,
        "hwaccel": (str, type(None)),
        "hwaccel_output_format": (str, type(None)),
        "preset": str,
        "tune": (str, type(None)),
        "rate_control": (str, type(None)),
        "cq": (float, int, type(None)),
        "bitrate": (str, type(None)),
        "maxrate": (str, type(None)),
        "bufsize": (str, type(None)),
        "profile": (str, type(None)),
        "pix_fmt": str,
        "crf": (float, int, type(None)),
        "overwrite": bool,
    }
    if fallback_on_decode_error:
        fields["fallback_on_decode_error"] = bool
    _validate_field_types(mapping, path=path, fields=fields)

    enabled = cast(bool, mapping["enabled"])
    fallback_enabled = fallback_on_decode_error and cast(
        bool, mapping["fallback_on_decode_error"]
    )
    if not enabled and not fallback_enabled:
        return mapping

    for key in ("ffmpeg_binary", "encoder", "preset", "pix_fmt"):
        _validate_trimmed_string(mapping[key], path=f"{path}.{key}")
    ffmpeg_binary = cast(str, mapping["ffmpeg_binary"])
    if Path(ffmpeg_binary).name != ffmpeg_binary:
        raise SemanticConfigurationError(
            f"{path}.ffmpeg_binary must be an executable name, not a path."
        )
    encoder = cast(str, mapping["encoder"])
    if encoder not in {"libx264", "h264_nvenc", "avc_nvenc"}:
        raise SemanticConfigurationError(
            f"{path}.encoder must be 'libx264', 'h264_nvenc', or 'avc_nvenc'."
        )
    for key in (
        "hwaccel",
        "hwaccel_output_format",
        "tune",
        "rate_control",
        "bitrate",
        "maxrate",
        "bufsize",
        "profile",
    ):
        _validate_optional_trimmed_string(mapping[key], path=f"{path}.{key}")
    if mapping["hwaccel_output_format"] is not None and mapping["hwaccel"] is None:
        raise SemanticConfigurationError(
            f"{path}.hwaccel_output_format requires {path}.hwaccel."
        )

    cq = None if mapping["cq"] is None else cast(float | int, mapping["cq"])
    crf = None if mapping["crf"] is None else cast(float | int, mapping["crf"])
    for key, number in (("cq", cq), ("crf", crf)):
        if number is not None and (not math.isfinite(number) or not 0 <= number <= 51):
            raise SemanticConfigurationError(f"{path}.{key} must be in [0, 51].")

    if encoder == "libx264":
        if crf is None:
            raise SemanticConfigurationError(
                f"{path}.crf is required for encoder='libx264'."
            )
        ignored = (
            "tune",
            "rate_control",
            "cq",
            "bitrate",
            "maxrate",
            "bufsize",
            "profile",
        )
        configured_ignored = [key for key in ignored if mapping[key] is not None]
        if configured_ignored:
            raise SemanticConfigurationError(
                f"{path}: encoder='libx264' does not consume "
                + ", ".join(f"{path}.{key}" for key in configured_ignored)
                + "."
            )
    else:
        if crf is not None:
            raise SemanticConfigurationError(
                f"{path}.crf must be null for NVENC encoders."
            )
        for key in ("tune", "rate_control", "cq", "bitrate"):
            if mapping[key] is None:
                raise SemanticConfigurationError(
                    f"{path}.{key} is required for NVENC encoders."
                )
    return mapping


def validate_youtube_boundary(config: DictConfig) -> None:
    """Validate YouTube data-generation roots and exact top-level sections."""
    paths = BallRuntimePaths.from_config(config)
    root = as_mapping(config, path="configuration")
    exact_mapping(root, path="configuration", required={"paths", "workflow"})
    workflow = as_mapping(root["workflow"], path="workflow")
    workflow_root = cast(str, typed(workflow, "root", str, path="workflow"))
    paths.data(workflow_root)
    BallYoutubePathContract.from_config(config)
    if "mode" in workflow:
        workflow = exact_mapping(
            workflow,
            path="workflow",
            required={"root", "video_id", "mode", "paths", "select", "prediction"},
        )
        workflow_paths = exact_mapping(
            workflow["paths"],
            path="workflow.paths",
            required={"frames_dir", "raw_dir", "staging_dir"},
        )
        for key in workflow_paths:
            child = _validate_relative_child(
                workflow_paths[key], path=f"workflow.paths.{key}"
            )
            paths.resolver.resolve(PathRole.DATA, workflow_root, child)
        raw_video_id = typed(
            workflow, "video_id", (str, type(None)), path="workflow"
        )
        if raw_video_id is None:
            raise SemanticConfigurationError(
                "workflow.video_id is required for candidate selection or prediction."
            )
        _validate_trimmed_string(raw_video_id, path="workflow.video_id")
        mode = cast(str, typed(workflow, "mode", str, path="workflow"))
        if mode not in {"select", "predict"}:
            raise SemanticConfigurationError(
                "workflow.mode must be 'select' or 'predict'."
            )
        select = exact_mapping(
            workflow["select"],
            path="workflow.select",
            required={
                "resume",
                "start_index",
                "window_name",
                "max_display_width",
                "max_display_height",
                "min_frames",
                "copy_mode",
                "overwrite",
                "skip_small",
                "skip_medium",
                "skip_large",
            },
        )
        _validate_field_types(
            select,
            path="workflow.select",
            fields={
                "resume": bool,
                "start_index": (int, type(None)),
                "window_name": str,
                "max_display_width": int,
                "max_display_height": int,
                "min_frames": int,
                "copy_mode": str,
                "overwrite": bool,
                "skip_small": int,
                "skip_medium": int,
                "skip_large": int,
            },
        )
        for key in (
            "max_display_width",
            "max_display_height",
            "min_frames",
            "skip_small",
            "skip_medium",
            "skip_large",
        ):
            _positive(cast(int, select[key]), path=f"workflow.select.{key}")
        if mode == "select":
            start_index = select["start_index"]
            if start_index is not None:
                _positive(
                    cast(int, start_index),
                    path="workflow.select.start_index",
                    allow_zero=True,
                )
            _validate_trimmed_string(
                select["window_name"], path="workflow.select.window_name"
            )
            if select["copy_mode"] not in {"hardlink", "copy"}:
                raise SemanticConfigurationError(
                    "workflow.select.copy_mode must be 'hardlink' or 'copy'."
                )
            skips = tuple(
                cast(int, select[key])
                for key in ("skip_small", "skip_medium", "skip_large")
            )
            if not skips[0] <= skips[1] <= skips[2] or skips[2] > 50:
                raise SemanticConfigurationError(
                    "workflow.select skip sizes must satisfy "
                    "0 < skip_small <= skip_medium <= skip_large <= 50."
                )
        prediction = exact_mapping(
            workflow["prediction"],
            path="workflow.prediction",
            required={
                "checkpoint",
                "device",
                "allow_device_fallback",
                "sequence_length",
                "window_stride",
                "batch_size",
                "image_size",
                "normalize_imagenet",
                "imagenet_mean",
                "imagenet_std",
                "peak_threshold",
                "nms_kernel",
                "max_candidates_per_frame",
                "aggregation",
                "overwrite",
                "strict",
                "weights_only",
                "subpixel_refine",
            },
        )
        _validate_field_types(
            prediction,
            path="workflow.prediction",
            fields={
                "checkpoint": str,
                "device": str,
                "allow_device_fallback": bool,
                "sequence_length": int,
                "window_stride": int,
                "batch_size": int,
                "image_size": (list, tuple),
                "normalize_imagenet": bool,
                "imagenet_mean": (list, tuple),
                "imagenet_std": (list, tuple),
                "peak_threshold": (float, int),
                "nms_kernel": int,
                "max_candidates_per_frame": int,
                "aggregation": str,
                "overwrite": bool,
                "strict": bool,
                "weights_only": bool,
                "subpixel_refine": bool,
            },
        )
        image_size = _required_sequence(
            prediction,
            "image_size",
            path="workflow.prediction",
            item_type=int,
            length=2,
        )
        if any(cast(int, value) <= 0 for value in image_size):
            raise SemanticConfigurationError(
                "workflow.prediction.image_size must contain positive integers."
            )
        normalization: dict[str, tuple[object, ...]] = {}
        for key in ("imagenet_mean", "imagenet_std"):
            normalization[key] = _required_sequence(
                prediction,
                key,
                path="workflow.prediction",
                item_type=(float, int),
                length=3,
            )
        for key in (
            "sequence_length",
            "window_stride",
            "batch_size",
            "nms_kernel",
            "max_candidates_per_frame",
        ):
            _positive(cast(int, prediction[key]), path=f"workflow.prediction.{key}")
        if mode == "predict":
            if cast(int, prediction["window_stride"]) > cast(
                int, prediction["sequence_length"]
            ):
                raise SemanticConfigurationError(
                    "workflow.prediction.window_stride must not exceed "
                    "workflow.prediction.sequence_length."
                )
            if cast(int, prediction["nms_kernel"]) % 2 == 0:
                raise SemanticConfigurationError(
                    "workflow.prediction.nms_kernel must be odd."
                )
            if prediction["aggregation"] not in {"mean_heatmap", "max_heatmap"}:
                raise SemanticConfigurationError(
                    "workflow.prediction.aggregation must be 'mean_heatmap' or "
                    "'max_heatmap'."
                )
            peak_threshold = cast(float | int, prediction["peak_threshold"])
            if not math.isfinite(peak_threshold) or not 0 <= peak_threshold <= 1:
                raise SemanticConfigurationError(
                    "workflow.prediction.peak_threshold must be in [0, 1]."
                )
            if cast(bool, prediction["normalize_imagenet"]):
                std = normalization["imagenet_std"]
                if any(
                    not math.isfinite(cast(float | int, value))
                    or cast(float | int, value) <= 0
                    for value in std
                ):
                    raise SemanticConfigurationError(
                        "workflow.prediction.imagenet_std must contain positive "
                        "finite numbers."
                    )
            _validate_trimmed_string(
                prediction["device"], path="workflow.prediction.device"
            )
        paths.checkpoint(cast(str, prediction["checkpoint"]))
        return
    if "discovery" not in workflow:
        workflow = exact_mapping(
            workflow,
            path="workflow",
            required={"root", "sources", "paths", "download", "transcode", "frames"},
        )
        _validate_youtube_paths(
            workflow["paths"],
            path="workflow.paths",
            keys={
                "videos_dir",
                "av1_dir",
                "h264_dir",
                "frames_dir",
                "raw_dir",
                "manifests_dir",
            },
            workflow_root=workflow_root,
            paths=paths,
        )
        sources = typed(workflow, "sources", (list, tuple), path="workflow")
        if not sequence(sources, path="workflow.sources"):
            raise SemanticConfigurationError("workflow.sources must not be empty.")
        source_ids: set[str] = set()
        for index, source in enumerate(sequence(sources, path="workflow.sources")):
            source_mapping = exact_mapping(
                source,
                path=f"workflow.sources[{index}]",
                required={"source_id", "url", "split"},
            )
            _validate_field_types(
                source_mapping,
                path=f"workflow.sources[{index}]",
                fields={"source_id": str, "url": str, "split": str},
            )
            source_id = _validate_trimmed_string(
                source_mapping["source_id"],
                path=f"workflow.sources[{index}].source_id",
            )
            _validate_http_url(
                source_mapping["url"], path=f"workflow.sources[{index}].url"
            )
            _validate_trimmed_string(
                source_mapping["split"], path=f"workflow.sources[{index}].split"
            )
            if source_id in source_ids:
                raise SemanticConfigurationError(
                    f"workflow.sources[{index}].source_id duplicates {source_id!r}."
                )
            source_ids.add(source_id)
            if source_mapping["split"] not in {"train", "val"}:
                raise SemanticConfigurationError(
                    f"workflow.sources[{index}].split must be 'train' or 'val'."
                )
        _validate_download(
            workflow["download"],
            path="workflow.download",
            require_av1=True,
        )
        _validate_transcode(
            workflow["transcode"],
            path="workflow.transcode",
            fallback_on_decode_error=False,
        )
        frames = exact_mapping(
            workflow["frames"],
            path="workflow.frames",
            required={
                "enabled",
                "output_ext",
                "jpeg_quality",
                "max_frames_per_video",
                "overwrite",
            },
        )
        _validate_field_types(
            frames,
            path="workflow.frames",
            fields={
                "enabled": bool,
                "output_ext": str,
                "jpeg_quality": int,
                "max_frames_per_video": (int, type(None)),
                "overwrite": bool,
            },
        )
        frames_enabled = cast(bool, frames["enabled"])
        max_frames = frames["max_frames_per_video"]
        if max_frames is not None:
            _positive(
                cast(int, max_frames),
                path="workflow.frames.max_frames_per_video",
                allow_zero=not frames_enabled,
            )
        if frames_enabled:
            if frames["output_ext"] not in {"jpg", "jpeg", "png"}:
                raise SemanticConfigurationError(
                    "workflow.frames.output_ext must be 'jpg', 'jpeg', or 'png'."
                )
            jpeg_quality = cast(int, frames["jpeg_quality"])
            if not 1 <= jpeg_quality <= 100:
                raise SemanticConfigurationError(
                    "workflow.frames.jpeg_quality must be in [1, 100]."
                )
        return
    workflow = exact_mapping(
        workflow,
        path="workflow",
        required={
            "root",
            "sources",
            "discovery",
            "paths",
            "processing",
            "storage",
            "download",
            "transcode",
            "frames",
            "gate",
        },
    )
    _validate_youtube_paths(
        workflow["paths"],
        path="workflow.paths",
        keys={
            "videos_dir",
            "source_dir",
            "h264_dir",
            "sampled_dir",
            "images_dir",
            "manifests_dir",
        },
        workflow_root=workflow_root,
        paths=paths,
    )
    sources = typed(workflow, "sources", (list, tuple), path="workflow")
    source_values = sequence(sources, path="workflow.sources")
    video_ids: set[str] = set()
    for index, source in enumerate(source_values):
        source_path = f"workflow.sources[{index}]"
        source_mapping = exact_mapping(
            source,
            path=source_path,
            required={"video_id", "url", "local_video"},
        )
        _validate_field_types(
            source_mapping,
            path=source_path,
            fields={
                "video_id": str,
                "url": str,
                "local_video": (dict, DictConfig, type(None)),
            },
        )
        video_id = _validate_trimmed_string(
            source_mapping["video_id"], path=f"{source_path}.video_id"
        )
        _validate_http_url(source_mapping["url"], path=f"{source_path}.url")
        if video_id in video_ids:
            raise SemanticConfigurationError(
                f"{source_path}.video_id duplicates {video_id!r}."
            )
        video_ids.add(video_id)
        local_video = source_mapping["local_video"]
        if local_video is not None:
            declaration = exact_mapping(
                local_video,
                path=f"{source_path}.local_video",
                required={"role", "path"},
            )
            role_value = cast(
                str,
                typed(
                    declaration,
                    "role",
                    str,
                    path=f"{source_path}.local_video",
                ),
            )
            if role_value not in {
                PathRole.DATA.value,
                PathRole.EXTERNAL_ASSET.value,
            }:
                raise SemanticConfigurationError(
                    f"{source_path}.local_video.role must be 'data' or "
                    "'external_asset'."
                )
            local_fragment = _validate_relative_child(
                typed(
                    declaration,
                    "path",
                    str,
                    path=f"{source_path}.local_video",
                ),
                path=f"{source_path}.local_video.path",
            )
            paths.resolver.resolve(PathRole(role_value), local_fragment)
    discovery = exact_mapping(
        workflow["discovery"],
        path="workflow.discovery",
        required={
            "enabled",
            "queries",
            "max_results_per_query",
            "min_duration_sec",
            "max_duration_sec",
            "allow_unknown_duration",
        },
    )
    _validate_field_types(
        discovery,
        path="workflow.discovery",
        fields={
            "enabled": bool,
            "queries": (list, tuple),
            "max_results_per_query": int,
            "min_duration_sec": (float, int),
            "max_duration_sec": (float, int),
            "allow_unknown_duration": bool,
        },
    )
    discovery_enabled = cast(bool, discovery["enabled"])
    _validate_string_sequence(
        discovery["queries"],
        path="workflow.discovery.queries",
        allow_empty=not discovery_enabled,
    )
    max_results = cast(int, discovery["max_results_per_query"])
    min_duration = cast(float | int, discovery["min_duration_sec"])
    max_duration = cast(float | int, discovery["max_duration_sec"])
    _positive(
        max_results,
        path="workflow.discovery.max_results_per_query",
        allow_zero=not discovery_enabled,
    )
    _positive(
        min_duration,
        path="workflow.discovery.min_duration_sec",
        allow_zero=True,
    )
    _positive(
        max_duration,
        path="workflow.discovery.max_duration_sec",
        allow_zero=not discovery_enabled,
    )
    if min_duration > max_duration:
        raise SemanticConfigurationError(
            "workflow.discovery.min_duration_sec must be <= "
            "workflow.discovery.max_duration_sec."
        )
    if not discovery_enabled and not source_values:
        raise SemanticConfigurationError(
            "workflow.sources must not be empty when workflow.discovery.enabled=false."
        )
    processing = exact_mapping(
        workflow["processing"],
        path="workflow.processing",
        required={
            "max_new_videos",
            "reprocess_existing",
            "cleanup_videos_after_processing",
            "cleanup_keep_info_json",
        },
    )
    _validate_field_types(
        processing,
        path="workflow.processing",
        fields={
            "max_new_videos": int,
            "reprocess_existing": bool,
            "cleanup_videos_after_processing": bool,
            "cleanup_keep_info_json": bool,
        },
    )
    _positive(
        cast(int, processing["max_new_videos"]),
        path="workflow.processing.max_new_videos",
    )
    storage = exact_mapping(
        workflow["storage"],
        path="workflow.storage",
        required={"enabled", "max_root_gb"},
    )
    _validate_field_types(
        storage,
        path="workflow.storage",
        fields={"enabled": bool, "max_root_gb": (float, int, type(None))},
    )
    storage_enabled = cast(bool, storage["enabled"])
    max_root_gb = storage["max_root_gb"]
    if max_root_gb is None:
        if storage_enabled:
            raise SemanticConfigurationError(
                "workflow.storage.max_root_gb is required when storage is enabled."
            )
    else:
        _positive(
            cast(float | int, max_root_gb),
            path="workflow.storage.max_root_gb",
            allow_zero=not storage_enabled,
        )
    frames = exact_mapping(
        workflow["frames"],
        path="workflow.frames",
        required={"frames_per_video", "output_ext", "jpeg_quality", "overwrite"},
    )
    _validate_field_types(
        frames,
        path="workflow.frames",
        fields={
            "frames_per_video": int,
            "output_ext": str,
            "jpeg_quality": int,
            "overwrite": bool,
        },
    )
    _positive(
        cast(int, frames["frames_per_video"]),
        path="workflow.frames.frames_per_video",
    )
    output_ext = cast(str, frames["output_ext"])
    if output_ext not in {"jpg", "jpeg", "png"}:
        raise SemanticConfigurationError(
            "workflow.frames.output_ext must be 'jpg', 'jpeg', or 'png'."
        )
    jpeg_quality = cast(int, frames["jpeg_quality"])
    if not 1 <= jpeg_quality <= 100:
        raise SemanticConfigurationError(
            "workflow.frames.jpeg_quality must be in [1, 100]."
        )
    _validate_download(
        workflow["download"],
        path="workflow.download",
        require_av1=False,
    )
    _validate_transcode(
        workflow["transcode"],
        path="workflow.transcode",
        fallback_on_decode_error=True,
    )
    gate = exact_mapping(
        workflow["gate"],
        path="workflow.gate",
        required={"backend", "mock", "contact_sheet", "vllm"},
    )
    backend = cast(str, typed(gate, "backend", str, path="workflow.gate"))
    if backend not in {"mock", "vllm"}:
        raise SemanticConfigurationError(
            "workflow.gate.backend must be 'mock' or 'vllm'."
        )
    mock = exact_mapping(
        gate["mock"], path="workflow.gate.mock", required={"accept_all"}
    )
    typed(mock, "accept_all", bool, path="workflow.gate.mock")
    contact_sheet = exact_mapping(
        gate["contact_sheet"],
        path="workflow.gate.contact_sheet",
        required={"max_images", "columns", "thumb_width", "thumb_height"},
    )
    for key in contact_sheet:
        _positive(
            cast(
                int, typed(contact_sheet, key, int, path="workflow.gate.contact_sheet")
            ),
            path=f"workflow.gate.contact_sheet.{key}",
        )
    vllm = exact_mapping(
        gate["vllm"],
        path="workflow.gate.vllm",
        required={
            "base_url",
            "model",
            "timeout_sec",
            "max_tokens",
            "accept_labels",
            "extra_body",
            "server",
            "prompt",
        },
    )
    server = exact_mapping(
        vllm["server"],
        path="workflow.gate.vllm.server",
        required={
            "enabled",
            "executable_role",
            "executable",
            "command",
            "env",
            "cwd",
            "health_url",
            "startup_timeout_sec",
            "poll_interval_sec",
            "request_timeout_sec",
            "shutdown_timeout_sec",
            "stop_on_exit",
            "log_path",
            "preflight",
        },
    )
    _validate_field_types(
        vllm,
        path="workflow.gate.vllm",
        fields={
            "base_url": str,
            "model": str,
            "timeout_sec": (float, int),
            "max_tokens": int,
            "accept_labels": (list, tuple),
            "extra_body": (dict, DictConfig),
            "server": (dict, DictConfig),
            "prompt": str,
        },
    )
    vllm_active = backend == "vllm"
    accept_labels = _validate_string_sequence(
        vllm["accept_labels"],
        path="workflow.gate.vllm.accept_labels",
        allow_empty=not vllm_active,
    )
    if vllm_active:
        _validate_http_url(
            vllm["base_url"], path="workflow.gate.vllm.base_url"
        )
        for key in ("model", "prompt"):
            _validate_trimmed_string(vllm[key], path=f"workflow.gate.vllm.{key}")
        normalized_labels = tuple(label.lower() for label in accept_labels)
        if normalized_labels != accept_labels:
            raise SemanticConfigurationError(
                "workflow.gate.vllm.accept_labels must use lowercase labels."
            )
        if len(set(normalized_labels)) != len(normalized_labels):
            raise SemanticConfigurationError(
                "workflow.gate.vllm.accept_labels must not contain duplicates."
            )
        if any(
            label not in {"tennis", "non_tennis", "unknown"}
            for label in normalized_labels
        ):
            raise SemanticConfigurationError(
                "workflow.gate.vllm.accept_labels may contain only 'tennis', "
                "'non_tennis', and 'unknown'."
            )
    _positive(
        cast(float | int, vllm["timeout_sec"]),
        path="workflow.gate.vllm.timeout_sec",
        allow_zero=not vllm_active,
    )
    _positive(
        cast(int, vllm["max_tokens"]),
        path="workflow.gate.vllm.max_tokens",
        allow_zero=not vllm_active,
    )
    _validate_field_types(
        server,
        path="workflow.gate.vllm.server",
        fields={
            "enabled": bool,
            "executable_role": str,
            "executable": str,
            "command": (list, tuple),
            "env": (dict, DictConfig),
            "cwd": (str, type(None)),
            "health_url": str,
            "startup_timeout_sec": (float, int),
            "poll_interval_sec": (float, int),
            "request_timeout_sec": (float, int),
            "shutdown_timeout_sec": (float, int),
            "stop_on_exit": bool,
            "log_path": (str, type(None)),
            "preflight": (dict, DictConfig),
        },
    )
    server_active = vllm_active and cast(bool, server["enabled"])
    _validate_string_sequence(
        server["command"],
        path="workflow.gate.vllm.server.command",
        allow_empty=not server_active,
    )
    if server_active:
        _validate_http_url(
            server["health_url"], path="workflow.gate.vllm.server.health_url"
        )
    for key in (
        "startup_timeout_sec",
        "poll_interval_sec",
        "request_timeout_sec",
        "shutdown_timeout_sec",
    ):
        _positive(
            cast(float | int, server[key]),
            path=f"workflow.gate.vllm.server.{key}",
            allow_zero=not server_active,
        )
    executable_role = cast(str, server["executable_role"])
    executable_fragment = _validate_relative_child(
        server["executable"], path="workflow.gate.vllm.server.executable"
    )
    if executable_role == PathRole.CACHE.value:
        paths.cache(executable_fragment)
    elif executable_role == PathRole.EXTERNAL_ASSET.value:
        paths.external_asset(executable_fragment)
    else:
        raise SemanticConfigurationError(
            "workflow.gate.vllm.server.executable_role must be 'cache' or "
            "'external_asset'."
        )
    environment = as_mapping(server["env"], path="workflow.gate.vllm.server.env")
    if any(
        type(key) is not str or type(value) is not str
        for key, value in environment.items()
    ):
        raise ConfigurationTypeError(
            "workflow.gate.vllm.server.env must map strings to strings."
        )
    cwd = server["cwd"]
    if cwd is not None:
        paths.project(cast(str, cwd))
    log_path = server["log_path"]
    if log_path is not None:
        paths.output(cast(str, log_path))
    preflight = exact_mapping(
        server["preflight"],
        path="workflow.gate.vllm.server.preflight",
        required={"enabled", "command"},
    )
    typed(preflight, "enabled", bool, path="workflow.gate.vllm.server.preflight")
    preflight_active = server_active and cast(bool, preflight["enabled"])
    _validate_string_sequence(
        typed(
            preflight,
            "command",
            (list, tuple),
            path="workflow.gate.vllm.server.preflight",
        ),
        path="workflow.gate.vllm.server.preflight.command",
        allow_empty=not preflight_active,
    )


def validate_annotation_boundary(config: DictConfig) -> None:
    """Validate the annotation session root and exact top-level shape."""
    paths = BallRuntimePaths.from_config(config)
    root = exact_mapping(config, path="configuration", required={"paths", "annotate"})
    annotation = exact_mapping(
        root["annotate"],
        path="annotate",
        required={
            "root",
            "video_id",
            "candidate_id",
            "start_index",
            "window_name",
            "max_display_width",
            "max_display_height",
            "point_radius",
            "point_thickness",
            "max_balls_per_frame",
            "zoom",
            "finalize",
        },
    )
    _validate_field_types(
        annotation,
        path="annotate",
        fields={
            "root": str,
            "video_id": str,
            "candidate_id": (str, type(None)),
            "start_index": (int, type(None)),
            "window_name": str,
            "max_display_width": int,
            "max_display_height": int,
            "point_radius": int,
            "point_thickness": int,
            "max_balls_per_frame": int,
            "zoom": (dict, DictConfig),
            "finalize": (dict, DictConfig),
        },
    )
    paths.data(cast(str, annotation["root"]))
    _validate_trimmed_string(annotation["video_id"], path="annotate.video_id")
    _validate_optional_trimmed_string(
        annotation["candidate_id"], path="annotate.candidate_id"
    )
    _validate_trimmed_string(annotation["window_name"], path="annotate.window_name")
    start_index = annotation["start_index"]
    if start_index is not None:
        _positive(
            cast(int, start_index), path="annotate.start_index", allow_zero=True
        )
    for key in (
        "max_display_width",
        "max_display_height",
        "point_radius",
        "point_thickness",
        "max_balls_per_frame",
    ):
        _positive(cast(int, annotation[key]), path=f"annotate.{key}")
    zoom = exact_mapping(
        annotation["zoom"], path="annotate.zoom", required={"key", "factor"}
    )
    _validate_field_types(
        zoom, path="annotate.zoom", fields={"key": str, "factor": (float, int)}
    )
    finalize = exact_mapping(
        annotation["finalize"], path="annotate.finalize", required={"key", "overwrite"}
    )
    _validate_field_types(
        finalize,
        path="annotate.finalize",
        fields={"key": str, "overwrite": bool},
    )
    zoom_key = cast(str, zoom["key"])
    finalize_key = cast(str, finalize["key"])
    if len(zoom_key) != 1 or len(finalize_key) != 1:
        raise SemanticConfigurationError(
            "annotate.zoom.key and annotate.finalize.key must each be one character."
        )
    if zoom_key.lower() == finalize_key.lower():
        raise SemanticConfigurationError(
            "annotate.zoom.key and annotate.finalize.key must differ."
        )
    factor = cast(float | int, zoom["factor"])
    if not math.isfinite(factor) or factor <= 1:
        raise SemanticConfigurationError("annotate.zoom.factor must be greater than 1.")


def _register() -> None:
    register_boundary_validator("ball.train", validate_training)
    register_boundary_validator("ball.train_staged", validate_training)
    register_boundary_validator("ball.visualize", validate_visualization)
    register_boundary_validator("ball.eval", validate_eval)
    register_boundary_validator("ball.evaluate_manifest", validate_manifest_boundary)
    register_boundary_validator("ball.preview", validate_preview)
    register_boundary_validator("ball.web_tool", validate_web_tool)
    register_boundary_validator("ball.youtube", validate_youtube_boundary)
    register_boundary_validator("ball.annotation", validate_annotation_boundary)


_register()


__all__ = [
    "BallRuntimePaths",
    "BallYoutubePathContract",
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

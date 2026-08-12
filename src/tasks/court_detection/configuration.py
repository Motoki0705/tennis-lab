"""Strict typed configuration contracts for court-detection training."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from omegaconf import DictConfig

from src.tasks.base.configuration import (
    TrainingRuntimeConfig,
    as_config_mapping,
    require_config_mapping,
    require_config_value,
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
from src.utils.paths import PROJECT_ROOT

if TYPE_CHECKING:
    from src.tasks.court_detection.visualization.rendering import CourtRenderStyle

ConfigMapping: TypeAlias = Mapping[str, object]
CourtSourceKind: TypeAlias = Literal["tennis_court_detector", "synthetic_court"]
CourtSourceSplit: TypeAlias = Literal["train", "val", "test"]
CourtTargetKind: TypeAlias = Literal["kp", "seg", "line"]

SEGMENTATION_TARGET_SCHEMA = "court_cell_segmentation_v1"
LINE_TARGET_SCHEMA = "court_line_binary_v1"


def _exact(mapping: ConfigMapping, keys: set[str], *, path: str) -> None:
    unknown = sorted(set(mapping) - keys)
    if unknown:
        raise UnknownConfigurationKeyError(
            f"Unknown configuration key(s): {', '.join(f'{path}.{key}' for key in unknown)}."
        )
    missing = sorted(keys - set(mapping))
    if missing:
        raise MissingConfigurationKeyError(
            f"Missing required configuration key(s): {', '.join(f'{path}.{key}' for key in missing)}."
        )


def _resolver(config: ConfigMapping) -> PathResolver:
    paths = require_config_mapping(config, "paths", path="configuration")
    return PathResolver(
        RuntimePathRoots.from_mapping(paths, repository_root=PROJECT_ROOT)
    )


def _number(mapping: ConfigMapping, key: str, *, path: str) -> float:
    value = float(
        cast("float | int", require_config_value(mapping, key, (float, int), path=path))
    )
    if not math.isfinite(value):
        raise SemanticConfigurationError(f"{path}.{key} must be finite.")
    return value


def _integer(mapping: ConfigMapping, key: str, *, path: str) -> int:
    return cast("int", require_config_value(mapping, key, int, path=path))


def _string(mapping: ConfigMapping, key: str, *, path: str) -> str:
    value = cast("str", require_config_value(mapping, key, str, path=path))
    if not value or value != value.strip():
        raise SemanticConfigurationError(f"{path}.{key} must be non-empty and trimmed.")
    return value


def _bool(mapping: ConfigMapping, key: str, *, path: str) -> bool:
    return cast("bool", require_config_value(mapping, key, bool, path=path))


def _sequence(mapping: ConfigMapping, key: str, *, path: str) -> Sequence[object]:
    return cast(
        "Sequence[object]",
        require_config_value(mapping, key, (list, tuple), path=path),
    )


def _float_tuple(
    mapping: ConfigMapping, key: str, length: int, *, path: str
) -> tuple[float, ...]:
    raw = _sequence(mapping, key, path=path)
    if len(raw) != length or any(type(item) not in (float, int) for item in raw):
        raise ConfigurationTypeError(
            f"{path}.{key} must contain exactly {length} numbers."
        )
    values = tuple(float(cast("float | int", item)) for item in raw)
    if any(not math.isfinite(item) for item in values):
        raise SemanticConfigurationError(f"{path}.{key} must contain finite values.")
    return values


def _int_tuple(mapping: ConfigMapping, key: str, *, path: str) -> tuple[int, ...]:
    raw = _sequence(mapping, key, path=path)
    if any(type(item) is not int for item in raw):
        raise ConfigurationTypeError(f"{path}.{key} must contain only integers.")
    return tuple(cast("int", item) for item in raw)


def _rgb_tuple(mapping: ConfigMapping, key: str, *, path: str) -> tuple[int, int, int]:
    values = _int_tuple(mapping, key, path=path)
    if len(values) != 3 or any(channel < 0 or channel > 255 for channel in values):
        raise SemanticConfigurationError(
            f"{path}.{key} must contain exactly three integers in [0, 255]."
        )
    return values


@dataclass(frozen=True, slots=True)
class CourtRenderConfig:
    """Exact 2-D panel style shared by visualization and qualitative logging."""

    background_rgb: tuple[int, int, int]
    text_color_rgb: tuple[int, int, int]
    text_scale: float
    text_thickness: int
    tile_gap: int
    panel_label_height: int
    header_height: int
    display_width: int
    kp_radius: int
    kp_color_rgb: tuple[int, int, int]
    kp_thickness: int
    line_threshold: float

    @classmethod
    def from_mapping(cls, value: object) -> CourtRenderConfig:
        mapping = as_config_mapping(value, path="render_style")
        _exact(mapping, {"draw", "layout"}, path="render_style")
        draw = require_config_mapping(mapping, "draw", path="render_style")
        layout = require_config_mapping(mapping, "layout", path="render_style")
        _exact(
            draw,
            {"kp_radius", "kp_color_rgb", "kp_thickness", "line_threshold"},
            path="render_style.draw",
        )
        _exact(
            layout,
            {
                "header_height",
                "display_width",
                "tile_gap",
                "text_scale",
                "text_thickness",
                "background_rgb",
                "text_color_rgb",
                "panel_label_height",
            },
            path="render_style.layout",
        )
        result = cls(
            background_rgb=_rgb_tuple(
                layout, "background_rgb", path="render_style.layout"
            ),
            text_color_rgb=_rgb_tuple(
                layout, "text_color_rgb", path="render_style.layout"
            ),
            text_scale=_number(layout, "text_scale", path="render_style.layout"),
            text_thickness=_integer(
                layout, "text_thickness", path="render_style.layout"
            ),
            tile_gap=_integer(layout, "tile_gap", path="render_style.layout"),
            panel_label_height=_integer(
                layout, "panel_label_height", path="render_style.layout"
            ),
            header_height=_integer(layout, "header_height", path="render_style.layout"),
            display_width=_integer(layout, "display_width", path="render_style.layout"),
            kp_radius=_integer(draw, "kp_radius", path="render_style.draw"),
            kp_color_rgb=_rgb_tuple(draw, "kp_color_rgb", path="render_style.draw"),
            kp_thickness=_integer(draw, "kp_thickness", path="render_style.draw"),
            line_threshold=_number(draw, "line_threshold", path="render_style.draw"),
        )
        if (
            result.text_scale <= 0
            or result.text_thickness <= 0
            or result.tile_gap < 0
            or result.panel_label_height < 0
            or result.header_height <= 0
            or result.display_width <= 0
            or result.kp_radius <= 0
            or result.kp_thickness < -1
        ):
            raise SemanticConfigurationError(
                "render_style layout/draw dimensions and thicknesses are invalid."
            )
        if not 0.0 <= result.line_threshold <= 1.0:
            raise SemanticConfigurationError(
                "render_style.draw.line_threshold must be in [0, 1]."
            )
        return result

    def build(self) -> CourtRenderStyle:
        from src.tasks.base.visualization.layout import PanelStyle
        from src.tasks.court_detection.visualization.rendering import CourtRenderStyle

        return CourtRenderStyle(
            panel=PanelStyle(
                background_rgb=self.background_rgb,
                text_color_rgb=self.text_color_rgb,
                text_scale=self.text_scale,
                text_thickness=self.text_thickness,
                tile_gap=self.tile_gap,
                panel_label_height=self.panel_label_height,
            ),
            header_height=self.header_height,
            display_width=self.display_width,
            kp_radius=self.kp_radius,
            kp_color_rgb=self.kp_color_rgb,
            kp_thickness=self.kp_thickness,
            line_threshold=self.line_threshold,
        )


@dataclass(frozen=True, slots=True)
class CourtAugmentationConfig:
    train_scales: tuple[int, ...]
    val_short_side: int
    crop_scale: tuple[float, float]
    crop_ratio: tuple[float, float]
    hflip_prob: float
    affine_degrees: float
    affine_translate: tuple[float, float]
    affine_scale: tuple[float, float]
    affine_shear: float
    perspective_distortion: float
    perspective_prob: float
    color_jitter: tuple[float, float, float, float]
    gaussian_blur_kernel: tuple[int, ...]
    gaussian_blur_sigma: tuple[float, float]
    gaussian_blur_prob: float
    min_visible_kp: int
    visibility_max_retries: int

    @classmethod
    def from_mapping(cls, value: object) -> CourtAugmentationConfig:
        mapping = as_config_mapping(value, path="data.augmentation")
        keys = {
            "train_scales",
            "val_short_side",
            "crop_scale",
            "crop_ratio",
            "hflip_prob",
            "affine_degrees",
            "affine_translate",
            "affine_scale",
            "affine_shear",
            "perspective_distortion",
            "perspective_prob",
            "color_jitter",
            "gaussian_blur_kernel",
            "gaussian_blur_sigma",
            "gaussian_blur_prob",
            "min_visible_kp",
            "visibility_max_retries",
        }
        _exact(mapping, keys, path="data.augmentation")
        result = cls(
            train_scales=_int_tuple(mapping, "train_scales", path="data.augmentation"),
            val_short_side=_integer(
                mapping, "val_short_side", path="data.augmentation"
            ),
            crop_scale=cast(
                "tuple[float, float]",
                _float_tuple(mapping, "crop_scale", 2, path="data.augmentation"),
            ),
            crop_ratio=cast(
                "tuple[float, float]",
                _float_tuple(mapping, "crop_ratio", 2, path="data.augmentation"),
            ),
            hflip_prob=_number(mapping, "hflip_prob", path="data.augmentation"),
            affine_degrees=_number(mapping, "affine_degrees", path="data.augmentation"),
            affine_translate=cast(
                "tuple[float, float]",
                _float_tuple(mapping, "affine_translate", 2, path="data.augmentation"),
            ),
            affine_scale=cast(
                "tuple[float, float]",
                _float_tuple(mapping, "affine_scale", 2, path="data.augmentation"),
            ),
            affine_shear=_number(mapping, "affine_shear", path="data.augmentation"),
            perspective_distortion=_number(
                mapping, "perspective_distortion", path="data.augmentation"
            ),
            perspective_prob=_number(
                mapping, "perspective_prob", path="data.augmentation"
            ),
            color_jitter=cast(
                "tuple[float, float, float, float]",
                _float_tuple(mapping, "color_jitter", 4, path="data.augmentation"),
            ),
            gaussian_blur_kernel=_int_tuple(
                mapping, "gaussian_blur_kernel", path="data.augmentation"
            ),
            gaussian_blur_sigma=cast(
                "tuple[float, float]",
                _float_tuple(
                    mapping, "gaussian_blur_sigma", 2, path="data.augmentation"
                ),
            ),
            gaussian_blur_prob=_number(
                mapping, "gaussian_blur_prob", path="data.augmentation"
            ),
            min_visible_kp=_integer(
                mapping, "min_visible_kp", path="data.augmentation"
            ),
            visibility_max_retries=_integer(
                mapping, "visibility_max_retries", path="data.augmentation"
            ),
        )
        if not result.train_scales or any(scale <= 0 for scale in result.train_scales):
            raise SemanticConfigurationError(
                "data.augmentation.train_scales must contain positive values."
            )
        if result.val_short_side <= 0 or result.visibility_max_retries <= 0:
            raise SemanticConfigurationError(
                "Court image sizes and retry counts must be positive."
            )
        if not 0.0 < result.crop_scale[0] <= result.crop_scale[1] <= 1.0:
            raise SemanticConfigurationError(
                "data.augmentation.crop_scale must be ordered in (0, 1]."
            )
        if not 0.0 < result.crop_ratio[0] <= result.crop_ratio[1]:
            raise SemanticConfigurationError(
                "data.augmentation.crop_ratio must be positive and ordered."
            )
        for name, probability in (
            ("hflip_prob", result.hflip_prob),
            ("perspective_prob", result.perspective_prob),
            ("gaussian_blur_prob", result.gaussian_blur_prob),
        ):
            if not 0.0 <= probability <= 1.0:
                raise SemanticConfigurationError(
                    f"data.augmentation.{name} must be in [0, 1]."
                )
        if result.affine_degrees < 0.0 or result.affine_shear < 0.0:
            raise SemanticConfigurationError(
                "data.augmentation affine degrees and shear must be non-negative."
            )
        if any(not 0.0 <= value <= 1.0 for value in result.affine_translate):
            raise SemanticConfigurationError(
                "data.augmentation.affine_translate must be in [0, 1]."
            )
        if not 0.0 < result.affine_scale[0] <= result.affine_scale[1]:
            raise SemanticConfigurationError(
                "data.augmentation.affine_scale must be positive and ordered."
            )
        if not 0.0 <= result.perspective_distortion <= 1.0:
            raise SemanticConfigurationError(
                "data.augmentation.perspective_distortion must be in [0, 1]."
            )
        if any(value < 0.0 for value in result.color_jitter[:3]) or not (
            0.0 <= result.color_jitter[3] <= 0.5
        ):
            raise SemanticConfigurationError(
                "data.augmentation.color_jitter brightness/contrast/saturation must "
                "be non-negative and hue must be in [0, 0.5]."
            )
        if not result.gaussian_blur_kernel or any(
            kernel <= 0 or kernel % 2 == 0
            for kernel in result.gaussian_blur_kernel
        ):
            raise SemanticConfigurationError(
                "data.augmentation.gaussian_blur_kernel must contain positive odd values."
            )
        if not 0.0 < result.gaussian_blur_sigma[0] <= result.gaussian_blur_sigma[1]:
            raise SemanticConfigurationError(
                "data.augmentation.gaussian_blur_sigma must be positive and ordered."
            )
        if result.min_visible_kp < 0:
            raise SemanticConfigurationError(
                "data.augmentation.min_visible_kp must be non-negative."
            )
        return result


@dataclass(frozen=True, slots=True)
class TennisCourtDetectorSourceConfig:
    kind: Literal["tennis_court_detector"]
    root: Path
    split_mapping: Mapping[CourtSourceSplit, str | None]

    @classmethod
    def from_mapping(
        cls, value: object, *, resolver: PathResolver
    ) -> TennisCourtDetectorSourceConfig:
        mapping = as_config_mapping(value, path="data.source")
        _exact(mapping, {"kind", "root", "split_mapping"}, path="data.source")
        if _string(mapping, "kind", path="data.source") != "tennis_court_detector":
            raise SemanticConfigurationError(
                "data.source.kind must be 'tennis_court_detector'."
            )
        raw_split = require_config_mapping(mapping, "split_mapping", path="data.source")
        _exact(raw_split, {"train", "val", "test"}, path="data.source.split_mapping")
        resolved: dict[CourtSourceSplit, str | None] = {}
        for split in ("train", "val", "test"):
            value_at_split = require_config_value(
                raw_split,
                split,
                (str, type(None)),
                path="data.source.split_mapping",
            )
            if value_at_split not in {"train", "val", None}:
                raise SemanticConfigurationError(
                    "TennisCourtDetector split mappings must be train, val, or null."
                )
            resolved[split] = value_at_split
        if resolved["train"] is None or resolved["val"] is None:
            raise SemanticConfigurationError(
                "TennisCourtDetector train and val mappings must be explicit."
            )
        return cls(
            kind="tennis_court_detector",
            root=resolver.resolve(
                PathRole.DATA, _string(mapping, "root", path="data.source")
            ),
            split_mapping=MappingProxyType(resolved),
        )


@dataclass(frozen=True, slots=True)
class SyntheticCourtSourceConfig:
    kind: Literal["synthetic_court"]
    workspace_root: Path
    scene_ids: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls, value: object, *, resolver: PathResolver
    ) -> SyntheticCourtSourceConfig:
        mapping = as_config_mapping(value, path="data.source")
        _exact(mapping, {"kind", "workspace_root", "scene_ids"}, path="data.source")
        if _string(mapping, "kind", path="data.source") != "synthetic_court":
            raise SemanticConfigurationError(
                "data.source.kind must be 'synthetic_court'."
            )
        raw_ids = _sequence(mapping, "scene_ids", path="data.source")
        scene_ids_list: list[str] = []
        for item in raw_ids:
            if (
                type(item) is not str
                or not item
                or item != item.strip()
                or "/" in item
                or "\\" in item
            ):
                raise ConfigurationTypeError(
                    "data.source.scene_ids must contain safe non-empty scene IDs."
                )
            scene_ids_list.append(item)
        if not scene_ids_list:
            raise ConfigurationTypeError(
                "data.source.scene_ids must contain safe non-empty scene IDs."
            )
        scene_ids = tuple(scene_ids_list)
        if len(set(scene_ids)) != len(scene_ids):
            raise SemanticConfigurationError(
                "data.source.scene_ids must not contain duplicates."
            )
        return cls(
            kind="synthetic_court",
            workspace_root=resolver.resolve(
                PathRole.DATA,
                _string(mapping, "workspace_root", path="data.source"),
            ),
            scene_ids=scene_ids,
        )


CourtSourceConfig: TypeAlias = (
    TennisCourtDetectorSourceConfig | SyntheticCourtSourceConfig
)


def _source_config(value: object, *, resolver: PathResolver) -> CourtSourceConfig:
    mapping = as_config_mapping(value, path="data.source")
    kind = _string(mapping, "kind", path="data.source")
    if kind == "tennis_court_detector":
        return TennisCourtDetectorSourceConfig.from_mapping(mapping, resolver=resolver)
    if kind == "synthetic_court":
        return SyntheticCourtSourceConfig.from_mapping(mapping, resolver=resolver)
    raise SemanticConfigurationError(
        "data.source.kind must be tennis_court_detector or synthetic_court."
    )


@dataclass(frozen=True, slots=True)
class CourtTargetConfig:
    kind: CourtTargetKind
    sigma_ratio: float | None
    target_schema: str | None

    @classmethod
    def from_mapping(cls, value: object, *, index: int) -> CourtTargetConfig:
        path = f"data.processing.targets[{index}]"
        mapping = as_config_mapping(value, path=path)
        kind = _string(mapping, "kind", path=path)
        if kind == "kp":
            _exact(mapping, {"kind", "sigma_ratio"}, path=path)
            sigma = _number(mapping, "sigma_ratio", path=path)
            if sigma <= 0.0:
                raise SemanticConfigurationError(f"{path}.sigma_ratio must be positive.")
            return cls(kind="kp", sigma_ratio=sigma, target_schema=None)
        if kind in {"seg", "line"}:
            _exact(mapping, {"kind", "target_schema"}, path=path)
            schema = _string(mapping, "target_schema", path=path)
            expected = (
                SEGMENTATION_TARGET_SCHEMA if kind == "seg" else LINE_TARGET_SCHEMA
            )
            if schema != expected:
                raise SemanticConfigurationError(
                    f"{path}.target_schema must be {expected!r}."
                )
            return cls(
                kind=cast(CourtTargetKind, kind),
                sigma_ratio=None,
                target_schema=schema,
            )
        raise SemanticConfigurationError(f"{path}.kind must be kp, seg, or line.")


@dataclass(frozen=True, slots=True)
class CourtProcessingConfig:
    derived_target_root: Path
    targets: tuple[CourtTargetConfig, ...]

    @classmethod
    def from_mapping(
        cls, value: object, *, resolver: PathResolver
    ) -> CourtProcessingConfig:
        mapping = as_config_mapping(value, path="data.processing")
        _exact(
            mapping,
            {"derived_target_root", "targets"},
            path="data.processing",
        )
        raw_targets = _sequence(mapping, "targets", path="data.processing")
        if not raw_targets:
            raise SemanticConfigurationError(
                "data.processing.targets must be non-empty."
            )
        targets = tuple(
            CourtTargetConfig.from_mapping(item, index=index)
            for index, item in enumerate(raw_targets)
        )
        kinds = tuple(target.kind for target in targets)
        if len(set(kinds)) != len(kinds):
            raise SemanticConfigurationError(
                "data.processing.targets must not repeat a target kind."
            )
        return cls(
            derived_target_root=resolver.resolve(
                PathRole.DATA,
                _string(
                    mapping,
                    "derived_target_root",
                    path="data.processing",
                ),
            ),
            targets=targets,
        )


@dataclass(frozen=True, slots=True)
class CourtDataConfig:
    source: CourtSourceConfig
    processing: CourtProcessingConfig
    batch_size: int
    num_workers: int
    pin_memory: bool
    augmentation: CourtAugmentationConfig

    @classmethod
    def from_mapping(cls, value: object, *, resolver: PathResolver) -> CourtDataConfig:
        mapping = as_config_mapping(value, path="data")
        _exact(
            mapping,
            {
                "source",
                "processing",
                "batch_size",
                "num_workers",
                "pin_memory",
                "augmentation",
            },
            path="data",
        )
        batch_size = _integer(mapping, "batch_size", path="data")
        num_workers = _integer(mapping, "num_workers", path="data")
        if batch_size <= 0 or num_workers < 0:
            raise SemanticConfigurationError(
                "data.batch_size must be positive and data.num_workers non-negative."
            )
        return cls(
            source=_source_config(
                require_config_mapping(mapping, "source", path="data"),
                resolver=resolver,
            ),
            processing=CourtProcessingConfig.from_mapping(
                require_config_mapping(mapping, "processing", path="data"),
                resolver=resolver,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=_bool(mapping, "pin_memory", path="data"),
            augmentation=CourtAugmentationConfig.from_mapping(
                require_config_mapping(mapping, "augmentation", path="data")
            ),
        )


@dataclass(frozen=True, slots=True)
class CourtLoRAConfig:
    enabled: bool
    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> CourtLoRAConfig:
        mapping = as_config_mapping(value, path="model.encoder.lora")
        _exact(
            mapping,
            {"enabled", "rank", "alpha", "dropout", "target_modules"},
            path="model.encoder.lora",
        )
        raw_targets = _sequence(mapping, "target_modules", path="model.encoder.lora")
        if not raw_targets or any(
            type(item) is not str or not item for item in raw_targets
        ):
            raise ConfigurationTypeError(
                "model.encoder.lora.target_modules must contain non-empty strings."
            )
        result = cls(
            enabled=_bool(mapping, "enabled", path="model.encoder.lora"),
            rank=_integer(mapping, "rank", path="model.encoder.lora"),
            alpha=_number(mapping, "alpha", path="model.encoder.lora"),
            dropout=_number(mapping, "dropout", path="model.encoder.lora"),
            target_modules=tuple(cast("str", item) for item in raw_targets),
        )
        if result.rank <= 0 or result.alpha <= 0 or not 0.0 <= result.dropout < 1.0:
            raise SemanticConfigurationError(
                "Invalid model.encoder.lora rank/alpha/dropout."
            )
        return result


@dataclass(frozen=True, slots=True)
class CourtEncoderConfig:
    name: str
    repository_path: Path | None
    checkpoint_path: Path | None
    backbone_name: str | None
    strict: bool | None
    train_mode: str | None
    last_n_blocks: int | None
    out_indices: tuple[int, ...] | None
    layer_mode: str | None
    lora: CourtLoRAConfig | None

    @classmethod
    def from_mapping(
        cls, value: object, *, resolver: PathResolver
    ) -> CourtEncoderConfig:
        mapping = as_config_mapping(value, path="model.encoder")
        name = _string(mapping, "name", path="model.encoder")
        if name == "default":
            _exact(mapping, {"name"}, path="model.encoder")
            return cls(
                name=name,
                repository_path=None,
                checkpoint_path=None,
                backbone_name=None,
                strict=None,
                train_mode=None,
                last_n_blocks=None,
                out_indices=None,
                layer_mode=None,
                lora=None,
            )
        if name != "dinov3":
            raise SemanticConfigurationError(
                "model.encoder.name must be 'default' or 'dinov3'."
            )
        keys = {
            "name",
            "backbone_name",
            "repository_path",
            "checkpoint_path",
            "strict",
            "train_mode",
            "last_n_blocks",
            "out_indices",
            "layer_mode",
            "lora",
        }
        _exact(mapping, keys, path="model.encoder")
        train_mode = _string(mapping, "train_mode", path="model.encoder")
        layer_mode = _string(mapping, "layer_mode", path="model.encoder")
        if train_mode not in {"frozen", "last_n", "full"}:
            raise SemanticConfigurationError("model.encoder.train_mode is invalid.")
        if layer_mode not in {"uniform", "last"}:
            raise SemanticConfigurationError("model.encoder.layer_mode is invalid.")
        out_indices = _int_tuple(mapping, "out_indices", path="model.encoder")
        if len(out_indices) != 4:
            raise SemanticConfigurationError(
                "model.encoder.out_indices must contain exactly four layers."
            )
        last_n_blocks = _integer(mapping, "last_n_blocks", path="model.encoder")
        if last_n_blocks < 0 or (train_mode == "last_n" and last_n_blocks == 0):
            raise SemanticConfigurationError(
                "model.encoder.last_n_blocks is invalid for train_mode."
            )
        return cls(
            name=name,
            repository_path=resolver.resolve(
                PathRole.EXTERNAL_ASSET,
                _string(mapping, "repository_path", path="model.encoder"),
            ),
            checkpoint_path=resolver.resolve_symlink_entry(
                PathRole.EXTERNAL_ASSET,
                _string(mapping, "checkpoint_path", path="model.encoder"),
            ),
            backbone_name=_string(mapping, "backbone_name", path="model.encoder"),
            strict=_bool(mapping, "strict", path="model.encoder"),
            train_mode=train_mode,
            last_n_blocks=last_n_blocks,
            out_indices=out_indices,
            layer_mode=layer_mode,
            lora=CourtLoRAConfig.from_mapping(
                require_config_mapping(mapping, "lora", path="model.encoder")
            ),
        )


@dataclass(frozen=True, slots=True)
class CourtDecoderConfig:
    name: str
    channels: int | tuple[int, ...]
    reassemble_factors: tuple[float, ...] | None

    @classmethod
    def from_mapping(cls, value: object) -> CourtDecoderConfig:
        mapping = as_config_mapping(value, path="model.decoder")
        name = _string(mapping, "name", path="model.decoder")
        expected = (
            {"name", "channels", "reassemble_factors"}
            if name == "dpt"
            else {"name", "channels"}
        )
        if name not in {"fpn", "unet", "dpt"}:
            raise SemanticConfigurationError(
                "model.decoder.name must be fpn, unet, or dpt."
            )
        _exact(mapping, expected, path="model.decoder")
        raw_channels = require_config_value(
            mapping, "channels", (int, list, tuple), path="model.decoder"
        )
        channels: int | tuple[int, ...]
        if type(raw_channels) is int:
            channels = raw_channels
        else:
            sequence = cast("Sequence[object]", raw_channels)
            if any(type(item) is not int for item in sequence):
                raise ConfigurationTypeError(
                    "model.decoder.channels must contain integers."
                )
            channels = tuple(cast("int", item) for item in sequence)
        if name == "dpt" and type(channels) is not int:
            raise SemanticConfigurationError("DPT decoder.channels must be an integer.")
        if name != "dpt" and not isinstance(channels, tuple):
            raise SemanticConfigurationError(
                "FPN/U-Net decoder.channels must be a sequence."
            )
        if (isinstance(channels, int) and channels <= 0) or (
            isinstance(channels, tuple)
            and (len(channels) != 4 or any(channel <= 0 for channel in channels))
        ):
            raise SemanticConfigurationError(
                "model.decoder.channels must define positive channel counts."
            )
        factors = (
            _float_tuple(mapping, "reassemble_factors", 4, path="model.decoder")
            if name == "dpt"
            else None
        )
        return cls(name=name, channels=channels, reassemble_factors=factors)


@dataclass(frozen=True, slots=True)
class CourtModelConfig:
    name: str
    in_channels: int
    encoder: CourtEncoderConfig
    decoder: CourtDecoderConfig

    @classmethod
    def from_mapping(
        cls, value: object, *, resolver: PathResolver
    ) -> CourtModelConfig:
        mapping = as_config_mapping(value, path="model")
        _exact(
            mapping,
            {"name", "in_channels", "encoder", "decoder"},
            path="model",
        )
        name = _string(mapping, "name", path="model")
        if name != "court_hierarchical":
            raise SemanticConfigurationError("model.name must be 'court_hierarchical'.")
        result = cls(
            name=name,
            in_channels=_integer(mapping, "in_channels", path="model"),
            encoder=CourtEncoderConfig.from_mapping(
                require_config_mapping(mapping, "encoder", path="model"),
                resolver=resolver,
            ),
            decoder=CourtDecoderConfig.from_mapping(
                require_config_mapping(mapping, "decoder", path="model")
            ),
        )
        if result.in_channels <= 0:
            raise SemanticConfigurationError("model.in_channels must be positive.")
        if result.decoder.name == "dpt" and result.encoder.name != "dinov3":
            raise SemanticConfigurationError("DPT decoder requires the DINOv3 encoder.")
        return result


@dataclass(frozen=True, slots=True)
class CourtLossConfig:
    seg_ce_weight: float
    seg_dice_weight: float
    kp_focal_gamma: float
    kp_positive_weight: float
    line_bce_weight: float
    line_dice_weight: float
    line_pos_weight: float

    @classmethod
    def from_mapping(cls, value: object) -> CourtLossConfig:
        mapping = as_config_mapping(value, path="loss")
        _exact(mapping, {"seg", "kp", "line"}, path="loss")
        seg = require_config_mapping(mapping, "seg", path="loss")
        kp = require_config_mapping(mapping, "kp", path="loss")
        line = require_config_mapping(mapping, "line", path="loss")
        _exact(seg, {"ce_weight", "dice_weight"}, path="loss.seg")
        _exact(kp, {"focal_gamma", "positive_weight"}, path="loss.kp")
        _exact(line, {"bce_weight", "dice_weight", "pos_weight"}, path="loss.line")
        result = cls(
            seg_ce_weight=_number(seg, "ce_weight", path="loss.seg"),
            seg_dice_weight=_number(seg, "dice_weight", path="loss.seg"),
            kp_focal_gamma=_number(kp, "focal_gamma", path="loss.kp"),
            kp_positive_weight=_number(kp, "positive_weight", path="loss.kp"),
            line_bce_weight=_number(line, "bce_weight", path="loss.line"),
            line_dice_weight=_number(line, "dice_weight", path="loss.line"),
            line_pos_weight=_number(line, "pos_weight", path="loss.line"),
        )
        if any(
            value < 0.0
            for value in (
                result.seg_ce_weight,
                result.seg_dice_weight,
                result.kp_focal_gamma,
                result.line_bce_weight,
                result.line_dice_weight,
            )
        ):
            raise SemanticConfigurationError(
                "Court loss weights and focal gamma must be non-negative."
            )
        if result.line_pos_weight <= 0.0:
            raise SemanticConfigurationError("loss.line.pos_weight must be positive.")
        if result.kp_positive_weight <= 0.0:
            raise SemanticConfigurationError(
                "loss.kp.positive_weight must be positive."
            )
        if result.seg_ce_weight == 0.0 and result.seg_dice_weight == 0.0:
            raise SemanticConfigurationError(
                "loss.seg must enable ce_weight or dice_weight."
            )
        if result.line_bce_weight == 0.0 and result.line_dice_weight == 0.0:
            raise SemanticConfigurationError(
                "loss.line must enable bce_weight or dice_weight."
            )
        return result


@dataclass(frozen=True, slots=True)
class CourtTrainingConfig:
    shared: TrainingRuntimeConfig
    data: CourtDataConfig
    model: CourtModelConfig
    loss: CourtLossConfig
    render_style: CourtRenderConfig
    qualitative_fps: int

    @classmethod
    def from_config(cls, value: object) -> CourtTrainingConfig:
        config = as_config_mapping(value, path="configuration")
        _exact(
            config,
            {"paths", "run", "training", "data", "model", "loss", "render_style"},
            path="configuration",
        )
        run_mapping = require_config_mapping(config, "run", path="configuration")
        _exact(
            run_mapping,
            {
                "output_dir",
                "seed",
                "gpus",
                "resume",
                "init_weights",
                "fast_dev_run",
                "dry_run",
                "test_after_fit",
            },
            path="run",
        )
        training_mapping = require_config_mapping(
            config, "training", path="configuration"
        )
        _exact(
            training_mapping,
            {
                "trainer",
                "learning_rate",
                "weight_decay",
                "warmup_epochs",
                "warmup_steps",
                "min_lr",
                "steps_per_epoch",
                "optimizer",
                "matmul_precision",
                "allow_tf32",
                "checkpoint",
                "early_stopping",
                "lr_monitor",
                "qualitative_logging",
                "gan",
            },
            path="training",
        )
        nested_keys = {
            "trainer": {
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
            "optimizer": {"name", "betas"},
            "checkpoint": {
                "enabled",
                "filename",
                "monitor",
                "mode",
                "save_top_k",
                "save_last",
            },
            "early_stopping": {
                "enabled",
                "monitor",
                "mode",
                "patience",
                "min_delta",
                "check_on_train_epoch_end",
            },
            "lr_monitor": {"enabled", "interval"},
            "qualitative_logging": {
                "enabled",
                "every_n_epochs",
                "num_samples",
                "selection_mode",
                "selected_indices",
                "fps",
            },
            "gan": {
                "enabled",
                "target_weight",
                "warmup_epochs",
                "generator_gradient_clip_val",
                "discriminator_gradient_clip_val",
                "transition",
            },
        }
        for section, keys in nested_keys.items():
            _exact(
                require_config_mapping(training_mapping, section, path="training"),
                keys,
                path=f"training.{section}",
            )
        gan = require_config_mapping(training_mapping, "gan", path="training")
        _exact(
            require_config_mapping(gan, "transition", path="training.gan"),
            {"start_epoch"},
            path="training.gan.transition",
        )
        optimizer = require_config_mapping(
            training_mapping, "optimizer", path="training"
        )
        if _string(optimizer, "name", path="training.optimizer") != "adamw":
            raise SemanticConfigurationError(
                "training.optimizer.name must be 'adamw'."
            )
        qualitative = require_config_mapping(
            training_mapping, "qualitative_logging", path="training"
        )
        _bool(qualitative, "enabled", path="training.qualitative_logging")
        for key in ("every_n_epochs", "num_samples"):
            if _integer(qualitative, key, path="training.qualitative_logging") <= 0:
                raise SemanticConfigurationError(
                    f"training.qualitative_logging.{key} must be positive."
                )
        selection_mode = _string(
            qualitative, "selection_mode", path="training.qualitative_logging"
        )
        if selection_mode not in {"first", "random", "indices"}:
            raise SemanticConfigurationError(
                "training.qualitative_logging.selection_mode is invalid."
            )
        selected_raw = require_config_value(
            qualitative,
            "selected_indices",
            (list, tuple, type(None)),
            path="training.qualitative_logging",
        )
        selected_indices = (
            ()
            if selected_raw is None
            else _int_tuple(
                qualitative, "selected_indices", path="training.qualitative_logging"
            )
        )
        if any(index < 0 for index in selected_indices):
            raise SemanticConfigurationError(
                "training.qualitative_logging.selected_indices must be non-negative."
            )
        if selection_mode == "indices" and not selected_indices:
            raise SemanticConfigurationError(
                "indices selection requires non-empty selected_indices."
            )
        shared_training = dict(training_mapping)
        shared_training["qualitative_logging"] = {
            key: item for key, item in qualitative.items() if key != "fps"
        }
        shared_config = dict(config)
        shared_config["training"] = shared_training
        shared = TrainingRuntimeConfig.from_config(
            shared_config, repository_root=PROJECT_ROOT
        )
        data = CourtDataConfig.from_mapping(
            require_config_mapping(config, "data", path="configuration"),
            resolver=shared.resolver,
        )
        model = CourtModelConfig.from_mapping(
            require_config_mapping(config, "model", path="configuration"),
            resolver=shared.resolver,
        )
        loss = CourtLossConfig.from_mapping(
            require_config_mapping(config, "loss", path="configuration")
        )
        render_style = CourtRenderConfig.from_mapping(
            require_config_mapping(config, "render_style", path="configuration")
        )
        qualitative_fps = _integer(
            qualitative, "fps", path="training.qualitative_logging"
        )
        if qualitative_fps <= 0:
            raise SemanticConfigurationError(
                "training.qualitative_logging.fps must be positive."
            )
        if (
            model.encoder.lora is not None
            and model.encoder.lora.enabled
            and model.encoder.train_mode != "frozen"
        ):
            raise SemanticConfigurationError(
                "LoRA requires model.encoder.train_mode=frozen."
            )
        return cls(
            shared=shared,
            data=data,
            model=model,
            loss=loss,
            render_style=render_style,
            qualitative_fps=qualitative_fps,
        )


def validate_train_boundary(config: DictConfig) -> None:
    CourtTrainingConfig.from_config(config)


def validate_paths_boundary(
    config: DictConfig, *, expected_sections: set[str]
) -> tuple[ConfigMapping, PathResolver]:
    mapping = as_config_mapping(config, path="configuration")
    _exact(mapping, {"paths"} | expected_sections, path="configuration")
    return mapping, _resolver(mapping)


__all__ = [
    "CourtAugmentationConfig",
    "CourtDataConfig",
    "CourtDecoderConfig",
    "CourtEncoderConfig",
    "CourtLoRAConfig",
    "CourtLossConfig",
    "CourtModelConfig",
    "CourtProcessingConfig",
    "CourtRenderConfig",
    "CourtSourceConfig",
    "CourtTargetConfig",
    "CourtTrainingConfig",
    "LINE_TARGET_SCHEMA",
    "SEGMENTATION_TARGET_SCHEMA",
    "SyntheticCourtSourceConfig",
    "TennisCourtDetectorSourceConfig",
    "validate_paths_boundary",
    "validate_train_boundary",
]

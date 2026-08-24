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
SyntheticCourtSchemaVersion: TypeAlias = Literal["v1", "v2", "v3"]
KeypointCourtScope: TypeAlias = Literal["all_courts", "target_court"]
CourtQueryPreset: TypeAlias = Literal["tiny", "small", "base", "raw"]
CourtQueryDecoderFamily: TypeAlias = Literal["linear", "progressive", "dpt"]
CourtConsistencyGradientFlow: TypeAlias = Literal[
    "both",
    "stopgrad_pose",
    "stopgrad_dense",
]

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
    preserve_fx_fy: bool
    canvas_size: int | None
    patch_size: int

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
            "preserve_fx_fy",
            "canvas_size",
            "patch_size",
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
            preserve_fx_fy=_bool(
                mapping, "preserve_fx_fy", path="data.augmentation"
            ),
            canvas_size=cast(
                "int | None",
                require_config_value(
                    mapping,
                    "canvas_size",
                    (int, type(None)),
                    path="data.augmentation",
                ),
            ),
            patch_size=_integer(mapping, "patch_size", path="data.augmentation"),
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
        if result.canvas_size is not None and result.canvas_size <= 0:
            raise SemanticConfigurationError(
                "data.augmentation.canvas_size must be null or positive."
            )
        if result.patch_size <= 0:
            raise SemanticConfigurationError(
                "data.augmentation.patch_size must be positive."
            )
        return result


@dataclass(frozen=True, slots=True)
class TennisCourtDetectorSourceConfig:
    kind: Literal["tennis_court_detector"]
    root: Path
    split_mapping: Mapping[CourtSourceSplit, str | None]

    def __post_init__(self) -> None:
        if dict(self.split_mapping) != {
            "train": "train",
            "val": "val",
            "test": None,
        }:
            raise SemanticConfigurationError(
                "TennisCourtDetector requires train->train, val->val, and test->null; "
                "validation cannot be reused as test."
            )

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
            expected_value = {"train": "train", "val": "val", "test": None}[
                split
            ]
            if value_at_split != expected_value:
                raise SemanticConfigurationError(
                    "TennisCourtDetector requires train->train, val->val, and "
                    "test->null; validation cannot be reused as test."
                )
            resolved[split] = value_at_split
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
    schema: SyntheticCourtSchemaVersion
    keypoint_court_scope: KeypointCourtScope
    workspace_root: Path
    scene_ids: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls, value: object, *, resolver: PathResolver
    ) -> SyntheticCourtSourceConfig:
        mapping = as_config_mapping(value, path="data.source")
        _exact(
            mapping,
            {
                "kind",
                "schema",
                "keypoint_court_scope",
                "workspace_root",
                "scene_ids",
            },
            path="data.source",
        )
        if _string(mapping, "kind", path="data.source") != "synthetic_court":
            raise SemanticConfigurationError(
                "data.source.kind must be 'synthetic_court'."
            )
        schema = _string(mapping, "schema", path="data.source")
        if schema not in {"v1", "v2", "v3"}:
            raise SemanticConfigurationError(
                "data.source.schema must be explicitly 'v1', 'v2', or 'v3'."
            )
        keypoint_court_scope = _string(
            mapping, "keypoint_court_scope", path="data.source"
        )
        if keypoint_court_scope not in {"all_courts", "target_court"}:
            raise SemanticConfigurationError(
                "data.source.keypoint_court_scope must be 'all_courts' or "
                "'target_court'."
            )
        if schema == "v1" and keypoint_court_scope == "target_court":
            raise SemanticConfigurationError(
                "data.source.keypoint_court_scope='target_court' requires "
                "data.source.schema='v2' or 'v3'."
            )
        raw_ids = _sequence(mapping, "scene_ids", path="data.source")
        scene_ids_list: list[str] = []
        for item in raw_ids:
            if (
                type(item) is not str
                or not item
                or item != item.strip()
                or item in {".", ".."}
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
            schema=cast(SyntheticCourtSchemaVersion, schema),
            keypoint_court_scope=cast(KeypointCourtScope, keypoint_court_scope),
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
    def from_mapping(
        cls,
        value: object,
        *,
        path: str = "model.encoder.lora",
    ) -> CourtLoRAConfig:
        mapping = as_config_mapping(value, path=path)
        _exact(
            mapping,
            {"enabled", "rank", "alpha", "dropout", "target_modules"},
            path=path,
        )
        raw_targets = _sequence(mapping, "target_modules", path=path)
        if not raw_targets or any(
            type(item) is not str or not item for item in raw_targets
        ):
            raise ConfigurationTypeError(
                f"{path}.target_modules must contain non-empty strings."
            )
        result = cls(
            enabled=_bool(mapping, "enabled", path=path),
            rank=_integer(mapping, "rank", path=path),
            alpha=_number(mapping, "alpha", path=path),
            dropout=_number(mapping, "dropout", path=path),
            target_modules=tuple(cast("str", item) for item in raw_targets),
        )
        if result.rank <= 0 or result.alpha <= 0 or not 0.0 <= result.dropout < 1.0:
            raise SemanticConfigurationError(
                f"Invalid {path} rank/alpha/dropout."
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
class CourtQueryBackboneConfig:
    """Exact DINOv3 asset and trainability contract for the query variant."""

    name: Literal["dinov3"]
    repository_path: Path
    checkpoint_path: Path
    backbone_name: str
    output_layer: Literal["normalized_patch_tokens"]
    strict: bool
    train_mode: Literal["frozen", "last_n", "full"]
    last_n_blocks: int
    lora: CourtLoRAConfig

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        resolver: PathResolver,
    ) -> CourtQueryBackboneConfig:
        path = "model.backbone"
        mapping = as_config_mapping(value, path=path)
        _exact(
            mapping,
            {
                "name",
                "repository_path",
                "checkpoint_path",
                "checkpoint_directory",
                "backbone_name",
                "output_layer",
                "strict",
                "train_mode",
                "last_n_blocks",
                "lora",
            },
            path=path,
        )
        if _string(mapping, "name", path=path) != "dinov3":
            raise SemanticConfigurationError("model.backbone.name must be 'dinov3'.")
        if _string(mapping, "checkpoint_directory", path=path) != "checkpoints":
            raise SemanticConfigurationError(
                "model.backbone.checkpoint_directory must be 'checkpoints'."
            )
        output_layer = _string(mapping, "output_layer", path=path)
        if output_layer != "normalized_patch_tokens":
            raise SemanticConfigurationError(
                "model.backbone.output_layer must be 'normalized_patch_tokens'; "
                "CLS/register tokens are not supported."
            )
        train_mode = _string(mapping, "train_mode", path=path)
        if train_mode not in {"frozen", "last_n", "full"}:
            raise SemanticConfigurationError(
                "model.backbone.train_mode must be frozen, last_n, or full."
            )
        last_n_blocks = _integer(mapping, "last_n_blocks", path=path)
        if last_n_blocks < 0 or (
            (train_mode == "last_n") != (last_n_blocks > 0)
        ):
            raise SemanticConfigurationError(
                "model.backbone.last_n_blocks must be positive exactly when "
                "train_mode='last_n'."
            )
        lora = CourtLoRAConfig.from_mapping(
            require_config_mapping(mapping, "lora", path=path),
            path="model.backbone.lora",
        )
        if lora.enabled and train_mode != "frozen":
            raise SemanticConfigurationError(
                "Query-backbone LoRA requires model.backbone.train_mode='frozen'."
            )
        return cls(
            name="dinov3",
            repository_path=resolver.resolve(
                PathRole.EXTERNAL_ASSET,
                _string(mapping, "repository_path", path=path),
            ),
            checkpoint_path=resolver.resolve_symlink_entry(
                PathRole.EXTERNAL_ASSET,
                _string(mapping, "checkpoint_path", path=path),
            ),
            backbone_name=_string(mapping, "backbone_name", path=path),
            output_layer="normalized_patch_tokens",
            strict=_bool(mapping, "strict", path=path),
            train_mode=cast(
                "Literal['frozen', 'last_n', 'full']",
                train_mode,
            ),
            last_n_blocks=last_n_blocks,
            lora=lora,
        )


@dataclass(frozen=True, slots=True)
class CourtQueryTaskEncoderConfig:
    hidden_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    rope_dim: int
    rope_theta: float
    tap_indices: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: object) -> CourtQueryTaskEncoderConfig:
        path = "model.task_encoder"
        mapping = as_config_mapping(value, path=path)
        _exact(
            mapping,
            {
                "hidden_dim",
                "depth",
                "num_heads",
                "mlp_ratio",
                "dropout",
                "rope_dim",
                "rope_theta",
                "tap_indices",
            },
            path=path,
        )
        result = cls(
            hidden_dim=_integer(mapping, "hidden_dim", path=path),
            depth=_integer(mapping, "depth", path=path),
            num_heads=_integer(mapping, "num_heads", path=path),
            mlp_ratio=_number(mapping, "mlp_ratio", path=path),
            dropout=_number(mapping, "dropout", path=path),
            rope_dim=_integer(mapping, "rope_dim", path=path),
            rope_theta=_number(mapping, "rope_theta", path=path),
            tap_indices=_int_tuple(mapping, "tap_indices", path=path),
        )
        if result.hidden_dim <= 0 or result.depth <= 0 or result.num_heads <= 0:
            raise SemanticConfigurationError(
                "model.task_encoder dimensions/depth/heads must be positive."
            )
        if result.hidden_dim % result.num_heads:
            raise SemanticConfigurationError(
                "model.task_encoder.hidden_dim must be divisible by num_heads."
            )
        head_dim = result.hidden_dim // result.num_heads
        if (
            result.rope_dim <= 0
            or result.rope_dim % 4
            or result.rope_dim > head_dim
        ):
            raise SemanticConfigurationError(
                "model.task_encoder.rope_dim must be positive, divisible by 4, "
                "and no larger than the attention head dimension."
            )
        if result.mlp_ratio <= 0.0 or not 0.0 <= result.dropout < 1.0:
            raise SemanticConfigurationError(
                "model.task_encoder.mlp_ratio must be positive and dropout in [0,1)."
            )
        if result.rope_theta <= 0.0:
            raise SemanticConfigurationError(
                "model.task_encoder.rope_theta must be positive."
            )
        _validate_query_taps(
            result.tap_indices,
            depth=result.depth,
            path="model.task_encoder.tap_indices",
        )
        return result


@dataclass(frozen=True, slots=True)
class CourtQueryLinearDecoderConfig:
    family: Literal["linear"]
    width: int
    tap_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CourtQueryProgressiveDecoderConfig:
    family: Literal["progressive"]
    width: int
    tap_indices: tuple[int, ...]
    stage_count: int


@dataclass(frozen=True, slots=True)
class CourtQueryDPTDecoderConfig:
    family: Literal["dpt"]
    width: int
    tap_indices: tuple[int, ...]
    fusion_levels: int
    reassemble_factors: tuple[float, ...]


CourtQueryDecoderConfig: TypeAlias = (
    CourtQueryLinearDecoderConfig
    | CourtQueryProgressiveDecoderConfig
    | CourtQueryDPTDecoderConfig
)


def _query_decoder_config(value: object) -> CourtQueryDecoderConfig:
    path = "model.decoder"
    mapping = as_config_mapping(value, path=path)
    family = _string(mapping, "family", path=path)
    if family not in {"linear", "progressive", "dpt"}:
        raise SemanticConfigurationError(
            "Query model.decoder.family must be linear, progressive, or dpt."
        )
    expected = {
        "linear": {"family", "width", "tap_indices"},
        "progressive": {"family", "width", "tap_indices", "stage_count"},
        "dpt": {
            "family",
            "width",
            "tap_indices",
            "fusion_levels",
            "reassemble_factors",
        },
    }[family]
    _exact(mapping, expected, path=path)
    width = _integer(mapping, "width", path=path)
    taps = _int_tuple(mapping, "tap_indices", path=path)
    if width <= 0:
        raise SemanticConfigurationError("model.decoder.width must be positive.")
    if len(set(taps)) != len(taps) or not taps or taps != tuple(sorted(taps)):
        raise SemanticConfigurationError(
            "model.decoder.tap_indices must be non-empty, unique, and increasing."
        )
    if family == "linear":
        if len(taps) != 1:
            raise SemanticConfigurationError(
                "Linear query decoder requires exactly one tap index."
            )
        return CourtQueryLinearDecoderConfig(
            family="linear",
            width=width,
            tap_indices=taps,
        )
    if family == "progressive":
        stages = _integer(mapping, "stage_count", path=path)
        if len(taps) != 1 or stages <= 0:
            raise SemanticConfigurationError(
                "Progressive query decoder requires one tap and positive stage_count."
            )
        return CourtQueryProgressiveDecoderConfig(
            family="progressive",
            width=width,
            tap_indices=taps,
            stage_count=stages,
        )
    fusion_levels = _integer(mapping, "fusion_levels", path=path)
    factors = _float_tuple(
        mapping,
        "reassemble_factors",
        len(taps),
        path=path,
    )
    if fusion_levels != len(taps):
        raise SemanticConfigurationError(
            "DPT query decoder requires fusion_levels equal to its tap count."
        )
    if any(factor <= 0.0 for factor in factors):
        raise SemanticConfigurationError(
            "model.decoder.reassemble_factors must be positive."
        )
    return CourtQueryDPTDecoderConfig(
        family="dpt",
        width=width,
        tap_indices=taps,
        fusion_levels=fusion_levels,
        reassemble_factors=factors,
    )


@dataclass(frozen=True, slots=True)
class CourtQueryHeadsConfig:
    pose_hidden_dim: int
    pose_depth: int
    dense_targets: tuple[CourtTargetKind, ...]

    @classmethod
    def from_mapping(cls, value: object) -> CourtQueryHeadsConfig:
        path = "model.heads"
        mapping = as_config_mapping(value, path=path)
        _exact(
            mapping,
            {"pose_hidden_dim", "pose_depth", "dense_targets"},
            path=path,
        )
        raw_targets = _sequence(mapping, "dense_targets", path=path)
        if not raw_targets or any(
            type(item) is not str or item not in {"kp", "seg", "line"}
            for item in raw_targets
        ):
            raise SemanticConfigurationError(
                "model.heads.dense_targets must be a non-empty kp/seg/line subset."
            )
        dense_targets = tuple(cast(CourtTargetKind, item) for item in raw_targets)
        if len(set(dense_targets)) != len(dense_targets):
            raise SemanticConfigurationError(
                "model.heads.dense_targets must not contain duplicates."
            )
        result = cls(
            pose_hidden_dim=_integer(mapping, "pose_hidden_dim", path=path),
            pose_depth=_integer(mapping, "pose_depth", path=path),
            dense_targets=dense_targets,
        )
        if result.pose_hidden_dim <= 0 or result.pose_depth <= 0:
            raise SemanticConfigurationError(
                "model.heads pose_hidden_dim and pose_depth must be positive."
            )
        return result


@dataclass(frozen=True, slots=True)
class CourtQueryModelConfig:
    """Strict discriminated configuration for the additive query ablation."""

    name: Literal["court_query_encoder"]
    preset: CourtQueryPreset
    in_channels: int
    backbone: CourtQueryBackboneConfig
    task_encoder: CourtQueryTaskEncoderConfig
    decoder: CourtQueryDecoderConfig
    heads: CourtQueryHeadsConfig

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        resolver: PathResolver,
    ) -> CourtQueryModelConfig:
        path = "model"
        mapping = as_config_mapping(value, path=path)
        _exact(
            mapping,
            {
                "name",
                "preset",
                "in_channels",
                "backbone",
                "task_encoder",
                "decoder",
                "heads",
            },
            path=path,
        )
        if _string(mapping, "name", path=path) != "court_query_encoder":
            raise SemanticConfigurationError(
                "CourtQueryModelConfig requires model.name='court_query_encoder'."
            )
        preset = _string(mapping, "preset", path=path)
        if preset not in {"tiny", "small", "base", "raw"}:
            raise SemanticConfigurationError(
                "model.preset must be tiny, small, base, or raw."
            )
        result = cls(
            name="court_query_encoder",
            preset=cast(CourtQueryPreset, preset),
            in_channels=_integer(mapping, "in_channels", path=path),
            backbone=CourtQueryBackboneConfig.from_mapping(
                require_config_mapping(mapping, "backbone", path=path),
                resolver=resolver,
            ),
            task_encoder=CourtQueryTaskEncoderConfig.from_mapping(
                require_config_mapping(mapping, "task_encoder", path=path)
            ),
            decoder=_query_decoder_config(
                require_config_mapping(mapping, "decoder", path=path)
            ),
            heads=CourtQueryHeadsConfig.from_mapping(
                require_config_mapping(mapping, "heads", path=path)
            ),
        )
        if result.in_channels != 3:
            raise SemanticConfigurationError(
                "Court query model requires exactly three RGB input channels."
            )
        decoder_taps = result.decoder.tap_indices
        if not set(decoder_taps).issubset(result.task_encoder.tap_indices):
            raise SemanticConfigurationError(
                "model.decoder.tap_indices must be declared by "
                "model.task_encoder.tap_indices."
            )
        if result.preset != "raw":
            _validate_query_preset(result)
        return result


CourtAnyModelConfig: TypeAlias = CourtModelConfig | CourtQueryModelConfig


def _validate_query_taps(
    taps: tuple[int, ...],
    *,
    depth: int,
    path: str,
) -> None:
    if not taps or len(set(taps)) != len(taps) or taps != tuple(sorted(taps)):
        raise SemanticConfigurationError(
            f"{path} must be non-empty, unique, and increasing."
        )
    if any(index < 0 or index >= depth for index in taps):
        raise SemanticConfigurationError(
            f"{path} must be within [0, {depth - 1}]."
        )


def _validate_query_preset(config: CourtQueryModelConfig) -> None:
    preset = cast(Literal["tiny", "small", "base"], config.preset)
    expected = {
        "tiny": {
            "task": (64, 2, 4, 2.0, 16, (0, 1)),
            "decoder_width": 32,
            "progressive_stages": 1,
            "dpt": ((0, 1), (2.0, 1.0)),
            "heads": (64, 2),
        },
        "small": {
            "task": (128, 4, 8, 3.0, 16, (0, 1, 2, 3)),
            "decoder_width": 64,
            "progressive_stages": 2,
            "dpt": ((0, 1, 2, 3), (4.0, 2.0, 1.0, 0.5)),
            "heads": (128, 3),
        },
        "base": {
            "task": (256, 8, 8, 4.0, 32, (1, 3, 5, 7)),
            "decoder_width": 128,
            "progressive_stages": 3,
            "dpt": ((1, 3, 5, 7), (4.0, 2.0, 1.0, 0.5)),
            "heads": (256, 4),
        },
    }[preset]
    task = config.task_encoder
    actual_task = (
        task.hidden_dim,
        task.depth,
        task.num_heads,
        task.mlp_ratio,
        task.rope_dim,
        task.tap_indices,
    )
    mismatched = actual_task != expected["task"]
    decoder = config.decoder
    mismatched = mismatched or decoder.width != expected["decoder_width"]
    task_taps = cast("tuple[int, ...]", cast("tuple[object, ...]", expected["task"])[-1])
    final_tap = task_taps[-1]
    if decoder.family in {"linear", "progressive"}:
        mismatched = mismatched or decoder.tap_indices != (final_tap,)
    if isinstance(decoder, CourtQueryProgressiveDecoderConfig):
        mismatched = mismatched or (
            decoder.stage_count != expected["progressive_stages"]
        )
    if isinstance(decoder, CourtQueryDPTDecoderConfig):
        dpt_taps, dpt_factors = cast(
            "tuple[tuple[int, ...], tuple[float, ...]]",
            expected["dpt"],
        )
        mismatched = mismatched or (
            decoder.tap_indices != dpt_taps
            or decoder.fusion_levels != len(dpt_taps)
            or decoder.reassemble_factors != dpt_factors
        )
    mismatched = mismatched or (
        (config.heads.pose_hidden_dim, config.heads.pose_depth)
        != expected["heads"]
    )
    if mismatched:
        raise SemanticConfigurationError(
            f"model preset {preset!r} disagrees with its raw component values; "
            "set model.preset=raw for independent raw overrides."
        )


def _model_config(value: object, *, resolver: PathResolver) -> CourtAnyModelConfig:
    mapping = as_config_mapping(value, path="model")
    name = _string(mapping, "name", path="model")
    if name == "court_hierarchical":
        return CourtModelConfig.from_mapping(mapping, resolver=resolver)
    if name == "court_query_encoder":
        return CourtQueryModelConfig.from_mapping(mapping, resolver=resolver)
    raise SemanticConfigurationError(
        "model.name must be 'court_hierarchical' or 'court_query_encoder'."
    )


@dataclass(frozen=True, slots=True)
class CourtLossConfig:
    seg_ce_weight: float
    seg_dice_weight: float
    kp_focal_gamma: float
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
        _exact(kp, {"focal_gamma"}, path="loss.kp")
        _exact(line, {"bce_weight", "dice_weight", "pos_weight"}, path="loss.line")
        result = cls(
            seg_ce_weight=_number(seg, "ce_weight", path="loss.seg"),
            seg_dice_weight=_number(seg, "dice_weight", path="loss.seg"),
            kp_focal_gamma=_number(kp, "focal_gamma", path="loss.kp"),
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
class CourtQueryPoseLossConfig:
    enabled: bool
    translation_weight: float
    rotation_weight: float
    focal_weight: float


@dataclass(frozen=True, slots=True)
class CourtQueryConsistencyLossConfig:
    """Strict query-only KP/predicted-pose consistency configuration."""

    enabled: bool
    weight: float
    temperature: float
    huber_delta: float
    min_depth_m: float
    depth_scale_m: float
    cheirality_weight: float
    warmup_fraction: float
    gradient_flow: CourtConsistencyGradientFlow

    @classmethod
    def from_mapping(cls, value: object) -> CourtQueryConsistencyLossConfig:
        mapping = as_config_mapping(value, path="loss.consistency")
        _exact(
            mapping,
            {
                "enabled",
                "weight",
                "temperature",
                "huber_delta",
                "min_depth_m",
                "depth_scale_m",
                "cheirality_weight",
                "warmup_fraction",
                "gradient_flow",
            },
            path="loss.consistency",
        )
        raw_gradient_flow = _string(
            mapping,
            "gradient_flow",
            path="loss.consistency",
        )
        if raw_gradient_flow not in {
            "both",
            "stopgrad_pose",
            "stopgrad_dense",
        }:
            raise SemanticConfigurationError(
                "loss.consistency.gradient_flow must be 'both', "
                "'stopgrad_pose', or 'stopgrad_dense'."
            )
        result = cls(
            enabled=_bool(mapping, "enabled", path="loss.consistency"),
            weight=_number(mapping, "weight", path="loss.consistency"),
            temperature=_number(
                mapping,
                "temperature",
                path="loss.consistency",
            ),
            huber_delta=_number(
                mapping,
                "huber_delta",
                path="loss.consistency",
            ),
            min_depth_m=_number(
                mapping,
                "min_depth_m",
                path="loss.consistency",
            ),
            depth_scale_m=_number(
                mapping,
                "depth_scale_m",
                path="loss.consistency",
            ),
            cheirality_weight=_number(
                mapping,
                "cheirality_weight",
                path="loss.consistency",
            ),
            warmup_fraction=_number(
                mapping,
                "warmup_fraction",
                path="loss.consistency",
            ),
            gradient_flow=cast(
                "CourtConsistencyGradientFlow",
                raw_gradient_flow,
            ),
        )
        positive_values = (
            result.temperature,
            result.huber_delta,
            result.min_depth_m,
            result.depth_scale_m,
        )
        if any(item <= 0.0 for item in positive_values):
            raise SemanticConfigurationError(
                "loss.consistency temperature, huber_delta, min_depth_m, and "
                "depth_scale_m must be positive."
            )
        if result.cheirality_weight < 0.0:
            raise SemanticConfigurationError(
                "loss.consistency.cheirality_weight must be non-negative."
            )
        if not 0.0 <= result.warmup_fraction < 1.0:
            raise SemanticConfigurationError(
                "loss.consistency.warmup_fraction must be in [0, 1)."
            )
        if result.enabled and result.weight <= 0.0:
            raise SemanticConfigurationError(
                "Enabled query consistency requires a positive weight."
            )
        if not result.enabled and (
            result.weight != 0.0
            or result.cheirality_weight != 0.0
            or result.warmup_fraction != 0.0
        ):
            raise SemanticConfigurationError(
                "Disabled query consistency requires zero weight, "
                "cheirality_weight, and warmup_fraction."
            )
        return result

    @classmethod
    def legacy_disabled(cls) -> CourtQueryConsistencyLossConfig:
        """Represent the two frozen direct-only query v1 loss schemas."""
        return cls(
            enabled=False,
            weight=0.0,
            temperature=1.0,
            huber_delta=0.01,
            min_depth_m=0.1,
            depth_scale_m=1.0,
            cheirality_weight=0.0,
            warmup_fraction=0.0,
            gradient_flow="both",
        )


@dataclass(frozen=True, slots=True)
class CourtQueryLossConfig:
    """Explicit dense and pose weights for query-model supervision."""

    name: str
    dense: CourtLossConfig
    dense_weights: Mapping[CourtTargetKind, float]
    pose: CourtQueryPoseLossConfig
    consistency: CourtQueryConsistencyLossConfig

    @classmethod
    def from_mapping(cls, value: object) -> CourtQueryLossConfig:
        mapping = as_config_mapping(value, path="loss")
        name = _string(mapping, "name", path="loss")
        has_consistency = "consistency" in mapping
        _exact(
            mapping,
            {"name", "seg", "kp", "line", "pose"}
            | ({"consistency"} if has_consistency else set()),
            path="loss",
        )
        if not has_consistency and name not in {"query_pose_v1", "query_dense_v1"}:
            raise MissingConfigurationKeyError(
                "Missing required configuration key: loss.consistency."
            )
        seg = require_config_mapping(mapping, "seg", path="loss")
        kp = require_config_mapping(mapping, "kp", path="loss")
        line = require_config_mapping(mapping, "line", path="loss")
        pose = require_config_mapping(mapping, "pose", path="loss")
        _exact(seg, {"ce_weight", "dice_weight", "weight"}, path="loss.seg")
        _exact(kp, {"focal_gamma", "weight"}, path="loss.kp")
        _exact(
            line,
            {"bce_weight", "dice_weight", "pos_weight", "weight"},
            path="loss.line",
        )
        _exact(
            pose,
            {"enabled", "translation_weight", "rotation_weight", "focal_weight"},
            path="loss.pose",
        )
        dense = CourtLossConfig.from_mapping(
            {
                "seg": {
                    "ce_weight": seg["ce_weight"],
                    "dice_weight": seg["dice_weight"],
                },
                "kp": {"focal_gamma": kp["focal_gamma"]},
                "line": {
                    "bce_weight": line["bce_weight"],
                    "dice_weight": line["dice_weight"],
                    "pos_weight": line["pos_weight"],
                },
            }
        )
        dense_weights: dict[CourtTargetKind, float] = {
            "kp": _number(kp, "weight", path="loss.kp"),
            "seg": _number(seg, "weight", path="loss.seg"),
            "line": _number(line, "weight", path="loss.line"),
        }
        if any(weight <= 0.0 for weight in dense_weights.values()):
            raise SemanticConfigurationError(
                "Query dense loss weights must all be positive."
            )
        pose_config = CourtQueryPoseLossConfig(
            enabled=_bool(pose, "enabled", path="loss.pose"),
            translation_weight=_number(
                pose, "translation_weight", path="loss.pose"
            ),
            rotation_weight=_number(pose, "rotation_weight", path="loss.pose"),
            focal_weight=_number(pose, "focal_weight", path="loss.pose"),
        )
        pose_weights = (
            pose_config.translation_weight,
            pose_config.rotation_weight,
            pose_config.focal_weight,
        )
        if pose_config.enabled and any(weight <= 0.0 for weight in pose_weights):
            raise SemanticConfigurationError(
                "Enabled query pose supervision requires positive translation, "
                "rotation, and focal weights."
            )
        if not pose_config.enabled and any(weight != 0.0 for weight in pose_weights):
            raise SemanticConfigurationError(
                "Disabled query pose supervision requires zero pose weights."
            )
        consistency = (
            CourtQueryConsistencyLossConfig.from_mapping(mapping["consistency"])
            if has_consistency
            else CourtQueryConsistencyLossConfig.legacy_disabled()
        )
        if consistency.enabled and not pose_config.enabled:
            raise SemanticConfigurationError(
                "Enabled query consistency requires enabled query pose supervision."
            )
        return cls(
            name=name,
            dense=dense,
            dense_weights=MappingProxyType(dense_weights),
            pose=pose_config,
            consistency=consistency,
        )


CourtAnyLossConfig: TypeAlias = CourtLossConfig | CourtQueryLossConfig


@dataclass(frozen=True, slots=True)
class CourtTrainingConfig:
    shared: TrainingRuntimeConfig
    data: CourtDataConfig
    model: CourtAnyModelConfig
    loss: CourtAnyLossConfig
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
                "compile",
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
            "compile": {"enabled", "backend", "mode", "fullgraph", "dynamic"},
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
        if selected_raw is None:
            selected_indices: tuple[int, ...] = ()
        else:
            selected_indices = _int_tuple(
                qualitative, "selected_indices", path="training.qualitative_logging"
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
        model = _model_config(
            require_config_mapping(config, "model", path="configuration"),
            resolver=shared.resolver,
        )
        loss_mapping = require_config_mapping(config, "loss", path="configuration")
        loss: CourtAnyLossConfig = (
            CourtQueryLossConfig.from_mapping(loss_mapping)
            if isinstance(model, CourtQueryModelConfig)
            else CourtLossConfig.from_mapping(loss_mapping)
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
        if isinstance(model, CourtModelConfig) and (
            model.encoder.lora is not None
            and model.encoder.lora.enabled
            and model.encoder.train_mode != "frozen"
        ):
            raise SemanticConfigurationError(
                "LoRA requires model.encoder.train_mode=frozen."
            )
        if isinstance(model, CourtQueryModelConfig):
            configured_targets = tuple(
                target.kind for target in data.processing.targets
            )
            if model.heads.dense_targets != configured_targets:
                raise SemanticConfigurationError(
                    "model.heads.dense_targets must exactly match the ordered "
                    "data.processing.targets for the query variant."
                )
            if not isinstance(loss, CourtQueryLossConfig):  # pragma: no cover
                raise TypeError("Query model requires CourtQueryLossConfig.")
            source = data.source
            if not isinstance(source, SyntheticCourtSourceConfig) or source.schema != "v3":
                raise SemanticConfigurationError(
                    "Court query model requires data.source Synthetic Court V3."
                )
            if source.keypoint_court_scope != "target_court":
                raise SemanticConfigurationError(
                    "Court query model requires data.source.keypoint_court_scope="
                    "'target_court'."
                )
            if "kp" not in configured_targets:
                raise SemanticConfigurationError(
                    "Court query model requires singleton target-court KP14 supervision."
                )
            _validate_pose_safe_augmentation(data.augmentation)
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


def _validate_pose_safe_augmentation(config: CourtAugmentationConfig) -> None:
    if not config.preserve_fx_fy:
        raise SemanticConfigurationError(
            "Pose supervision requires data.augmentation.preserve_fx_fy=true."
        )
    if config.canvas_size is not None:
        raise SemanticConfigurationError(
            "Pose-safe augmentation must not use a square canvas_size; "
            "padding is restricted to patch alignment."
        )
    unsupported = (
        config.crop_scale != (1.0, 1.0)
        or config.crop_ratio != (1.0, 1.0)
        or config.hflip_prob != 0.0
        or config.affine_degrees != 0.0
        or config.affine_translate != (0.0, 0.0)
        or config.affine_scale != (1.0, 1.0)
        or config.affine_shear != 0.0
        or config.perspective_distortion != 0.0
        or config.perspective_prob != 0.0
    )
    if unsupported:
        raise SemanticConfigurationError(
            "Pose supervision rejects horizontal flip, random-resized crop, unequal "
            "axes, affine, shear, and perspective transforms."
        )
    if len(config.train_scales) != 1 or config.train_scales[0] != config.val_short_side:
        raise SemanticConfigurationError(
            "Pose-safe train_scales must contain exactly the validation long-side size."
        )


def validate_paths_boundary(
    config: DictConfig, *, expected_sections: set[str]
) -> tuple[ConfigMapping, PathResolver]:
    mapping = as_config_mapping(config, path="configuration")
    _exact(mapping, {"paths"} | expected_sections, path="configuration")
    return mapping, _resolver(mapping)


__all__ = [
    "CourtConsistencyGradientFlow",
    "CourtAnyModelConfig",
    "CourtAugmentationConfig",
    "CourtAnyLossConfig",
    "CourtDataConfig",
    "CourtDecoderConfig",
    "CourtEncoderConfig",
    "CourtLoRAConfig",
    "CourtLossConfig",
    "CourtModelConfig",
    "CourtQueryBackboneConfig",
    "CourtQueryDPTDecoderConfig",
    "CourtQueryDecoderConfig",
    "CourtQueryDecoderFamily",
    "CourtQueryHeadsConfig",
    "CourtQueryLinearDecoderConfig",
    "CourtQueryConsistencyLossConfig",
    "CourtQueryLossConfig",
    "CourtQueryModelConfig",
    "CourtQueryPreset",
    "CourtQueryProgressiveDecoderConfig",
    "CourtQueryPoseLossConfig",
    "CourtQueryTaskEncoderConfig",
    "CourtProcessingConfig",
    "CourtRenderConfig",
    "CourtSourceConfig",
    "CourtTargetConfig",
    "CourtTrainingConfig",
    "KeypointCourtScope",
    "LINE_TARGET_SCHEMA",
    "SEGMENTATION_TARGET_SCHEMA",
    "SyntheticCourtSourceConfig",
    "SyntheticCourtSchemaVersion",
    "TennisCourtDetectorSourceConfig",
    "validate_paths_boundary",
    "validate_train_boundary",
]

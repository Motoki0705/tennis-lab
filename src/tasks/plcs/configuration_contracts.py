"""Shared PLCS configuration value objects below runtime boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from src.tasks.base.configuration import (
    as_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.tasks.base.generate_dataset.timeline_composer import TimelineConfig
from src.utils.configuration import (
    ConfigurationTypeError,
    PathResolver,
    RuntimePathRoots,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.configuration.paths import PathRole
from src.utils.paths import PROJECT_ROOT

__all__ = ["PLCSGenerationComponents", "PLCSPathConfig"]


def _reject_unknown(
    mapping: object,
    allowed: set[str],
    *,
    path: str,
) -> Mapping[str, object]:
    resolved: Mapping[str, object] = as_config_mapping(mapping, path=path)
    unknown = sorted(set(resolved) - allowed)
    if unknown:
        raise UnknownConfigurationKeyError(
            f"Unknown configuration key(s): "
            f"{', '.join(f'{path}.{key}' for key in unknown)}."
        )
    return resolved


def _number(mapping: Mapping[str, object], key: str, *, path: str) -> float:
    value = float(
        cast(
            "float | int",
            require_config_value(mapping, key, (float, int), path=path),
        )
    )
    if not math.isfinite(value):
        raise SemanticConfigurationError(f"{path}.{key} must be finite.")
    return value


def _sequence(
    mapping: Mapping[str, object],
    key: str,
    *,
    path: str,
    item_types: tuple[type[object], ...],
    length: int | None = None,
) -> tuple[object, ...]:
    raw = require_config_value(mapping, key, (list, tuple), path=path)
    values = tuple(cast("Sequence[object]", raw))
    if not values:
        raise SemanticConfigurationError(f"{path}.{key} must not be empty.")
    if length is not None and len(values) != length:
        raise ConfigurationTypeError(
            f"{path}.{key} must contain exactly {length} values."
        )
    if any(type(item) not in item_types for item in values):
        raise ConfigurationTypeError(
            f"{path}.{key} contains a value with an invalid exact type."
        )
    if item_types in {(float, int), (int, float)} and any(
        not math.isfinite(float(cast("float | int", item))) for item in values
    ):
        raise SemanticConfigurationError(f"{path}.{key} values must be finite.")
    return values


def _ordered_range(
    mapping: Mapping[str, object],
    key: str,
    *,
    path: str,
    positive: bool = False,
    upper_bound: float | None = None,
) -> tuple[float, float]:
    values = _sequence(
        mapping,
        key,
        path=path,
        item_types=(float, int),
        length=2,
    )
    low, high = (float(cast("float | int", item)) for item in values)
    if low > high:
        raise SemanticConfigurationError(
            f"{path}.{key} must be ordered low-to-high."
        )
    if positive and low <= 0.0:
        raise SemanticConfigurationError(f"{path}.{key} values must be positive.")
    if upper_bound is not None and high > upper_bound:
        raise SemanticConfigurationError(
            f"{path}.{key} values must be <= {upper_bound}."
        )
    return low, high


@dataclass(frozen=True, slots=True)
class PLCSPathConfig:
    """Seven shared runtime roots for every PLCS execution boundary."""

    resolver: PathResolver

    @classmethod
    def from_config(cls, value: object) -> PLCSPathConfig:
        root = as_config_mapping(value, path="configuration")
        paths = require_config_mapping(root, "paths", path="configuration")
        roots = RuntimePathRoots.from_mapping(paths, repository_root=PROJECT_ROOT)
        return cls(resolver=PathResolver(roots))


def _validate_generation_mode(root: Mapping[str, object]) -> str:
    mode_config = require_config_mapping(root, "generation", path="configuration")
    mode = cast(
        "str", require_config_value(mode_config, "mode", str, path="generation")
    )
    if mode not in {"single_object", "multi_object"}:
        raise SemanticConfigurationError(
            "generation.mode must be 'single_object' or 'multi_object'."
        )
    mode_config = _reject_unknown(
        mode_config,
        (
            {"mode", "min_persons", "max_persons"}
            if mode == "single_object"
            else {"mode", "timeline"}
        ),
        path="generation",
    )
    if mode == "single_object":
        min_persons = cast(
            "int",
            require_config_value(mode_config, "min_persons", int, path="generation"),
        )
        max_persons = cast(
            "int",
            require_config_value(mode_config, "max_persons", int, path="generation"),
        )
        if min_persons < 1 or max_persons < min_persons:
            raise SemanticConfigurationError(
                "generation min/max persons must define a positive ordered range."
            )
        return mode

    timeline_fields = {
        "num_frames",
        "min_tracks",
        "max_tracks",
        "max_concurrent",
        "min_reuse_gap_frames",
        "start_index_range",
        "min_active_frames",
        "overlap_probability",
        "min_gap_frames",
        "max_gap_frames",
    }
    timeline = _reject_unknown(
        require_config_mapping(mode_config, "timeline", path="generation"),
        timeline_fields,
        path="generation.timeline",
    )
    for key in timeline_fields - {"start_index_range", "overlap_probability"}:
        require_config_value(timeline, key, int, path="generation.timeline")
    _sequence(
        timeline,
        "start_index_range",
        path="generation.timeline",
        item_types=(int,),
        length=2,
    )
    require_config_value(
        timeline,
        "overlap_probability",
        float,
        path="generation.timeline",
    )
    try:
        TimelineConfig.from_mapping(timeline)
    except ValueError as error:
        raise SemanticConfigurationError(str(error)) from error
    return mode


def _validate_camera(root: Mapping[str, object]) -> None:
    camera = require_config_mapping(root, "camera", path="configuration")
    camera_fields = {
        "layout",
        "z_min",
        "z_max",
        "hfov_deg",
        "image_size",
        "fixed_look_at",
        "fixed_baseline_clear_extra",
        "fixed_position_noise_radius",
        "fixed_look_at_xy_radius",
        "broadcast_setback",
        "broadcast_height",
        "broadcast_hfov_deg",
        "broadcast_look_at_y",
        "broadcast_look_at_height",
        "broadcast_position_noise_radius",
        "broadcast_look_at_xy_radius",
        "broadcast_hfov_jitter_deg",
        "broadcast_setback_range",
        "broadcast_height_range",
        "broadcast_court_width_frac_range",
    }
    camera = _reject_unknown(camera, camera_fields, path="camera")
    layout = cast("str", require_config_value(camera, "layout", str, path="camera"))
    if layout not in {"fixed", "broadcast"}:
        raise SemanticConfigurationError(
            "camera.layout must be 'fixed' or 'broadcast'."
        )
    image_size = _sequence(
        camera,
        "image_size",
        path="camera",
        item_types=(int,),
        length=2,
    )
    if any(cast("int", item) <= 0 for item in image_size):
        raise SemanticConfigurationError("camera.image_size values must be positive.")
    _sequence(
        camera,
        "fixed_look_at",
        path="camera",
        item_types=(float, int),
        length=3,
    )
    optional_ranges = {
        "broadcast_setback_range",
        "broadcast_height_range",
        "broadcast_court_width_frac_range",
    }
    for key in camera_fields - {"layout", "image_size", "fixed_look_at"}:
        if key in optional_ranges and camera[key] is None:
            continue
        if key in optional_ranges:
            _ordered_range(
                camera,
                key,
                path="camera",
                positive=True,
                upper_bound=(
                    1.0 if key == "broadcast_court_width_frac_range" else None
                ),
            )
        else:
            _number(camera, key, path="camera")
    z_min = _number(camera, "z_min", path="camera")
    z_max = _number(camera, "z_max", path="camera")
    if z_min <= 0.0 or z_max < z_min:
        raise SemanticConfigurationError(
            "camera z range must satisfy 0 < z_min <= z_max."
        )
    for key in {"hfov_deg", "broadcast_hfov_deg"}:
        angle = _number(camera, key, path="camera")
        if not 0.0 < angle < 180.0:
            raise SemanticConfigurationError(
                f"camera.{key} must be within (0, 180)."
            )
    for key in {
        "fixed_baseline_clear_extra",
        "fixed_position_noise_radius",
        "fixed_look_at_xy_radius",
        "broadcast_setback",
        "broadcast_look_at_height",
        "broadcast_position_noise_radius",
        "broadcast_look_at_xy_radius",
        "broadcast_hfov_jitter_deg",
    }:
        if _number(camera, key, path="camera") < 0.0:
            raise SemanticConfigurationError(f"camera.{key} must be non-negative.")
    if _number(camera, "broadcast_height", path="camera") <= 0.0:
        raise SemanticConfigurationError("camera.broadcast_height must be positive.")
    jitter = _number(camera, "broadcast_hfov_jitter_deg", path="camera")
    broadcast_hfov = _number(camera, "broadcast_hfov_deg", path="camera")
    if broadcast_hfov - jitter <= 0.0 or broadcast_hfov + jitter >= 180.0:
        raise SemanticConfigurationError(
            "camera.broadcast_hfov_jitter_deg must keep every sampled HFOV in (0, 180)."
        )
    if camera["broadcast_court_width_frac_range"] is not None and jitter != 0.0:
        raise SemanticConfigurationError(
            "camera.broadcast_court_width_frac_range and non-zero "
            "camera.broadcast_hfov_jitter_deg are mutually exclusive."
        )


def _validate_motion_sources(
    root: Mapping[str, object],
    *,
    resolver: PathResolver,
) -> None:
    motion_sources = require_config_mapping(
        root,
        "motion_sources",
        path="configuration",
    )
    if not motion_sources:
        raise SemanticConfigurationError("motion_sources must not be empty.")
    for category, source in motion_sources.items():
        if not category.strip():
            raise SemanticConfigurationError(
                "motion_sources category names must not be empty."
            )
        source_mapping = _reject_unknown(
            source,
            {"paths", "weight"},
            path=f"motion_sources.{category}",
        )
        source_paths = _sequence(
            source_mapping,
            "paths",
            path=f"motion_sources.{category}",
            item_types=(str,),
        )
        for item in source_paths:
            resolver.resolve(PathRole.EXTERNAL_ASSET, cast("str", item))
        if _number(source_mapping, "weight", path=f"motion_sources.{category}") <= 0.0:
            raise SemanticConfigurationError(
                f"motion_sources.{category}.weight must be positive."
            )


@dataclass(frozen=True, slots=True)
class PLCSGenerationComponents:
    """Shared generation values required by standalone and chunked runtimes."""

    mode: str
    paths: PLCSPathConfig

    @classmethod
    def from_config(cls, value: object) -> PLCSGenerationComponents:
        root = as_config_mapping(value, path="configuration")
        paths = PLCSPathConfig.from_config(root)
        mode = _validate_generation_mode(root)
        _validate_camera(root)
        _validate_motion_sources(root, resolver=paths.resolver)
        return cls(mode=mode, paths=paths)

"""Strict PLCS dataset-generation configuration and role-based path resolution."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast

from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import (
    as_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.tasks.base.generate_dataset.timeline_composer import TimelineConfig
from src.tasks.plcs.configuration import PLCSPathConfig
from src.utils.configuration import (
    ConfigurationTypeError,
    PathResolver,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.configuration.paths import PathRole
from src.utils.device import resolve_device
from src.utils.hydra import register_boundary_validator


def _reject_unknown(
    mapping: object, allowed: set[str], *, path: str
) -> Mapping[str, object]:
    resolved: Mapping[str, object] = as_config_mapping(mapping, path=path)
    unknown = sorted(set(resolved) - allowed)
    if unknown:
        raise UnknownConfigurationKeyError(
            f"Unknown configuration key(s): {', '.join(f'{path}.{key}' for key in unknown)}."
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


def _ordered_range(
    mapping: Mapping[str, object],
    key: str,
    *,
    path: str,
    positive: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float, float]:
    values = _sequence(
        mapping, key, path=path, item_types=(float, int), length=2
    )
    lo, hi = (float(cast("float | int", item)) for item in values)
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise SemanticConfigurationError(f"{path}.{key} values must be finite.")
    if lo > hi:
        raise SemanticConfigurationError(
            f"{path}.{key} must be ordered low-to-high."
        )
    if positive and lo <= 0.0:
        raise SemanticConfigurationError(f"{path}.{key} values must be positive.")
    if lower_bound is not None and lo < lower_bound:
        raise SemanticConfigurationError(
            f"{path}.{key} values must be >= {lower_bound}."
        )
    if upper_bound is not None and hi > upper_bound:
        raise SemanticConfigurationError(
            f"{path}.{key} values must be <= {upper_bound}."
        )
    return lo, hi


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


@dataclass(frozen=True, slots=True)
class PLCSExternalAssets:
    """Resolved external assets required by generation workers."""

    smplh_model_path: Path


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
        timeline, "overlap_probability", float, path="generation.timeline"
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
        camera, "image_size", path="camera", item_types=(int,), length=2
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
            raise SemanticConfigurationError(f"camera.{key} must be within (0, 180).")
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
    root: Mapping[str, object], *, resolver: PathResolver
) -> None:
    motion_sources = require_config_mapping(
        root, "motion_sources", path="configuration"
    )
    if not motion_sources:
        raise SemanticConfigurationError("motion_sources must not be empty.")
    for category, source in motion_sources.items():
        if not category.strip():
            raise SemanticConfigurationError(
                "motion_sources category names must not be empty."
            )
        source_mapping = _reject_unknown(
            source, {"paths", "weight"}, path=f"motion_sources.{category}"
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


def validate_generation_components(value: object) -> str:
    """Validate generation sections shared by standalone and chunked runtimes."""
    root = as_config_mapping(value, path="configuration")
    mode = _validate_generation_mode(root)
    _validate_camera(root)
    _validate_motion_sources(
        root,
        resolver=PLCSPathConfig.from_config(value).resolver,
    )
    return mode


@dataclass(frozen=True, slots=True)
class PLCSGenerationConfig:
    """Validated generation boundary with a fully resolved worker config."""

    config: DictConfig
    output_dir: Path
    device: str
    seed: int
    num_workers: int
    num_scenes: int
    generation_mode: str
    external_assets: PLCSExternalAssets
    train_ratio: float
    val_ratio: float
    test_ratio: float

    OUTPUT_ROLE: ClassVar[PathRole] = PathRole.DATA

    @classmethod
    def from_config(cls, value: object) -> PLCSGenerationConfig:
        if not isinstance(value, DictConfig):
            raise ConfigurationTypeError(
                "PLCS generation boundary requires DictConfig."
            )
        path_config = PLCSPathConfig.from_config(value)
        root = as_config_mapping(
            OmegaConf.to_container(value, resolve=True), path="configuration"
        )
        root = _reject_unknown(
            root,
            {
                "generation",
                "paths",
                "external_assets",
                "simulation",
                "camera",
                "motion_sources",
                "run",
            },
            path="configuration",
        )
        run = _reject_unknown(
            require_config_mapping(root, "run", path="configuration"),
            {
                "output_dir",
                "seed",
                "device",
                "num_workers",
                "train_ratio",
                "val_ratio",
                "test_ratio",
            },
            path="run",
        )
        mode = validate_generation_components(root)
        output_relative = cast(
            "str", require_config_value(run, "output_dir", str, path="run")
        )
        output_dir = path_config.resolver.resolve(PathRole.DATA, output_relative)
        requested_device = cast(
            "str", require_config_value(run, "device", str, path="run")
        )
        try:
            device = str(resolve_device(requested_device, allow_fallback=False))
        except (RuntimeError, ValueError) as error:
            raise SemanticConfigurationError(
                f"run.device is not an available device: {requested_device!r}."
            ) from error
        if device != "cpu":
            raise SemanticConfigurationError(
                "run.device must resolve to 'cpu' for parallel PLCS generation."
            )
        workers = cast("int", require_config_value(run, "num_workers", int, path="run"))
        simulation = _reject_unknown(
            require_config_mapping(root, "simulation", path="configuration"),
            {"num_scenes"},
            path="simulation",
        )
        scenes = cast(
            "int",
            require_config_value(simulation, "num_scenes", int, path="simulation"),
        )
        if workers < 1 or scenes < 1:
            raise SemanticConfigurationError(
                "run.num_workers and simulation.num_scenes must be positive."
            )
        ratios = tuple(
            float(
                cast(
                    "float | int",
                    require_config_value(run, key, (float, int), path="run"),
                )
            )
            for key in ("train_ratio", "val_ratio", "test_ratio")
        )
        if (
            any(not math.isfinite(ratio) or ratio < 0 for ratio in ratios)
            or abs(sum(ratios) - 1.0) > 1e-8
        ):
            raise SemanticConfigurationError(
                "Generation split ratios must be non-negative and sum to 1."
            )

        resolved = resolve_generation_paths(value)
        resolved.run.output_dir = str(output_dir)
        resolved.run.device = device
        return cls(
            config=resolved,
            output_dir=output_dir,
            device=device,
            seed=cast("int", require_config_value(run, "seed", int, path="run")),
            num_workers=workers,
            num_scenes=scenes,
            generation_mode=mode,
            external_assets=PLCSExternalAssets(
                smplh_model_path=Path(str(resolved.external_assets.smplh_model_path))
            ),
            train_ratio=ratios[0],
            val_ratio=ratios[1],
            test_ratio=ratios[2],
        )


def _validate_boundary(config: DictConfig) -> None:
    PLCSGenerationConfig.from_config(config)


register_boundary_validator("plcs.generate_dataset", _validate_boundary)

__all__ = [
    "PLCSGenerationConfig",
    "resolve_generation_paths",
    "validate_generation_components",
]


def resolve_generation_paths(value: DictConfig) -> DictConfig:
    """Return a resolved copy for generation consumers, including chunk workers."""
    path_config = PLCSPathConfig.from_config(value)
    root = as_config_mapping(
        OmegaConf.to_container(value, resolve=True), path="configuration"
    )
    external_assets = require_config_mapping(
        root, "external_assets", path="configuration"
    )
    _reject_unknown(external_assets, {"smplh_model_path"}, path="external_assets")
    smplh_relative = cast(
        "str",
        require_config_value(
            external_assets, "smplh_model_path", str, path="external_assets"
        ),
    )
    container = OmegaConf.to_container(value, resolve=True)
    resolved = OmegaConf.create(container)
    if not isinstance(resolved, DictConfig):
        raise ConfigurationTypeError(
            "PLCS generation config must resolve to DictConfig."
        )
    resolved.external_assets.smplh_model_path = str(
        path_config.resolver.resolve(PathRole.EXTERNAL_ASSET, smplh_relative)
    )
    for category, source in resolved.motion_sources.items():
        source_mapping = as_config_mapping(source, path=f"motion_sources.{category}")
        paths = _sequence(
            source_mapping,
            "paths",
            path=f"motion_sources.{category}",
            item_types=(str,),
        )
        source.paths = [
            str(
                path_config.resolver.resolve(PathRole.EXTERNAL_ASSET, cast("str", path))
            )
            for path in paths
        ]
    return resolved

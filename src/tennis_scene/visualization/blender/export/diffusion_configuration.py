"""Strict CPU exporter configuration and explicit path roles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.configuration import PATHS_SCHEMA
from src.tennis_scene.visualization.blender.export.diffusion import DiffusionConfig
from src.utils.configuration import (
    ConfigField,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    SemanticConfigurationError,
    StrictConfigSchema,
)
from src.utils.paths import PROJECT_ROOT

DIFFUSION_SCHEMA = StrictConfigSchema(
    name="tennis_scene.export_blender_diffusion",
    fields={
        "paths": ConfigField.mapping(PATHS_SCHEMA),
        "input": ConfigField.of(str),
        "output": ConfigField.of(str),
        "badge_font": ConfigField.of(str),
        **{
            name: ConfigField.of(int)
            for name in (
                "seed",
                "noise_knots",
                "fps",
                "tube_sides",
                "width",
                "height",
                "samples",
                "gif_width",
                "gif_fps",
            )
        },
        **{
            name: ConfigField.of(int, float)
            for name in (
                "smoothing_lambda",
                "point_tolerance_m",
                "conditions_seconds",
                "denoise_seconds",
                "playback_seconds",
                "curve_radius_m",
                "condition_radius_m",
                "gap_radius_m",
                "ball_radius_m",
            )
        },
        "noise_scale_m": ConfigField.sequence(ConfigField.of(int, float)),
        "noise_center_m": ConfigField.sequence(ConfigField.of(int, float)),
    },
)


@dataclass(frozen=True)
class DiffusionRuntime:
    input_path: Path
    output_path: Path
    font_path: Path
    config: DiffusionConfig
    resolved_config: dict[str, Any]


def parse_diffusion_config(cfg: DictConfig) -> DiffusionRuntime:
    plain = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    value = DIFFUSION_SCHEMA.validate(plain)
    roots = RuntimePathRoots.from_mapping(value["paths"], repository_root=PROJECT_ROOT)
    resolver = PathResolver(roots)
    parameters = {name: value[name] for name in DiffusionConfig.__dataclass_fields__}
    try:
        config = DiffusionConfig(**parameters)
    except ValueError as error:
        raise SemanticConfigurationError(str(error)) from error
    return DiffusionRuntime(
        resolver.resolve(PathRole.ARTIFACT, value["input"]),
        resolver.resolve(PathRole.OUTPUT, value["output"]),
        resolver.resolve(PathRole.EXTERNAL_ASSET, value["badge_font"]),
        config,
        plain,
    )


def validate_diffusion_boundary(cfg: DictConfig) -> None:
    parse_diffusion_config(cfg)

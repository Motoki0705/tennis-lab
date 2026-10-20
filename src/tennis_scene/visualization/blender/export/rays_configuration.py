"""Strict configuration boundary for the read-only rays exporter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.configuration import PATHS_SCHEMA
from src.tennis_scene.visualization.blender.export.rays import RaysConfig
from src.utils.configuration import (
    ConfigField,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    SemanticConfigurationError,
    StrictConfigSchema,
)
from src.utils.paths import PROJECT_ROOT

RAYS_CONFIG_SCHEMA = StrictConfigSchema(
    name="tennis_scene.export_blender_rays",
    fields={
        "paths": ConfigField.mapping(PATHS_SCHEMA),
        "input": ConfigField.of(str),
        "output": ConfigField.of(str),
        "mode": ConfigField.of(str),
        "start": ConfigField.of(int, type(None)),
        "stop": ConfigField.of(int, type(None)),
        "t0": ConfigField.of(int, type(None)),
        "window_frames": ConfigField.of(int),
        "ranking_suppression": ConfigField.of(str),
        "fallback_ray_length_m": ConfigField.of(int, float),
        "plate_max_width": ConfigField.of(int),
        "plate_jpeg_quality": ConfigField.of(int),
    },
)


@dataclass(frozen=True)
class RaysRuntime:
    input_path: Path
    output_path: Path
    mode: str
    start: int | None
    stop: int | None
    t0: int | None
    config: RaysConfig
    resolved_config: dict[str, Any]


def parse_rays_config(cfg: DictConfig) -> RaysRuntime:
    """Resolve path roles and require an explicit interval for mode=export."""
    plain = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    value = RAYS_CONFIG_SCHEMA.validate(plain)
    roots = RuntimePathRoots.from_mapping(value["paths"], repository_root=PROJECT_ROOT)
    resolver = PathResolver(roots)
    mode = value["mode"]
    if mode not in ("rank", "export"):
        raise SemanticConfigurationError("mode must be rank or export")
    start, stop, t0 = (value[name] for name in ("start", "stop", "t0"))
    if mode == "export" and (
        start is None or stop is None or t0 is None or not 0 <= start <= t0 < stop
    ):
        raise SemanticConfigurationError("mode=export requires 0 <= start <= t0 < stop")
    if mode == "rank" and any(item is not None for item in (start, stop, t0)):
        raise SemanticConfigurationError("mode=rank does not accept start/stop/t0")
    try:
        config = RaysConfig(
            window_frames=value["window_frames"],
            ranking_suppression=value["ranking_suppression"],
            fallback_ray_length_m=float(value["fallback_ray_length_m"]),
            plate_max_width=value["plate_max_width"],
            plate_jpeg_quality=value["plate_jpeg_quality"],
        )
    except ValueError as error:
        raise SemanticConfigurationError(str(error)) from error
    return RaysRuntime(
        input_path=resolver.resolve(PathRole.ARTIFACT, value["input"]),
        output_path=resolver.resolve(PathRole.OUTPUT, value["output"]),
        mode=mode,
        start=start,
        stop=stop,
        t0=t0,
        config=config,
        resolved_config=plain,
    )


def validate_rays_boundary(cfg: DictConfig) -> None:
    """Validate before any store reads or output creation."""
    parse_rays_config(cfg)

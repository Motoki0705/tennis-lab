"""Strict configuration and path boundary for the court learning story."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.configuration.schema import ConfigField, StrictConfigSchema
from src.utils.hydra import register_boundary_validator
from src.utils.paths import PROJECT_ROOT

BOUNDARY = "court_detection.learning_stages"
DEFAULT_CONFIG = (
    PROJECT_ROOT / "src/tasks/court_detection/configs/visualize_learning_stages.yaml"
)

TAG_SCHEMA = StrictConfigSchema(
    name="court learning scalar tags",
    fields={
        name: ConfigField.of(str)
        for name in (
            "train_loss",
            "train_kp",
            "train_line",
            "train_seg",
            "train_semline",
            "train_pose",
            "val_kp",
            "val_line",
            "val_seg",
            "val_semline",
            "val_pose",
            "val_loss",
            "learning_rate",
        )
    },
)
_DURATION_SCHEMA = StrictConfigSchema(
    name="court story durations",
    fields={
        name: ConfigField.of(int, float)
        for name in ("intro", "warmup", "train", "select", "handoff")
    },
)
_RUN_SCHEMA = StrictConfigSchema(name="run", fields={"output_dir": ConfigField.of(str)})
_SCHEMA = StrictConfigSchema(
    name="court learning presentation",
    fields={
        "paths": ConfigField.of(dict),
        "run_bundle": ConfigField.of(str),
        "scalar_summary": ConfigField.of(str),
        "run": ConfigField.mapping(_RUN_SCHEMA),
        "tags": ConfigField.mapping(TAG_SCHEMA),
        "seconds": ConfigField.mapping(_DURATION_SCHEMA),
        "fps": ConfigField.of(int, float),
        "width": ConfigField.of(int),
        "height": ConfigField.of(int),
        "gif_width": ConfigField.of(int),
        "gif_fps": ConfigField.of(int, float),
    },
)


@dataclass(frozen=True)
class CourtPresentationRequest:
    bundle: Path
    scalar_summary: Path
    output_dir: Path
    tags: Mapping[str, str]
    seconds: Mapping[str, float]
    fps: float
    width: int
    height: int
    gif_width: int
    gif_fps: float
    resolved_config: Mapping[str, Any]


def default_tags() -> Mapping[str, str]:
    """Use the task's composition-owned defaults for the bundle-only API."""
    config = OmegaConf.load(DEFAULT_CONFIG)
    values = OmegaConf.to_container(config.tags, resolve=True)
    return cast(
        Mapping[str, str], TAG_SCHEMA.validate(cast(Mapping[str, object], values))
    )


def default_scalar_summary() -> Path:
    """Resolve the composition-owned scalar asset for the bundle-only API."""
    config = OmegaConf.load(DEFAULT_CONFIG)
    paths = OmegaConf.to_container(
        OmegaConf.load(DEFAULT_CONFIG.parent / "paths/default.yaml"), resolve=True
    )
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            cast(Mapping[str, object], paths), repository_root=PROJECT_ROOT
        )
    )
    return cast(Path, resolver.resolve(PathRole.PROJECT, str(config.scalar_summary)))


def build_request(config: DictConfig) -> CourtPresentationRequest:
    resolved = OmegaConf.to_container(config, resolve=True)
    raw = _SCHEMA.validate(cast(Mapping[str, object], resolved))
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            cast(Mapping[str, object], raw["paths"]),
            repository_root=PROJECT_ROOT,
        )
    )
    seconds = {
        key: float(value)
        for key, value in cast(Mapping[str, float], raw["seconds"]).items()
    }
    for name in ("fps", "gif_fps", "width", "height", "gif_width"):
        value = cast(float, raw[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if any(not math.isfinite(value) or value <= 0 for value in seconds.values()):
        raise ValueError("seconds must be finite and positive")
    if cast(int, raw["width"]) % 2 or cast(int, raw["height"]) % 2:
        raise ValueError("width and height must be even for H.264")
    tags = cast(Mapping[str, str], raw["tags"])
    run = cast(Mapping[str, str], raw["run"])
    if any(not value.strip() for value in tags.values()) or len(
        set(tags.values())
    ) != len(tags):
        raise ValueError("tags must be nonempty and unique")
    return CourtPresentationRequest(
        bundle=resolver.resolve(PathRole.PROJECT, cast(str, raw["run_bundle"])),
        scalar_summary=resolver.resolve(
            PathRole.PROJECT, cast(str, raw["scalar_summary"])
        ),
        output_dir=resolver.resolve(PathRole.OUTPUT, run["output_dir"])
        / "presentation",
        tags=tags,
        seconds=seconds,
        fps=float(cast(float, raw["fps"])),
        width=cast(int, raw["width"]),
        height=cast(int, raw["height"]),
        gif_width=cast(int, raw["gif_width"]),
        gif_fps=float(cast(float, raw["gif_fps"])),
        resolved_config=cast(Mapping[str, Any], resolved),
    )


def validate_learning_stages_boundary(config: DictConfig) -> None:
    build_request(config)


register_boundary_validator(BOUNDARY, validate_learning_stages_boundary)

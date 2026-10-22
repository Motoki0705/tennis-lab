"""Strict composition-owned configuration for the CPU annotation comparison."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from omegaconf import DictConfig, OmegaConf

from src.utils.configuration import (
    BoundaryPathField,
    ConfigField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    SemanticConfigurationError,
    StrictConfigSchema,
)
from src.utils.paths import PROJECT_ROOT

_INPUT_PATHS = NonHydraPathBoundary(
    name="annotation_correction.inputs",
    fields=(
        BoundaryPathField(
            name="annotation_root",
            role=PathRole.ARTIFACT,
            direction=PathDirection.INPUT,
            kind=PathKind.DIRECTORY,
            allow_role_root=True,
        ),
        BoundaryPathField(
            name="campaign_dir",
            role=PathRole.ARTIFACT,
            direction=PathDirection.INPUT,
            kind=PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)

_PATHS_SCHEMA = StrictConfigSchema(
    fields={
        **{f"{role.value}_root": ConfigField.of(str) for role in PathRole},
        "annotation_root": ConfigField.of(str),
        "campaign_dir": ConfigField.of(str),
    }
)
_RUN_SCHEMA = StrictConfigSchema(fields={"output_dir": ConfigField.of(str)})
_RESOLUTION_SCHEMA = StrictConfigSchema(
    fields={
        "width": ConfigField.of(int),
        "height": ConfigField.of(int),
    }
)
_PLAYBACK_SCHEMA = StrictConfigSchema(
    fields={
        "hold_on_change_frames": ConfigField.of(int),
    }
)
_GIF_SCHEMA = StrictConfigSchema(
    fields={
        "width": ConfigField.of(int),
        "fps": ConfigField.of(int, float),
    }
)
_FRAMES_SCHEMA = StrictConfigSchema(
    fields={
        "source": ConfigField.of(str),
        "range": ConfigField.of(str, type(None)),
        "store_dir": ConfigField.of(str, type(None)),
    }
)
_SCHEMA = StrictConfigSchema(
    fields={
        "paths": ConfigField.mapping(_PATHS_SCHEMA),
        "run": ConfigField.mapping(_RUN_SCHEMA),
        "mode": ConfigField.of(str),
        "clip_id": ConfigField.of(str, type(None)),
        "frames": ConfigField.mapping(_FRAMES_SCHEMA),
        "radius_px": ConfigField.of(int, float),
        "resolution": ConfigField.mapping(_RESOLUTION_SCHEMA),
        "playback": ConfigField.mapping(_PLAYBACK_SCHEMA),
        "gif": ConfigField.mapping(_GIF_SCHEMA),
    }
)


@dataclass(frozen=True)
class AnnotationCorrectionConfig:
    mode: Literal["rank", "render"]
    annotation_root: Path
    campaign_dir: Path
    run_dir: Path
    clip_id: str | None
    frame_range: tuple[int, int] | None
    frame_source: Literal["video", "store"]
    store_dir: Path | None
    radius_px: float
    width: int
    height: int
    hold_on_change_frames: int
    gif_width: int
    gif_fps: float
    resolved_config: dict[str, Any]


def build_runtime_config(config: DictConfig) -> AnnotationCorrectionConfig:
    raw = cast(
        dict[str, Any],
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )
    values = cast(dict[str, Any], _SCHEMA.validate(raw))
    paths = values["paths"]
    roots = RuntimePathRoots.from_mapping(
        {f"{role.value}_root": paths[f"{role.value}_root"] for role in PathRole},
        repository_root=PROJECT_ROOT,
    )
    resolver = PathResolver(roots)
    run_dir = resolver.resolve(PathRole.OUTPUT, values["run"]["output_dir"])
    inputs = _INPUT_PATHS.validate(
        {key: paths[key] for key in ("annotation_root", "campaign_dir")},
        resolver=resolver,
        independent_artifact_inputs=True,
    )
    annotation_root = inputs.declared("annotation_root").path
    campaign_dir = inputs.declared("campaign_dir").path
    mode = values["mode"]
    if mode not in ("rank", "render"):
        raise SemanticConfigurationError("mode must be rank or render")
    frame_source = values["frames"]["source"]
    store_fragment = values["frames"]["store_dir"]
    if frame_source not in ("video", "store"):
        raise SemanticConfigurationError("frames.source must be video or store")
    store_dir = None
    if frame_source == "store":
        if store_fragment is None:
            raise SemanticConfigurationError(
                "frames.source=store requires frames.store_dir"
            )
        store_dir = resolver.resolve(PathRole.DATA, store_fragment)
    elif store_fragment is not None:
        raise SemanticConfigurationError(
            "frames.store_dir requires frames.source=store"
        )
    clip_id, frames = values["clip_id"], values["frames"]["range"]
    frame_range = None
    if mode == "render":
        if clip_id is None or re.fullmatch(r"[A-Za-z0-9_-]+", clip_id) is None:
            raise SemanticConfigurationError("render requires a filename-stem clip_id")
        match = re.fullmatch(r"(\d+):(\d+)", frames) if frames is not None else None
        if match is None or int(match[1]) >= int(match[2]):
            raise SemanticConfigurationError(
                "render requires frames.range=start:stop (nonempty, half-open)"
            )
        frame_range = (int(match[1]), int(match[2]))
    elif clip_id is not None or frames is not None:
        raise SemanticConfigurationError(
            "rank requires clip_id=null and frames.range=null"
        )
    if mode == "rank" and frame_source != "video":
        raise SemanticConfigurationError("frames.source=store is a render-only option")
    radius = float(values["radius_px"])
    width, height = values["resolution"]["width"], values["resolution"]["height"]
    hold = values["playback"]["hold_on_change_frames"]
    gif_width, gif_fps = values["gif"]["width"], float(values["gif"]["fps"])
    if not math.isfinite(radius) or radius <= 0:
        raise SemanticConfigurationError("radius_px must be finite and positive")
    if (
        width < 160
        or height < 90
        or width % 2
        or height % 2
        or width * 9 != height * 16
    ):
        raise SemanticConfigurationError(
            "resolution must be even, 16:9, and at least 160x90"
        )
    if hold < 0:
        raise SemanticConfigurationError("hold_on_change_frames must be non-negative")
    if gif_width <= 0 or not math.isfinite(gif_fps) or gif_fps <= 0:
        raise SemanticConfigurationError(
            "gif width and fps must be positive and finite"
        )
    raw["paths"].update(roots.as_mapping())
    raw["paths"]["annotation_root"] = str(annotation_root)
    raw["paths"]["campaign_dir"] = str(campaign_dir)
    return AnnotationCorrectionConfig(
        mode,
        annotation_root,
        campaign_dir,
        run_dir,
        clip_id,
        frame_range,
        frame_source,
        store_dir,
        radius,
        width,
        height,
        hold,
        gif_width,
        gif_fps,
        raw,
    )

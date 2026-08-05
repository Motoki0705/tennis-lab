"""Orchestrate court-detection prediction visualizations (kp / seg / line)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from omegaconf import DictConfig

from src.tasks.base.configuration import require_config_mapping, require_config_value
from src.tasks.base.visualization.gif import save_gif
from src.tasks.court_detection.configuration import (
    CourtRenderConfig,
    validate_paths_boundary,
)
from src.tasks.court_detection.visualization.api.predict import (
    predict_kp,
    predict_line,
    predict_seg,
)
from src.tasks.court_detection.visualization.io.frames import load_court_frames
from src.tasks.court_detection.visualization.rendering import (
    CourtRenderStyle,
    render_kp_frames,
    render_line_frames,
    render_seg_frames,
)
from src.utils.configuration import (
    MissingConfigurationKeyError,
    PathResolver,
    PathRole,
    UnknownConfigurationKeyError,
)

logger = logging.getLogger(__name__)

_VALID_TASKS = {"kp", "seg", "line"}


def _number(value: object) -> float:
    return float(cast("float | int", value))


def _integer(value: object) -> int:
    return cast("int", value)


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime settings for court-detection visualization."""

    task: str
    image_source: str
    checkpoint: str
    save: Path
    device: str
    allow_device_fallback: bool
    resolver: PathResolver
    fps: float
    max_frames: int | None
    gif_loop: int
    info: bool
    clip_label: str
    style: CourtRenderStyle


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Build runtime config from composed Hydra config."""
    root, resolver = validate_paths_boundary(
        cfg, expected_sections={"visualization", "run", "render_style"}
    )
    vis = require_config_mapping(root, "visualization", path="configuration")
    run = require_config_mapping(root, "run", path="configuration")
    _require_exact(run, {"output_dir", "device", "allow_device_fallback"}, path="run")
    _require_exact(
        vis,
        {
            "task",
            "image_source",
            "checkpoint",
            "save",
            "fps",
            "max_frames",
            "info",
            "gif",
        },
        path="visualization",
    )

    task = str(require_config_value(vis, "task", str, path="visualization"))
    if task not in _VALID_TASKS:
        raise ValueError(
            f"visualization.task must be one of {sorted(_VALID_TASKS)}, got {task!r}."
        )

    image_source_raw = str(
        require_config_value(vis, "image_source", str, path="visualization")
    )
    checkpoint = str(require_config_value(vis, "checkpoint", str, path="visualization"))
    save_raw = str(require_config_value(vis, "save", str, path="visualization"))
    image_source = str(resolver.resolve(PathRole.DATA, image_source_raw))
    save_path = resolver.resolve(PathRole.ARTIFACT, save_raw)
    resolver.resolve(
        PathRole.OUTPUT,
        str(require_config_value(run, "output_dir", str, path="run")),
    )

    fps = _number(require_config_value(vis, "fps", (float, int), path="visualization"))
    if fps <= 0:
        raise ValueError("visualization.fps must be positive.")

    max_frames_raw = require_config_value(
        vis, "max_frames", (int, type(None)), path="visualization"
    )
    max_frames = cast("int | None", max_frames_raw)
    if max_frames is not None and max_frames <= 0:
        raise ValueError("visualization.max_frames must be positive when set.")

    gif_cfg = require_config_mapping(vis, "gif", path="visualization")
    _require_exact(gif_cfg, {"loop"}, path="visualization.gif")
    style = CourtRenderConfig.from_mapping(
        require_config_mapping(root, "render_style", path="configuration")
    ).build()

    return RuntimeConfig(
        task=task,
        image_source=image_source,
        checkpoint=checkpoint,
        save=save_path,
        device=str(require_config_value(run, "device", str, path="run")),
        allow_device_fallback=cast(
            "bool", require_config_value(run, "allow_device_fallback", bool, path="run")
        ),
        resolver=resolver,
        fps=fps,
        max_frames=max_frames,
        gif_loop=_integer(
            require_config_value(gif_cfg, "loop", int, path="visualization.gif")
        ),
        info=cast(
            "bool", require_config_value(vis, "info", bool, path="visualization")
        ),
        clip_label=f"court/{task}",
        style=style,
    )


def _require_exact(mapping: Mapping[str, object], keys: set[str], *, path: str) -> None:
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


def run_visualization(cfg: RuntimeConfig) -> int:
    """Run court-detection visualization for the configured task and save a GIF."""
    frames = load_court_frames(cfg.image_source, max_frames=cfg.max_frames)
    logger.info("Loaded %d frame(s) from %s.", len(frames), cfg.image_source)

    if cfg.info:
        logger.info("Task: %s", cfg.task)
        logger.info("Checkpoint: %s", cfg.checkpoint)
        logger.info("Save path: %s", cfg.save)
        return 0

    if cfg.task == "kp":
        predictions = predict_kp(
            checkpoint_path=cfg.checkpoint,
            device=cfg.device,
            resolver=cfg.resolver,
            allow_device_fallback=cfg.allow_device_fallback,
            frames=frames,
        )
        rendered = render_kp_frames(
            frames=frames,
            predictions=predictions,
            style=cfg.style,
            clip_label=cfg.clip_label,
        )
    elif cfg.task == "seg":
        masks = predict_seg(
            checkpoint_path=cfg.checkpoint,
            device=cfg.device,
            resolver=cfg.resolver,
            allow_device_fallback=cfg.allow_device_fallback,
            frames=frames,
        )
        rendered = render_seg_frames(
            frames=frames,
            masks=masks,
            style=cfg.style,
            clip_label=cfg.clip_label,
        )
    else:  # line
        probs = predict_line(
            checkpoint_path=cfg.checkpoint,
            device=cfg.device,
            resolver=cfg.resolver,
            allow_device_fallback=cfg.allow_device_fallback,
            frames=frames,
        )
        rendered = render_line_frames(
            frames=frames,
            probs=probs,
            style=cfg.style,
            clip_label=cfg.clip_label,
        )

    save_gif(frames_rgb=rendered, path=cfg.save, fps=cfg.fps, loop=cfg.gif_loop)
    logger.info("Saved %s visualization to %s", cfg.task, cfg.save)
    return 0

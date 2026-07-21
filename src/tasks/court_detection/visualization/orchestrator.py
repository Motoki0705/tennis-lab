"""Orchestrate court-detection prediction visualizations (kp / seg / line)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.tasks.base.visualization.layout import PanelStyle
from src.tasks.base.visualization.orchestrator import (
    parse_rgb as _parse_rgb,
)
from src.tasks.base.visualization.orchestrator import resolve_device
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
from src.utils.video.animation import save_rgb_animation

logger = logging.getLogger(__name__)

_VALID_TASKS = {"kp", "seg", "line"}


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime settings for court-detection visualization."""

    task: str
    image_source: str
    checkpoint: Path
    save: Path
    device: str
    fps: float
    max_frames: int | None
    gif_loop: int
    info: bool
    clip_label: str
    style: CourtRenderStyle


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Build runtime config from composed Hydra config."""
    vis = cfg.visualization
    run = cfg.get("run", {})

    task = str(vis.task).strip().lower()
    if task not in _VALID_TASKS:
        raise ValueError(f"visualization.task must be one of {sorted(_VALID_TASKS)}, got {task!r}.")

    image_source = to_absolute_path(str(vis.image_source))
    checkpoint = Path(to_absolute_path(str(vis.checkpoint)))
    output_dir = Path(to_absolute_path(str(run.get("output_dir", "outputs/court_detection/visualization"))))
    save_raw = vis.get("save")
    save_path = (
        Path(to_absolute_path(str(save_raw)))
        if save_raw
        else output_dir / f"{task}.mp4"
    )

    fps = float(vis.get("fps", 6.0))
    if fps <= 0:
        raise ValueError("visualization.fps must be positive.")

    max_frames_raw = vis.get("max_frames")
    max_frames = None if max_frames_raw in {None, "", 0} else int(max_frames_raw)
    if max_frames is not None and max_frames <= 0:
        raise ValueError("visualization.max_frames must be positive when set.")

    draw_cfg = vis.get("draw", {})
    layout_cfg = vis.get("layout", {})
    gif_cfg = vis.get("gif", {})

    style = CourtRenderStyle(
        panel=PanelStyle(
            background_rgb=_parse_rgb(layout_cfg.get("background_rgb", [18, 18, 18]), name="visualization.layout.background_rgb"),
            text_color_rgb=_parse_rgb(layout_cfg.get("text_color_rgb", [245, 245, 245]), name="visualization.layout.text_color_rgb"),
            text_scale=float(layout_cfg.get("text_scale", 0.52)),
            text_thickness=int(layout_cfg.get("text_thickness", 1)),
            tile_gap=int(layout_cfg.get("tile_gap", 12)),
            panel_label_height=int(layout_cfg.get("panel_label_height", 24)),
        ),
        header_height=int(layout_cfg.get("header_height", 36)),
        display_width=int(layout_cfg.get("display_width", 640)),
        kp_radius=int(draw_cfg.get("kp_radius", 4)),
        kp_color_rgb=_parse_rgb(draw_cfg.get("kp_color_rgb", [96, 255, 128]), name="visualization.draw.kp_color_rgb"),
        kp_thickness=int(draw_cfg.get("kp_thickness", -1)),
        line_threshold=float(draw_cfg.get("line_threshold", 0.5)),
    )

    return RuntimeConfig(
        task=task,
        image_source=image_source,
        checkpoint=checkpoint,
        save=save_path,
        device=resolve_device(str(run.get("device", "auto"))),
        fps=fps,
        max_frames=max_frames,
        gif_loop=int(gif_cfg.get("loop", 0)),
        info=bool(vis.get("info", False)),
        clip_label=f"court/{task}",
        style=style,
    )


def run_visualization(cfg: RuntimeConfig) -> int:
    """Run court-detection visualization and save an animation."""
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
            frames=frames,
        )
        rendered = render_line_frames(
            frames=frames,
            probs=probs,
            style=cfg.style,
            clip_label=cfg.clip_label,
        )

    save_rgb_animation(rendered, cfg.save, fps=cfg.fps, loop=cfg.gif_loop)
    logger.info("Saved %s visualization to %s", cfg.task, cfg.save)
    return 0

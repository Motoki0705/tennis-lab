"""Orchestrate clip-level ball detection prediction visualizations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig

from src.tasks.ball_detection.configuration import BallRuntimePaths
from src.tasks.ball_detection.inference import BallDetectionPredictor
from src.tasks.ball_detection.visualization.api.predict import (
    build_mdd_frames,
    predict_clip,
)
from src.tasks.ball_detection.visualization.io.clip import load_clip_sequence
from src.tasks.ball_detection.visualization.rendering import (
    DrawStyle,
    LayoutStyle,
    render_animation_frames,
)
from src.tasks.base.visualization.gif import save_gif
from src.tasks.base.visualization.orchestrator import (
    parse_float_triplet as _parse_float_triplet,
)
from src.tasks.base.visualization.orchestrator import (
    parse_hw as _parse_hw,
)
from src.tasks.base.visualization.orchestrator import (
    parse_rgb as _parse_rgb,
)
from src.tasks.base.visualization.orchestrator import resolve_device
from src.utils.configuration import PathResolver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime settings for ball detection visualization."""

    clip_dir: Path
    checkpoint: Path
    save: Path
    device: str
    fps: float
    window_stride: int
    inference_batch_size: int
    sequence_length: int
    image_size_hw: tuple[int, int]
    peak_threshold: float
    max_frames: int | None
    normalize_imagenet: bool
    imagenet_mean: tuple[float, float, float]
    imagenet_std: tuple[float, float, float]
    gif_loop: int
    info: bool
    clip_label: str
    draw: DrawStyle
    layout: LayoutStyle
    resolver: PathResolver
    allow_device_fallback: bool
    subpixel_refine: bool
    strict: bool
    weights_only: bool


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Build runtime config from composed Hydra config."""
    vis = cfg.visualization
    run = cfg.run
    data_cfg = cfg.data
    metrics_cfg = cfg.metrics
    paths = BallRuntimePaths.from_config(cfg)

    clip_dir = paths.data(str(vis.clip_dir))
    checkpoint = paths.checkpoint(str(vis.checkpoint))
    save_path = paths.artifact(str(vis.save))

    image_size_hw = _parse_hw(data_cfg.image_size, name="data.image_size")
    sequence_length = int(cfg.model.num_frames)
    window_stride = int(vis.window_stride)
    if window_stride <= 0 or window_stride > sequence_length:
        raise ValueError(
            f"visualization.window_stride must be in [1, {sequence_length}], got {window_stride}."
        )

    inference_batch_size = int(vis.inference_batch_size)
    if inference_batch_size <= 0:
        raise ValueError("visualization.inference_batch_size must be positive.")

    fps = float(vis.fps)
    if fps <= 0:
        raise ValueError("visualization.fps must be positive.")

    max_frames_raw = vis.max_frames
    max_frames = None if max_frames_raw is None else int(max_frames_raw)
    if max_frames is not None and max_frames <= 0:
        raise ValueError("visualization.max_frames must be positive when set.")

    normalize_cfg = data_cfg.augmentation.normalize_imagenet
    normalize_imagenet = bool(normalize_cfg.enabled)
    imagenet_mean = _parse_float_triplet(
        normalize_cfg.mean,
        name="data.augmentation.normalize_imagenet.mean",
    )
    imagenet_std = _parse_float_triplet(
        normalize_cfg.std,
        name="data.augmentation.normalize_imagenet.std",
    )

    draw_cfg = vis.draw
    layout_cfg = vis.layout
    gif_cfg = vis.gif

    return RuntimeConfig(
        clip_dir=clip_dir,
        checkpoint=checkpoint,
        save=save_path,
        device=resolve_device(str(run.device)),
        fps=fps,
        window_stride=window_stride,
        inference_batch_size=inference_batch_size,
        sequence_length=sequence_length,
        image_size_hw=image_size_hw,
        peak_threshold=float(vis.peak_threshold),
        max_frames=max_frames,
        normalize_imagenet=normalize_imagenet,
        imagenet_mean=imagenet_mean,
        imagenet_std=imagenet_std,
        gif_loop=int(gif_cfg.loop),
        info=bool(vis.info),
        clip_label=f"{clip_dir.parent.name}/{clip_dir.name}",
        draw=DrawStyle(
            gt_radius=int(draw_cfg.gt_radius),
            pred_radius=int(draw_cfg.pred_radius),
            thickness=int(draw_cfg.thickness),
            gt_color_rgb=_parse_rgb(
                draw_cfg.gt_color_rgb, name="visualization.draw.gt_color_rgb"
            ),
            pred_color_rgb=_parse_rgb(
                draw_cfg.pred_color_rgb, name="visualization.draw.pred_color_rgb"
            ),
            text_color_rgb=_parse_rgb(
                draw_cfg.text_color_rgb, name="visualization.draw.text_color_rgb"
            ),
            muted_text_color_rgb=_parse_rgb(
                draw_cfg.muted_text_color_rgb,
                name="visualization.draw.muted_text_color_rgb",
            ),
        ),
        layout=LayoutStyle(
            header_height=int(layout_cfg.header_height),
            tile_gap=int(layout_cfg.tile_gap),
            text_scale=float(layout_cfg.text_scale),
            text_thickness=int(layout_cfg.text_thickness),
            background_rgb=_parse_rgb(
                layout_cfg.background_rgb, name="visualization.layout.background_rgb"
            ),
            panel_label_height=int(layout_cfg.panel_label_height),
        ),
        resolver=paths.resolver,
        allow_device_fallback=bool(run.allow_device_fallback),
        subpixel_refine=bool(metrics_cfg.subpixel_refine),
        strict=bool(vis.strict),
        weights_only=bool(vis.weights_only),
    )


def run_visualization(cfg: RuntimeConfig) -> int:
    """Run clip-level visualization and save a GIF."""
    clip = load_clip_sequence(
        clip_dir=cfg.clip_dir,
        sequence_length=cfg.sequence_length,
        image_size_hw=cfg.image_size_hw,
        max_frames=cfg.max_frames,
        normalize_imagenet=cfg.normalize_imagenet,
        imagenet_mean=cfg.imagenet_mean,
        imagenet_std=cfg.imagenet_std,
    )

    logger.info("Loaded clip %s with %d frames.", cfg.clip_label, len(clip.frame_names))
    if cfg.info:
        logger.info("Checkpoint: %s", cfg.checkpoint)
        logger.info("Save path: %s", cfg.save)
        logger.info("Sequence length: %d", cfg.sequence_length)
        logger.info("Window stride: %d", cfg.window_stride)
        return 0

    predictor = BallDetectionPredictor.load_from_checkpoint(
        cfg.checkpoint,
        resolver=cfg.resolver,
        device=cfg.device,
        allow_device_fallback=cfg.allow_device_fallback,
        subpixel_refine=cfg.subpixel_refine,
        strict=cfg.strict,
        weights_only=cfg.weights_only,
    )
    predictions = predict_clip(
        predictor=predictor,
        clip=clip,
        sequence_length=cfg.sequence_length,
        window_stride=cfg.window_stride,
        inference_batch_size=cfg.inference_batch_size,
        image_size_hw=cfg.image_size_hw,
        peak_threshold=cfg.peak_threshold,
    )
    mdd_frames_rgb = build_mdd_frames(predictor=predictor, clip=clip)

    rendered_frames = render_animation_frames(
        frames_rgb=[frame.cpu().numpy() for frame in clip.render_frames_rgb],
        frame_names=clip.frame_names,
        mdd_frames_rgb=mdd_frames_rgb,
        pred_coords_px=[
            _coord_tensor_to_tuple(coord) for coord in predictions.coords_px
        ],
        pred_visibility=[bool(value.item()) for value in predictions.visibility],
        pred_confidences=[float(value.item()) for value in predictions.confidences],
        pred_heatmaps=[heatmap.cpu().numpy() for heatmap in predictions.heatmaps],
        peak_threshold=cfg.peak_threshold,
        clip_label=cfg.clip_label,
        draw=cfg.draw,
        layout=cfg.layout,
    )
    save_gif(
        frames_rgb=rendered_frames,
        path=cfg.save,
        fps=cfg.fps,
        loop=cfg.gif_loop,
    )
    logger.info("Saved clip visualization to %s", cfg.save)
    return 0


def _coord_tensor_to_tuple(coord: torch.Tensor) -> tuple[float, float]:
    return float(coord[0].item()), float(coord[1].item())

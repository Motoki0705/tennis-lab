"""Orchestrate clip-level ball detection prediction visualizations."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.tasks.ball_detection.data.augmentation import normalize_tensor_images_imagenet
from src.tasks.ball_detection.data.types import FrameLabel
from src.tasks.ball_detection.inference import BallDetectionPredictor
from src.tasks.ball_detection.visualization.rendering import (
    DrawStyle,
    LayoutStyle,
    render_animation_frames,
    save_gif,
)
from src.utils.data.heatmaps import heatmaps_to_argmax

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


@dataclass(frozen=True)
class ClipSequence:
    """Loaded and preprocessed clip data for inference and rendering."""

    frame_names: tuple[str, ...]
    render_frames_rgb: tuple[torch.Tensor, ...]
    model_images: torch.Tensor
    gt_coords_px: torch.Tensor
    gt_visibility: torch.Tensor


@dataclass(frozen=True)
class PredictionSequence:
    """Aggregated per-frame predictions for one clip."""

    heatmaps: torch.Tensor
    coords_px: torch.Tensor
    confidences: torch.Tensor
    visibility: torch.Tensor


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Build runtime config from composed Hydra config."""
    vis = cfg.visualization
    run = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    metrics_cfg = cfg.get("metrics", {})

    clip_dir = Path(to_absolute_path(str(vis.clip_dir)))
    checkpoint = Path(to_absolute_path(str(vis.checkpoint)))
    output_dir = Path(to_absolute_path(str(run.get("output_dir", "outputs/ball_detection/visualization"))))
    save_raw = vis.get("save")
    save_path = (
        Path(to_absolute_path(str(save_raw)))
        if save_raw
        else output_dir / _default_gif_name(clip_dir)
    )

    image_size_hw = _parse_hw(data_cfg.get("image_size", [288, 512]), name="data.image_size")
    sequence_length = int(cfg.model.get("num_frames", 8))
    window_stride = int(vis.get("window_stride", 1))
    if window_stride <= 0 or window_stride > sequence_length:
        raise ValueError(
            f"visualization.window_stride must be in [1, {sequence_length}], got {window_stride}."
        )

    inference_batch_size = int(vis.get("inference_batch_size", 4))
    if inference_batch_size <= 0:
        raise ValueError("visualization.inference_batch_size must be positive.")

    fps = float(vis.get("fps", 12.0))
    if fps <= 0:
        raise ValueError("visualization.fps must be positive.")

    max_frames_raw = vis.get("max_frames")
    max_frames = None if max_frames_raw in {None, "", 0} else int(max_frames_raw)
    if max_frames is not None and max_frames <= 0:
        raise ValueError("visualization.max_frames must be positive when set.")

    normalize_cfg = data_cfg.get("augmentation", {}).get("normalize_imagenet", {})
    normalize_imagenet = bool(normalize_cfg.get("enabled", False))
    imagenet_mean = _parse_float_triplet(
        normalize_cfg.get("mean", (0.485, 0.456, 0.406)),
        name="data.augmentation.normalize_imagenet.mean",
    )
    imagenet_std = _parse_float_triplet(
        normalize_cfg.get("std", (0.229, 0.224, 0.225)),
        name="data.augmentation.normalize_imagenet.std",
    )

    draw_cfg = vis.get("draw", {})
    layout_cfg = vis.get("layout", {})
    gif_cfg = vis.get("gif", {})

    return RuntimeConfig(
        clip_dir=clip_dir,
        checkpoint=checkpoint,
        save=save_path,
        device=_resolve_device(str(run.get("device", "auto"))),
        fps=fps,
        window_stride=window_stride,
        inference_batch_size=inference_batch_size,
        sequence_length=sequence_length,
        image_size_hw=image_size_hw,
        peak_threshold=float(vis.get("peak_threshold", metrics_cfg.get("peak_threshold", 0.5))),
        max_frames=max_frames,
        normalize_imagenet=normalize_imagenet,
        imagenet_mean=imagenet_mean,
        imagenet_std=imagenet_std,
        gif_loop=int(gif_cfg.get("loop", 0)),
        info=bool(vis.get("info", False)),
        clip_label=f"{clip_dir.parent.name}/{clip_dir.name}",
        draw=DrawStyle(
            gt_radius=int(draw_cfg.get("gt_radius", 6)),
            pred_radius=int(draw_cfg.get("pred_radius", 6)),
            thickness=int(draw_cfg.get("thickness", 2)),
            gt_color_rgb=_parse_rgb(draw_cfg.get("gt_color_rgb", [255, 96, 96]), name="visualization.draw.gt_color_rgb"),
            pred_color_rgb=_parse_rgb(draw_cfg.get("pred_color_rgb", [96, 255, 128]), name="visualization.draw.pred_color_rgb"),
            text_color_rgb=_parse_rgb(draw_cfg.get("text_color_rgb", [245, 245, 245]), name="visualization.draw.text_color_rgb"),
            muted_text_color_rgb=_parse_rgb(draw_cfg.get("muted_text_color_rgb", [168, 168, 168]), name="visualization.draw.muted_text_color_rgb"),
        ),
        layout=LayoutStyle(
            header_height=int(layout_cfg.get("header_height", 44)),
            tile_gap=int(layout_cfg.get("tile_gap", 12)),
            text_scale=float(layout_cfg.get("text_scale", 0.52)),
            text_thickness=int(layout_cfg.get("text_thickness", 1)),
            background_rgb=_parse_rgb(layout_cfg.get("background_rgb", [18, 18, 18]), name="visualization.layout.background_rgb"),
            heatmap_alpha=float(layout_cfg.get("heatmap_alpha", 0.45)),
            show_heatmap_panel=bool(layout_cfg.get("show_heatmap_panel", True)),
        ),
    )


def run_visualization(cfg: RuntimeConfig) -> int:
    """Run clip-level visualization and save a GIF."""
    clip = _load_clip_sequence(cfg)

    logger.info("Loaded clip %s with %d frames.", cfg.clip_label, len(clip.frame_names))
    if cfg.info:
        logger.info("Checkpoint: %s", cfg.checkpoint)
        logger.info("Save path: %s", cfg.save)
        logger.info("Sequence length: %d", cfg.sequence_length)
        logger.info("Window stride: %d", cfg.window_stride)
        return 0

    predictor = BallDetectionPredictor.load_from_checkpoint(
        cfg.checkpoint,
        device=cfg.device,
    )
    predictions = _predict_clip(cfg=cfg, clip=clip, predictor=predictor)

    rendered_frames = render_animation_frames(
        frames_rgb=[frame.cpu().numpy() for frame in clip.render_frames_rgb],
        frame_names=clip.frame_names,
        gt_coords_px=[_coord_tensor_to_tuple(coord) for coord in clip.gt_coords_px],
        gt_visibility=[bool(value.item() > 0.5) for value in clip.gt_visibility],
        pred_coords_px=[_coord_tensor_to_tuple(coord) for coord in predictions.coords_px],
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


def _load_clip_sequence(cfg: RuntimeConfig) -> ClipSequence:
    label_path = cfg.clip_dir / "Label.csv"
    if not cfg.clip_dir.exists():
        raise FileNotFoundError(f"Clip directory not found: {cfg.clip_dir}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label.csv not found for clip: {cfg.clip_dir}")

    frame_paths = sorted(cfg.clip_dir.glob("*.jpg"))
    if cfg.max_frames is not None:
        frame_paths = frame_paths[: cfg.max_frames]
    if len(frame_paths) < cfg.sequence_length:
        raise ValueError(
            f"Clip {cfg.clip_dir} has only {len(frame_paths)} frames, but model.num_frames={cfg.sequence_length}."
        )

    labels = _read_label_csv(label_path)
    image_height, image_width = cfg.image_size_hw
    render_frames_rgb: list[torch.Tensor] = []
    model_frames: list[torch.Tensor] = []
    gt_coords_px: list[tuple[float, float]] = []
    gt_visibility: list[float] = []
    frame_names: list[str] = []

    for frame_path in frame_paths:
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        original_height, original_width = frame_rgb.shape[:2]
        resized_rgb = cv2.resize(
            frame_rgb,
            (image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )

        render_frames_rgb.append(torch.from_numpy(resized_rgb.copy()).to(torch.uint8))

        model_frame = torch.from_numpy(resized_rgb.transpose(2, 0, 1)).to(torch.float32) / 255.0
        model_frames.append(model_frame)

        label = labels.get(frame_path.name, FrameLabel(visibility=0.0, x=0.0, y=0.0))
        if label.visibility > 0:
            gt_coords_px.append(
                (
                    label.x * image_width / max(original_width, 1),
                    label.y * image_height / max(original_height, 1),
                )
            )
            gt_visibility.append(1.0)
        else:
            gt_coords_px.append((0.0, 0.0))
            gt_visibility.append(0.0)

        frame_names.append(frame_path.name)

    model_images = torch.stack(model_frames, dim=0)
    if cfg.normalize_imagenet:
        model_images = normalize_tensor_images_imagenet(
            model_images,
            mean=cfg.imagenet_mean,
            std=cfg.imagenet_std,
        )

    return ClipSequence(
        frame_names=tuple(frame_names),
        render_frames_rgb=tuple(render_frames_rgb),
        model_images=model_images,
        gt_coords_px=torch.tensor(gt_coords_px, dtype=torch.float32),
        gt_visibility=torch.tensor(gt_visibility, dtype=torch.float32),
    )


def _predict_clip(
    *,
    cfg: RuntimeConfig,
    clip: ClipSequence,
    predictor: BallDetectionPredictor,
) -> PredictionSequence:
    window_starts = _build_window_starts(
        frame_count=len(clip.frame_names),
        sequence_length=cfg.sequence_length,
        stride=cfg.window_stride,
    )
    logger.info("Running predictor over %d overlapping window(s).", len(window_starts))

    heatmap_sum: torch.Tensor | None = None
    heatmap_count = torch.zeros(len(clip.frame_names), dtype=torch.float32)

    for start_chunk in _chunked(window_starts, cfg.inference_batch_size):
        batch = torch.stack(
            [clip.model_images[start : start + cfg.sequence_length] for start in start_chunk],
            dim=0,
        )
        outputs = predictor.predict(batch, return_heatmaps=True)
        batch_heatmaps = outputs["heatmaps"].to(torch.float32)

        if heatmap_sum is None:
            heatmap_sum = torch.zeros(
                (len(clip.frame_names), *batch_heatmaps.shape[-2:]),
                dtype=torch.float32,
            )

        for window_index, start in enumerate(start_chunk):
            end = start + cfg.sequence_length
            heatmap_sum[start:end] += batch_heatmaps[window_index]
            heatmap_count[start:end] += 1.0

    if heatmap_sum is None:
        raise RuntimeError("Failed to aggregate prediction heatmaps for the clip.")

    averaged_heatmaps = heatmap_sum / torch.clamp(heatmap_count, min=1.0).view(-1, 1, 1)
    coords_normalized, confidences = heatmaps_to_argmax(averaged_heatmaps)

    image_height, image_width = cfg.image_size_hw
    coords_px = torch.empty_like(coords_normalized)
    coords_px[:, 0] = coords_normalized[:, 0] * max(image_width - 1, 0)
    coords_px[:, 1] = coords_normalized[:, 1] * max(image_height - 1, 0)
    visibility = confidences >= cfg.peak_threshold

    return PredictionSequence(
        heatmaps=averaged_heatmaps,
        coords_px=coords_px,
        confidences=confidences,
        visibility=visibility,
    )


def _build_window_starts(*, frame_count: int, sequence_length: int, stride: int) -> list[int]:
    if frame_count < sequence_length:
        raise ValueError(
            f"frame_count must be >= sequence_length, got {frame_count} < {sequence_length}."
        )

    starts = list(range(0, frame_count - sequence_length + 1, stride))
    last_start = frame_count - sequence_length
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _chunked(values: Sequence[int], chunk_size: int) -> Iterator[list[int]]:
    for start_index in range(0, len(values), chunk_size):
        yield list(values[start_index : start_index + chunk_size])


def _read_label_csv(path: Path) -> dict[str, FrameLabel]:
    labels: dict[str, FrameLabel] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"file name", "visibility", "x-coordinate", "y-coordinate"}
        missing = required_fields.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns in {path}: {sorted(missing)}")
        for row in reader:
            frame_name = str(row["file name"]).strip()
            labels[frame_name] = FrameLabel(
                visibility=float(row["visibility"] or 0.0),
                x=float(row["x-coordinate"] or 0.0),
                y=float(row["y-coordinate"] or 0.0),
            )
    return labels


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _parse_hw(value: object, *, name: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a length-2 sequence.")
    return int(value[0]), int(value[1])


def _parse_rgb(value: object, *, name: str) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must be a length-3 RGB sequence.")
    return int(value[0]), int(value[1]), int(value[2])


def _parse_float_triplet(value: object, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must be a length-3 sequence.")
    return float(value[0]), float(value[1]), float(value[2])


def _default_gif_name(clip_dir: Path) -> str:
    return f"{clip_dir.parent.name.lower()}_{clip_dir.name.lower()}_prediction.gif"


def _coord_tensor_to_tuple(coord: torch.Tensor) -> tuple[float, float]:
    return float(coord[0].item()), float(coord[1].item())

"""Video IO and runtime-config helpers for ball_detection visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.ball_detection.inference.config import build_inference_config
from src.ball_detection.inference.types import InferenceConfig, InferenceMemberConfig
from src.ball_detection.visualization.types import RuntimeConfig, VideoInputs
from src.wasb.utils.video_extractor import VideoExtractor


def _parse_color(raw: object, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return (int(raw[0]), int(raw[1]), int(raw[2]))
    return default


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _resolve_optional_path(raw: Any) -> Path | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if value == "" or value.lower() in {"none", "null"}:
        return None
    return Path(to_absolute_path(value))


def _resolve_path(path: Path) -> Path:
    return Path(to_absolute_path(str(path)))


def _resolve_optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null"}:
        return None
    return float(raw)


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Build runtime config from composed Hydra config."""
    vis = cfg.get("visualization", {}) or {}
    run = cfg.get("run", {}) or {}

    inference_cfg = build_inference_config(cfg)
    run_device = str(run.get("device", inference_cfg.device))
    single_member = InferenceMemberConfig(
        backend=inference_cfg.single_member.backend,
        checkpoint=_resolve_path(inference_cfg.single_member.checkpoint),
        model_config_path=(
            _resolve_path(inference_cfg.single_member.model_config_path)
            if inference_cfg.single_member.model_config_path is not None
            else None
        ),
        weight=inference_cfg.single_member.weight,
        score_threshold=inference_cfg.single_member.score_threshold,
    )
    ensemble_members = tuple(
        InferenceMemberConfig(
            backend=member.backend,
            checkpoint=_resolve_path(member.checkpoint),
            model_config_path=(
                _resolve_path(member.model_config_path)
                if member.model_config_path is not None
                else None
            ),
            weight=member.weight,
            score_threshold=member.score_threshold,
        )
        for member in inference_cfg.ensemble_members
    )
    inference_cfg = InferenceConfig(
        strategy=inference_cfg.strategy,
        device=_resolve_device(run_device),
        image_h=inference_cfg.image_h,
        image_w=inference_cfg.image_w,
        batch_size=inference_cfg.batch_size,
        max_frames=inference_cfg.max_frames,
        window_size=inference_cfg.window_size,
        clip_frames=inference_cfg.clip_frames,
        clip_stride=inference_cfg.clip_stride,
        visibility_threshold=inference_cfg.visibility_threshold,
        single_member=single_member,
        ensemble_members=ensemble_members,
    )

    return RuntimeConfig(
        mode=str(vis.get("mode", "visualize")),
        video_path=Path(to_absolute_path(str(vis.get("video_path", "data/samples/test.mp4")))),
        output_video_path=_resolve_optional_path(vis.get("output_video_path")),
        output_npz_path=_resolve_optional_path(vis.get("output_npz_path")),
        fps=_resolve_optional_float(vis.get("fps")),
        info=bool(vis.get("info", False)),
        radius=max(1, int(vis.get("radius", 6))),
        thickness=int(vis.get("thickness", 2)),
        color_detected_bgr=_parse_color(vis.get("color_detected_bgr"), (0, 255, 0)),
        color_trail_bgr=_parse_color(vis.get("color_trail_bgr"), (255, 255, 0)),
        show_score=bool(vis.get("show_score", True)),
        show_trail=bool(vis.get("show_trail", True)),
        trail_length=max(1, int(vis.get("trail_length", 24))),
        inference=inference_cfg,
        hydra_cfg=cfg,
    )


def load_video_inputs(cfg: RuntimeConfig) -> VideoInputs:
    """Load video frames and metadata for prediction/rendering."""
    extractor = VideoExtractor(cfg.video_path)
    frames_rgb = extractor.load_all_frames(max_frames=cfg.inference.max_frames)

    fps = cfg.fps if cfg.fps is not None else float(extractor.fps)
    if fps <= 0:
        fps = 30.0

    return VideoInputs(
        frames_rgb=frames_rgb,
        width=int(extractor.width),
        height=int(extractor.height),
        fps=float(fps),
    )

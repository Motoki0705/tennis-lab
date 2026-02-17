"""Orchestrate ball_detection prediction/visualization workflow."""

from __future__ import annotations

import logging

from src.ball_detection.visualization.adapters.predict_inputs import (
    build_ball_detection_predict_inputs,
)
from src.ball_detection.visualization.analysis.report import (
    print_prediction_summary,
    print_video_info,
    save_predictions,
)
from src.ball_detection.visualization.api.predict import predict_video
from src.ball_detection.visualization.io.video import build_runtime_config, load_video_inputs
from src.ball_detection.visualization.rendering.video_renderer import (
    OverlayRenderConfig,
    render_overlay_video,
)
from src.ball_detection.visualization.types import RuntimeConfig

logger = logging.getLogger(__name__)


def _build_overlay_config(cfg: RuntimeConfig) -> OverlayRenderConfig:
    return OverlayRenderConfig(
        radius=cfg.radius,
        thickness=cfg.thickness,
        color_detected_bgr=cfg.color_detected_bgr,
        color_trail_bgr=cfg.color_trail_bgr,
        show_score=cfg.show_score,
        show_trail=cfg.show_trail,
        trail_length=cfg.trail_length,
    )


def run_visualization(cfg: RuntimeConfig) -> int:
    """Run ball_detection visualization orchestration."""
    mode = cfg.mode.strip().lower()
    if mode not in {"visualize", "predict", "info"}:
        logger.error(
            "Error: unknown visualization.mode '%s' (expected visualize|predict|info)",
            cfg.mode,
        )
        return 1

    video_inputs = load_video_inputs(cfg)
    if cfg.info or mode == "info":
        print_video_info(video_inputs)
        return 0

    predict_inputs = build_ball_detection_predict_inputs(video_inputs)
    outputs = predict_video(inputs=predict_inputs, inference_config=cfg.inference)
    print_prediction_summary(outputs)

    if cfg.output_npz_path is not None:
        save_predictions(cfg.output_npz_path, outputs)
        print(f"Saved predictions to {cfg.output_npz_path}")

    if mode == "predict":
        return 0

    if cfg.output_video_path is None:
        logger.error("Error: visualization.output_video_path must be set for visualize mode.")
        return 1

    render_overlay_video(
        frames_rgb=predict_inputs.frames_rgb,
        predictions=outputs,
        output_path=cfg.output_video_path,
        fps=predict_inputs.fps,
        config=_build_overlay_config(cfg),
    )
    print(f"Saved overlay video to {cfg.output_video_path}")
    return 0


__all__ = ["RuntimeConfig", "build_runtime_config", "run_visualization"]

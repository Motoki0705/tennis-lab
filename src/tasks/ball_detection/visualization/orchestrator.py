"""Orchestrate ball_detection prediction/visualization workflow."""

from __future__ import annotations

import logging

import numpy as np

from src.ball_detection.inference.types import InferenceResult
from src.ball_detection.visualization.adapters.predict_inputs import (
    build_ball_detection_predict_inputs,
    build_prediction_clips,
)
from src.ball_detection.visualization.analysis.report import (
    print_prediction_summary,
    print_video_info,
    save_predictions,
)
from src.ball_detection.visualization.api.predict import build_predictor_runtime
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


def _uv_to_xy_px(ball_uv: np.ndarray, *, width: int, height: int) -> np.ndarray:
    ball_xy_px = np.asarray(ball_uv, dtype=np.float32).copy()
    if ball_xy_px.size == 0:
        return ball_xy_px
    ball_xy_px[:, 0] *= float(max(width - 1, 1))
    ball_xy_px[:, 1] *= float(max(height - 1, 1))
    return ball_xy_px


def _merge_clip_results(
    *,
    total_frames: int,
    clip_results: list[InferenceResult],
    visibility_threshold: float,
    width: int,
    height: int,
) -> InferenceResult:
    frame_indices = np.arange(total_frames, dtype=np.int64)
    sum_uv = np.zeros((total_frames, 2), dtype=np.float32)
    sum_score = np.zeros((total_frames,), dtype=np.float32)
    counts = np.zeros((total_frames,), dtype=np.float32)

    for result in clip_results:
        idx = result.frame_indices.astype(np.int64, copy=False)
        sum_uv[idx] += result.ball_uv.astype(np.float32, copy=False)
        sum_score[idx] += result.score.astype(np.float32, copy=False)
        counts[idx] += 1.0

    valid = counts > 0
    ball_uv = np.zeros((total_frames, 2), dtype=np.float32)
    score = np.zeros((total_frames,), dtype=np.float32)
    ball_uv[valid] = sum_uv[valid] / counts[valid, None]
    score[valid] = sum_score[valid] / counts[valid]

    ball_uv = np.clip(ball_uv, 0.0, 1.0).astype(np.float32, copy=False)
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
    visibility = (score >= float(visibility_threshold)).astype(bool)
    ball_xy_px = _uv_to_xy_px(ball_uv, width=width, height=height)

    return InferenceResult(
        frame_indices=frame_indices,
        ball_uv=ball_uv,
        ball_xy_px=ball_xy_px,
        visibility=visibility,
        score=score,
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
    clips = build_prediction_clips(
        predict_inputs,
        clip_frames=cfg.inference.clip_frames,
        clip_stride=cfg.inference.clip_stride,
    )

    runtime = build_predictor_runtime(inference_config=cfg.inference)
    clip_results: list[InferenceResult] = []
    for clip_idx, clip in enumerate(clips):
        clip_results.append(
            runtime.predict_clip(
                clip=clip,
                reset_state=clip_idx == 0,
                output_width=predict_inputs.width,
                output_height=predict_inputs.height,
            )
        )

    outputs = _merge_clip_results(
        total_frames=int(predict_inputs.frame_indices.shape[0]),
        clip_results=clip_results,
        visibility_threshold=float(cfg.inference.visibility_threshold),
        width=int(predict_inputs.width),
        height=int(predict_inputs.height),
    )

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

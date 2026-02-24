"""Orchestrate court detection visualization and prediction flow."""

from __future__ import annotations

import logging

import cv2

from src.tasks.court_detection.visualization.adapters.predict_inputs import build_court_predict_inputs
from src.tasks.court_detection.visualization.analysis.report import (
    print_summary,
    save_overlay,
    save_prediction_json,
)
from src.tasks.court_detection.visualization.api.predict import load_predictor, predict_keypoints
from src.tasks.court_detection.visualization.io.scene import build_runtime_config, load_scene_images
from src.tasks.court_detection.visualization.rendering.keypoints import visualize_keypoints
from src.tasks.court_detection.visualization.types import RunSummary, RuntimeConfig

logger = logging.getLogger(__name__)


def _render_config(cfg: RuntimeConfig) -> dict[str, object]:
    return {
        "point_radius": cfg.point_radius,
        "point_color": list(cfg.point_color),
        "line_color": list(cfg.line_color),
        "text_color": list(cfg.text_color),
        "line_thickness": cfg.line_thickness,
        "show_keypoint_ids": cfg.show_keypoint_ids,
        "show_court_lines": cfg.show_court_lines,
        "visibility_threshold": cfg.visibility_threshold,
    }


def run_visualization(cfg: RuntimeConfig) -> int:
    """Run court detection visualization orchestration."""
    mode = cfg.mode.strip().lower()
    if mode not in {"visualize", "predict"}:
        logger.error("Error: unknown visualization.mode '%s' (expected visualize|predict)", cfg.mode)
        return 1
    if cfg.checkpoint is None:
        logger.error("Error: visualization.checkpoint is required.")
        return 1

    scenes = load_scene_images(cfg)
    if not scenes:
        logger.error("No input images found at: %s", cfg.input_path)
        return 1

    predictor = load_predictor(checkpoint_path=cfg.checkpoint, device=cfg.device)

    failed = []
    succeeded = 0
    render_cfg = _render_config(cfg)

    for scene in scenes:
        try:
            predict_inputs = build_court_predict_inputs(scene)
            pred = predict_keypoints(predictor=predictor, inputs=predict_inputs)

            if cfg.save_json:
                save_prediction_json(cfg.output_dir, scene, pred)

            if mode == "visualize" and cfg.save_overlay:
                image_bgr = cv2.cvtColor(scene.image_rgb, cv2.COLOR_RGB2BGR)
                overlay = visualize_keypoints(
                    image_bgr,
                    pred.keypoints,
                    pred.visibility,
                    config=render_cfg,
                )
                save_overlay(cfg.output_dir, scene, overlay)

            succeeded += 1
        except Exception:  # noqa: BLE001
            logger.exception("Failed to process image: %s", scene.image_path)
            failed.append(scene.image_path)

    print_summary(
        RunSummary(
            total_inputs=len(scenes),
            succeeded=succeeded,
            failed=failed,
        )
    )
    return 0 if not failed else 1


__all__ = ["RuntimeConfig", "build_runtime_config", "run_visualization"]

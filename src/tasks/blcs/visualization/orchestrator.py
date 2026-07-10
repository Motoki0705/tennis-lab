"""Orchestrate BLCS animation and prediction comparison visualization."""

from __future__ import annotations

import logging

from src.tasks.base.visualization.orchestrator import (
    BaseVisualizationRuntimeConfig as RuntimeConfig,
)
from src.tasks.base.visualization.orchestrator import (
    build_scene_runtime_config as build_runtime_config,
)
from src.tasks.base.visualization.orchestrator import (
    save_or_show_animation,
)
from src.tasks.blcs.visualization.adapters.predict_inputs import build_predict_inputs
from src.tasks.blcs.visualization.api.predict import predict_positions
from src.tasks.blcs.visualization.io.scene import load_scene_bundle
from src.tasks.blcs.visualization.rendering import (
    BLCSSceneRenderer,
    extract_ball_events,
)

logger = logging.getLogger(__name__)

__all__ = ["RuntimeConfig", "build_runtime_config", "run_visualization"]


def run_visualization(cfg: RuntimeConfig) -> int:
    """Run BLCS visualization orchestration."""
    logger.info(f"Loading scene bundle from: {cfg.scene_path}")
    bundle = load_scene_bundle(
        scene_path=cfg.scene_path,
        camera=cfg.camera,
        cameras=cfg.cameras,
    )
    logger.info("Scene loaded successfully.")

    renderer = BLCSSceneRenderer(style=cfg.style, camera=cfg.view_3d)
    if cfg.info:
        renderer.print_scene_info(bundle.scene)
        return 0

    fps = cfg.fps or bundle.fps
    if cfg.animation_view not in {"2d", "3d"}:
        logger.error("Error: visualization.animation_view must be '2d' or '3d'.")
        return 1

    mode = cfg.mode.strip().lower()
    if mode == "predict":
        if cfg.checkpoint is None:
            logger.error("Error: visualization.checkpoint must be set for predict mode.")
            return 1
        logger.info(f"Predict mode: loading model with checkpoint: {cfg.checkpoint}")
        predict_inputs = build_predict_inputs(
            scene=bundle.scene,
            cameras=bundle.cameras,
        )
        pred_positions = predict_positions(
            checkpoint_path=cfg.checkpoint,
            device=cfg.device,
            inputs=predict_inputs,
        )
        logger.info("Creating comparison animation...")
        anim = renderer.create_comparison_animation(
            gt_positions=bundle.gt_positions,
            pred_positions=pred_positions,
            view=cfg.animation_view,
            fps=fps,
            title="GT vs Prediction",
            events=extract_ball_events(bundle.scene["meta"]),
        )
        if anim is None:
            logger.error("Error: Failed to create comparison animation.")
            return 1
    elif mode == "visualize":
        logger.info("Visualize mode: using existing scene data.")
        logger.info(f"Creating {cfg.animation_view} animation...")
        anim = renderer.create_animation(
            scene=bundle.scene,
            view=cfg.animation_view,
            fps=fps,
        )
        if anim is None:
            logger.error("Error: Failed to create visualization animation.")
            return 1
    else:
        logger.error(f"Error: unknown visualization.mode '{cfg.mode}'.")
        return 1

    save_or_show_animation(anim, cfg.save, fps)

    return 0

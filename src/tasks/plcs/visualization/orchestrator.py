"""Orchestrate PLCS animation visualization."""

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
from src.tasks.plcs.visualization.api.predict import predict_scene
from src.tasks.plcs.visualization.io.scene import load_scene_bundle
from src.tasks.plcs.visualization.rendering import PLCSSceneRenderer

logger = logging.getLogger(__name__)

__all__ = ["RuntimeConfig", "build_runtime_config", "run_visualization"]


def run_visualization(cfg: RuntimeConfig) -> int:
    """Run PLCS visualization orchestration."""
    try:
        logger.info(f"Loading scene bundle from: {cfg.scene_path}")
        bundle = load_scene_bundle(
            scene_path=cfg.scene_path,
            camera=cfg.camera,
            cameras=cfg.cameras,
        )
        logger.info(f"Scene loaded successfully. Num frames: {len(bundle.scene.position)}")
    except ValueError as exc:
        logger.error(f"Error: {exc}")
        return 1

    renderer = PLCSSceneRenderer(style=cfg.style, camera=cfg.view_3d)
    if cfg.info:
        renderer.print_scene_info(bundle.scene)
        return 0

    if cfg.animation_view not in {"3d", "2d_topdown", "camera"}:
        logger.error("Error: visualization.animation_view must be one of '3d', '2d_topdown', 'camera'.")
        return 1

    mode = cfg.mode.strip().lower()
    if mode == "predict":
        if cfg.checkpoint is None:
            logger.error("Error: visualization.checkpoint must be set for predict mode.")
            return 1
        logger.info(f"Predict mode: loading model with checkpoint: {cfg.checkpoint}")
        try:
            render_scene = predict_scene(
                checkpoint_path=cfg.checkpoint,
                device=cfg.device,
                scene=bundle.scene,
                cameras=bundle.cameras,
            )
        except ValueError as exc:
            logger.error(f"Error: {exc}")
            return 1
        if cfg.animation_view == "camera":
            logger.error(
                "Error: visualization.animation_view='camera' is not supported in predict mode. "
                "Use '3d' or '2d_topdown' for GT vs Prediction comparison."
            )
            return 1
    elif mode == "visualize":
        logger.info("Visualize mode: using existing scene data.")
        render_scene = bundle.scene
    else:
        logger.error(f"Error: unknown visualization.mode '{cfg.mode}'.")
        return 1

    fps = cfg.fps or bundle.fps
    camera_idx = bundle.cameras[0]
    logger.info(f"Creating {cfg.animation_view} animation...")
    if mode == "predict":
        anim = renderer.create_comparison_animation(
            gt_scene=bundle.scene,
            pred_scene=render_scene,
            view=cfg.animation_view,
            camera_idx=camera_idx,
            fps=fps,
            title="GT vs Prediction",
        )
    else:
        anim = renderer.create_animation(
            render_scene,
            view=cfg.animation_view,
            camera_idx=camera_idx,
            fps=fps,
        )

    save_or_show_animation(anim, cfg.save, fps)

    return 0

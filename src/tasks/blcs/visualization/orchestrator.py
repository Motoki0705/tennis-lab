"""Orchestrate BLCS animation and prediction comparison visualization."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from omegaconf import DictConfig

from src.tasks.base.visualization.orchestrator import (
    BaseVisualizationRuntimeConfig,
    build_scene_runtime_config,
    save_or_show_animation,
)
from src.tasks.blcs.configuration import (
    build_path_resolver,
    validate_visualization_boundary,
)
from src.tasks.blcs.visualization.api.predict import predict_positions
from src.tasks.blcs.visualization.io.scene import load_scene_bundle
from src.tasks.blcs.visualization.rendering import (
    BLCSSceneRenderer,
    extract_ball_events,
)
from src.utils.configuration import PathResolver

logger = logging.getLogger(__name__)

__all__ = ["RuntimeConfig", "build_runtime_config", "run_visualization"]


@dataclass(frozen=True)
class RuntimeConfig(BaseVisualizationRuntimeConfig):
    """BLCS visualization contract including its originating path resolver."""

    fps: float
    resolver: PathResolver


def build_runtime_config(config: DictConfig) -> RuntimeConfig:
    """Validate and build the BLCS visualization runtime contract."""
    validate_visualization_boundary(config)
    base = build_scene_runtime_config(config)
    if base.fps is None:
        raise RuntimeError("Validated BLCS visualization fps must be explicit.")
    return RuntimeConfig(
        mode=base.mode,
        scene_path=base.scene_path,
        checkpoint=base.checkpoint,
        device=base.device,
        animation_view=base.animation_view,
        fps=base.fps,
        save=base.save,
        camera=base.camera,
        cameras=base.cameras,
        info=base.info,
        style=base.style,
        view_3d=base.view_3d,
        resolver=build_path_resolver(config),
    )


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

    fps = cfg.fps

    mode = cfg.mode
    if mode == "predict":
        if cfg.checkpoint is None:
            raise RuntimeError("Validated predict mode requires a checkpoint.")
        logger.info(f"Predict mode: loading model with checkpoint: {cfg.checkpoint}")
        pred_positions = predict_positions(
            checkpoint_path=cfg.checkpoint,
            resolver=cfg.resolver,
            device=cfg.device,
            scene=bundle.scene,
            cameras=bundle.cameras,
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
        raise RuntimeError(f"Unexpected validated visualization mode {cfg.mode!r}.")

    save_or_show_animation(anim, cfg.save, fps)

    return 0

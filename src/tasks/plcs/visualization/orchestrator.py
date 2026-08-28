"""Orchestrate PLCS animation visualization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

from omegaconf import DictConfig

from src.tasks.base.configuration import require_config_value
from src.tasks.base.generate_dataset import CourtKeypointContract
from src.tasks.base.visualization.orchestrator import (
    BaseVisualizationRuntimeConfig,
    build_scene_runtime_config,
    save_or_show_animation,
)
from src.tasks.plcs.configuration_contracts import PLCSPathConfig
from src.tasks.plcs.court_keypoint_contract import PLCSCourtKeypointRuntimeConfig
from src.tasks.plcs.visualization.api.predict import (
    CanonicalPoseSource,
    predict_scene,
)
from src.tasks.plcs.visualization.io.scene import load_scene_bundle
from src.tasks.plcs.visualization.rendering import PLCSSceneRenderer
from src.utils.configuration import PathResolver

logger = logging.getLogger(__name__)

__all__ = ["RuntimeConfig", "build_runtime_config", "run_visualization"]


@dataclass(frozen=True)
class RuntimeConfig(BaseVisualizationRuntimeConfig):
    """PLCS visualization settings resolved from Hydra configuration."""

    canonical_pose_source: CanonicalPoseSource
    resolver: PathResolver
    court_keypoint_contract: CourtKeypointContract
    reference_camera_id: str | None


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Resolve shared settings plus the PLCS canonical pose source."""
    base = build_scene_runtime_config(
        cfg,
        visualization_extension_keys={
            "canonical_pose_source",
            "reference_camera_id",
        },
    )
    source = cast(
        "str",
        require_config_value(
            cfg.visualization,
            "canonical_pose_source",
            str,
            path="visualization",
        ),
    )
    path_config = PLCSPathConfig.from_config(cfg)
    if source not in {"gt", "prediction"}:
        raise ValueError(
            "visualization.canonical_pose_source must be 'gt' or 'prediction', "
            f"got '{source}'."
        )
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
        canonical_pose_source=cast(CanonicalPoseSource, source),
        resolver=path_config.resolver,
        court_keypoint_contract=(
            PLCSCourtKeypointRuntimeConfig.from_config(cfg).contract
        ),
        reference_camera_id=(
            None
            if cfg.visualization.reference_camera_id is None
            else str(cfg.visualization.reference_camera_id)
        ),
    )


def run_visualization(cfg: RuntimeConfig) -> int:
    """Run PLCS visualization orchestration."""
    try:
        logger.info(f"Loading scene bundle from: {cfg.scene_path}")
        bundle = load_scene_bundle(
            scene_path=cfg.scene_path,
            camera=cfg.camera,
            cameras=cfg.cameras,
            court_keypoint_contract=cfg.court_keypoint_contract,
        )
        logger.info(
            f"Scene loaded successfully. Num frames: {len(bundle.scene.position)}"
        )
    except ValueError as exc:
        logger.error(f"Error: {exc}")
        return 1

    renderer = PLCSSceneRenderer(style=cfg.style, camera=cfg.view_3d)
    if cfg.info:
        renderer.print_scene_info(bundle.scene)
        return 0

    if cfg.animation_view not in {"3d", "2d_topdown", "camera"}:
        logger.error(
            "Error: visualization.animation_view must be one of '3d', '2d_topdown', 'camera'."
        )
        return 1

    mode = cfg.mode.strip().lower()
    if mode == "predict":
        if cfg.checkpoint is None:
            logger.error(
                "Error: visualization.checkpoint must be set for predict mode."
            )
            return 1
        logger.info(f"Predict mode: loading model with checkpoint: {cfg.checkpoint}")
        try:
            render_scene = predict_scene(
                checkpoint_path=cfg.checkpoint,
                device=cfg.device,
                scene=bundle.scene,
                cameras=bundle.cameras,
                canonical_pose_source=cfg.canonical_pose_source,
                resolver=cfg.resolver,
                court_keypoint_contract=cfg.court_keypoint_contract,
                reference_camera_id=cfg.reference_camera_id,
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

    if cfg.fps is None:
        raise ValueError("PLCS visualization requires an explicit visualization.fps.")
    fps = cfg.fps
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

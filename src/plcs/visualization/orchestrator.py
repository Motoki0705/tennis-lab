"""Orchestrate PLCS animation visualization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.plcs.visualization.api.predict import predict_scene
from src.plcs.visualization.io.scene import load_scene_bundle
from src.plcs.visualization.rendering import PLCSSceneRenderer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime settings for PLCS visualization."""

    mode: str
    scene_path: Path
    checkpoint: Path | None
    device: str
    animation_view: str
    fps: float | None
    save: Path | None
    camera: int
    cameras: list[int] | str | None
    info: bool


def _resolve_device(device: str) -> str:
    """Resolve ``auto`` device selection."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _parse_cameras(raw_value: object) -> list[int] | str | None:
    """Parse Hydra camera selection value into optional list[int]."""
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped == "":
            return None
        if stripped == "all":
            return "all"
        return [int(part.strip()) for part in stripped.split(",")]
    return [int(v) for v in raw_value]


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Build runtime config from composed Hydra config."""
    vis = cfg.visualization
    run = cfg.get("run", {})
    run_device = run.get("device", vis.get("device", "auto"))

    return RuntimeConfig(
        mode=str(vis.mode),
        scene_path=Path(to_absolute_path(str(vis.scene_path))),
        checkpoint=Path(to_absolute_path(str(vis.checkpoint))) if vis.checkpoint else None,
        device=_resolve_device(str(run_device)),
        animation_view=str(vis.animation_view),
        fps=float(vis.fps) if vis.fps is not None else None,
        save=Path(to_absolute_path(str(vis.save))) if vis.save else None,
        camera=int(vis.get("camera", 0)),
        cameras=_parse_cameras(vis.get("cameras")),
        info=bool(vis.info),
    )


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

    renderer = PLCSSceneRenderer()
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
    elif mode == "visualize":
        logger.info("Visualize mode: using existing scene data.")
        render_scene = bundle.scene
    else:
        logger.error(f"Error: unknown visualization.mode '{cfg.mode}'.")
        return 1

    fps = cfg.fps or bundle.fps
    camera_idx = bundle.cameras[0]
    logger.info(f"Creating {cfg.animation_view} animation...")
    anim = renderer.create_animation(
        render_scene,
        view=cfg.animation_view,
        camera_idx=camera_idx,
        fps=fps,
    )

    if cfg.save is not None:
        cfg.save.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(cfg.save), fps=fps)
        plt.close()
        logger.info(f"Saved animation to {cfg.save}")
    else:
        plt.show()

    return 0

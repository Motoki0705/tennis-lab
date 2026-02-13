"""Orchestrate BLCS animation and prediction comparison visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.blcs.visualization.api.predict import predict_positions
from src.blcs.visualization.io.scene import load_scene_bundle
from src.blcs.visualization.rendering import BLCSSceneRenderer


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime settings for visualization orchestration."""

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
    run = cfg.run

    return RuntimeConfig(
        mode=str(vis.mode),
        scene_path=Path(to_absolute_path(str(vis.scene_path))),
        checkpoint=(
            Path(to_absolute_path(str(vis.checkpoint))) if vis.checkpoint else None
        ),
        device=_resolve_device(str(run.device)),
        animation_view=str(vis.animation_view),
        fps=float(vis.fps) if vis.fps is not None else None,
        save=Path(to_absolute_path(str(vis.save))) if vis.save else None,
        camera=int(vis.get("camera", 0)),
        cameras=_parse_cameras(vis.get("cameras")),
        info=bool(vis.info),
    )


def run_visualization(cfg: RuntimeConfig) -> int:
    """Run BLCS visualization orchestration."""
    bundle = load_scene_bundle(
        scene_path=cfg.scene_path,
        camera=cfg.camera,
        cameras=cfg.cameras,
    )

    renderer = BLCSSceneRenderer()
    if cfg.info:
        renderer.print_scene_info(bundle.scene)
        return 0

    fps = cfg.fps or bundle.fps
    if cfg.animation_view not in {"2d", "3d"}:
        print("Error: visualization.animation_view must be '2d' or '3d'.")
        return 1

    mode = cfg.mode.strip().lower()
    if mode == "predict":
        if cfg.checkpoint is None:
            print("Error: visualization.checkpoint must be set for predict mode.")
            return 1
        pred_positions = predict_positions(
            checkpoint_path=cfg.checkpoint,
            device=cfg.device,
            inputs=bundle.predict_inputs,
        )
        anim = renderer.create_comparison_animation(
            gt_positions=bundle.gt_positions,
            pred_positions=pred_positions,
            view=cfg.animation_view,
            fps=fps,
            title="GT vs Prediction",
        )
        if anim is None:
            print("Error: Failed to create comparison animation.")
            return 1
    elif mode == "visualize":
        anim = renderer.create_animation(
            scene=bundle.scene,
            view=cfg.animation_view,
            fps=fps,
        )
        if anim is None:
            print("Error: Failed to create visualization animation.")
            return 1
    else:
        print(f"Error: unknown visualization.mode '{cfg.mode}'.")
        return 1

    if cfg.save is not None:
        cfg.save.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(cfg.save), fps=fps)
        plt.close()
        print(f"Saved comparison animation to {cfg.save}")
    else:
        plt.show()

    return 0

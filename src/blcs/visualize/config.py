"""Configuration handling for BLCS visualization."""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@dataclass(frozen=True)
class VisualizationConfig:
    """Resolved runtime configuration for BLCS visualization."""

    mode: str  # visualize | predict | predict-multiview
    scene_path: Path
    frame: int
    animation_view: str  # 2d_camera | 3d
    camera: int | None  # For single camera mode
    cameras: list[int]  # For multiview mode
    fps: float | None
    save: Path | None
    save_input: Path | None
    info: bool
    checkpoint: str | None
    device: str
    output: str | None


def _resolve_device(device: str) -> str:
    """Resolve device string (auto -> cuda/cpu)."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _validate_animation_view(animation_view: str) -> str:
    """Validate and normalize animation_view.
    
    Only '2d_camera' and '3d' are allowed.
    Maps deprecated values with warnings.
    """
    # Normalize underscores/hyphens
    normalized = animation_view.replace("-", "_").lower()
    
    # Map deprecated values
    deprecated_mapping = {
        "2d": "2d_camera",
        "camera": "2d_camera",
    }
    
    if normalized in deprecated_mapping:
        new_value = deprecated_mapping[normalized]
        warnings.warn(
            f"animation_view='{animation_view}' is deprecated. "
            f"Use '{new_value}' instead. "
            f"Only '2d_camera' and '3d' are supported.",
            DeprecationWarning,
            stacklevel=3,
        )
        return new_value
    
    # Validate allowed values
    allowed = {"2d_camera", "3d"}
    if normalized not in allowed:
        print(
            f"Error: animation_view='{animation_view}' is not supported. "
            f"Only '2d_camera' and '3d' are allowed.",
            file=sys.stderr,
        )
        sys.exit(1)
    
    return normalized


def build_visualization_config(cfg: DictConfig) -> VisualizationConfig:
    """Build VisualizationConfig from Hydra DictConfig.
    
    Args:
        cfg: Hydra composed configuration.
        
    Returns:
        Validated and resolved configuration.
    """
    vis = cfg.visualization
    run = cfg.run
    
    # Handle deprecated 'view' field
    if hasattr(vis, "view") and vis.view is not None:
        view_value = str(vis.view)
        if view_value != "animation":
            warnings.warn(
                f"visualization.view='{view_value}' is deprecated. "
                "Only animation is supported. All output is now animation-based. "
                "Use visualization.animation_view to control the view type.",
                DeprecationWarning,
                stacklevel=2,
            )
    
    # Parse cameras (multiview support)
    cameras_raw = vis.get("cameras", None)
    if cameras_raw == "all":
        cameras = []  # Will be resolved based on scene
    elif cameras_raw is None:
        cameras = []
    elif isinstance(cameras_raw, str):
        cameras = [int(c.strip()) for c in cameras_raw.split(",")]
    else:
        cameras = list(cameras_raw)
    
    # Get camera for single-camera mode
    camera = int(vis.get("camera", 0)) if "camera" in vis else None
    
    # Validate animation_view
    animation_view = _validate_animation_view(str(vis.animation_view))
    
    return VisualizationConfig(
        mode=str(vis.mode),
        scene_path=Path(to_absolute_path(str(vis.scene_path))),
        frame=int(vis.frame),
        animation_view=animation_view,
        camera=camera,
        cameras=cameras,
        fps=float(vis.fps) if vis.fps is not None else None,
        save=Path(to_absolute_path(str(vis.save))) if vis.save else None,
        save_input=(
            Path(to_absolute_path(str(vis.save_input))) if vis.save_input else None
        ),
        info=bool(vis.info),
        checkpoint=(
            to_absolute_path(str(vis.checkpoint)) if vis.checkpoint else None
        ),
        device=_resolve_device(str(run.device)),
        output=to_absolute_path(str(vis.output)) if vis.output else None,
    )

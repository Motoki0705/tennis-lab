"""Unified BLCS visualization entry point (Hydra-based).

This script provides a single entry point for all BLCS visualization tasks:
- Ground-truth scene visualization
- Single-camera prediction
- Multi-camera prediction

All domain logic is delegated to the src.blcs.visualize module.

Example commands:
    # Visualize ground-truth scene
    `uv run python -m src.blcs.scripts.visualize`
    `uv run python -m src.blcs.scripts.visualize visualization.scene_path=data/blcs/scenes/scene_000000.npz`
    
    # Single-camera prediction
    `uv run python -m src.blcs.scripts.visualize visualization.mode=predict visualization.checkpoint=outputs/blcs/single/logs/version_0/checkpoints/last.ckpt`
    
    # Multi-camera prediction
    `uv run python -m src.blcs.scripts.visualize visualization.mode=predict visualization.cameras=all visualization.checkpoint=outputs/blcs/multiview/logs/version_0/checkpoints/last.ckpt`

Config entry point: `src/blcs/configs/visualize.yaml`
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, TypeVar, cast

import hydra
from omegaconf import DictConfig

from src.blcs.visualize.config import build_visualization_config
from src.blcs.visualize.usecases import visualize_prediction, visualize_scene

F = TypeVar("F", bound=Callable[..., Any])


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for hydra.main to keep mypy satisfied."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))


@hydra_main(config_path="../configs", config_name="visualize", version_base="1.3")
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point for BLCS visualization.
    
    Dispatches to appropriate use case based on mode:
    - visualize: Ground-truth scene visualization
    - predict: Single or multi-camera prediction
    """
    # Build configuration (handles validation and deprecation warnings)
    vis_cfg = build_visualization_config(cfg)
    
    # Dispatch to appropriate use case
    if vis_cfg.mode == "visualize":
        return visualize_scene(vis_cfg)
    elif vis_cfg.mode in {"predict", "predict-multiview", "predict_multiview"}:
        return visualize_prediction(vis_cfg)
    else:
        print(
            f"Error: unknown visualization.mode '{vis_cfg.mode}' "
            "(expected visualize|predict)"
        )
        return 1


if __name__ == "__main__":
    sys.exit(cast(Callable[[], int], main)())

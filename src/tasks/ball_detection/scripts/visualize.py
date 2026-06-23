"""Visualize ball detection predictions for one clip.

Usage:
    python -m src.tasks.ball_detection.scripts.visualize
    python -m src.tasks.ball_detection.scripts.visualize visualization.clip_dir=data/tennis/tracknet/game1/Clip1
    python -m src.tasks.ball_detection.scripts.visualize visualization.save=assets/ball_detection/custom_clip.gif

Notes:
    - Hydra loads configuration from `src/tasks/ball_detection/configs/visualize.yaml`.
    - The script runs sliding-window inference over one clip and saves an animated GIF.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, TypeVar, cast

import hydra
from omegaconf import DictConfig

from src.tasks.ball_detection.visualization.orchestrator import (
    build_runtime_config,
    run_visualization,
)

F = TypeVar("F", bound=Callable[..., Any])


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for ``hydra.main``."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))


@hydra_main(config_path="../configs", config_name="visualize", version_base="1.3")
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    runtime = build_runtime_config(cfg)
    return int(run_visualization(runtime))


if __name__ == "__main__":
    sys.exit(cast(Callable[[], int], main)())

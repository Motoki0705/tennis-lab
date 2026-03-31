"""Run ball multi-task visualization orchestration with Hydra configuration.

Usage:
    python -m src.developing.ball_multitask.scripts.visualize
    python -m src.developing.ball_multitask.scripts.visualize run.dry_run=true

Notes:
    - Configuration is loaded from `src/developing/ball_multitask/configs/visualize.yaml`.
    - Hydra handles runtime overrides.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, TypeVar, cast

import hydra
from omegaconf import DictConfig

from src.developing.ball_multitask.visualization.orchestrator import (
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
    return run_visualization(runtime)


if __name__ == "__main__":
    sys.exit(cast(Callable[[], int], main)())

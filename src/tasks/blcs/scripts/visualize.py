"""Visualize BLCS reconstruction or inference outputs.

Usage:
    python -m src.tasks.blcs.scripts.visualize
    python -m src.tasks.blcs.scripts.visualize output_dir=outputs/blcs/preview

Notes:
    - Hydra loads configuration from `src/tasks/blcs/configs/visualize.yaml`.
    - The script builds a runtime visualization config and forwards it to the orchestrator.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, TypeVar, cast

import hydra
from omegaconf import DictConfig

from src.tasks.blcs.visualization.orchestrator import (
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

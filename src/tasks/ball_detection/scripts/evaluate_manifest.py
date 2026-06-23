"""Evaluate multiple ball-detector checkpoints and generate comparison reports.

Usage:
    python -m src.tasks.ball_detection.scripts.evaluate_manifest
    python -m src.tasks.ball_detection.scripts.evaluate_manifest manifest_path=path/to/manifest.yaml

Notes:
    - Hydra loads CLI settings from `src/tasks/ball_detection/configs/evaluate_manifest.yaml`.
    - The manifest defines checkpoints, fixed val/test datasets, comparison categories, and resume behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import hydra
from omegaconf import DictConfig

from src.tasks.ball_detection.evaluation import (
    EvaluationPipeline,
    load_evaluation_manifest,
)

FMain = TypeVar("FMain", bound=Callable[..., Any])


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[FMain], FMain]:
    """Typed wrapper for ``hydra.main``."""
    return cast(Callable[[FMain], FMain], hydra.main(*args, **kwargs))


@hydra_main(
    config_path="../configs",
    config_name="evaluate_manifest",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Run all evaluation jobs in the configured manifest."""
    manifest = load_evaluation_manifest(str(cfg.manifest_path))
    pipeline = EvaluationPipeline(
        manifest,
        output_dir=(
            None if cfg.get("output_dir") is None else Path(str(cfg.output_dir))
        ),
        resume=(
            None if cfg.get("resume") is None else bool(cfg.get("resume"))
        ),
        fail_fast=(
            None
            if cfg.get("fail_fast") is None
            else bool(cfg.get("fail_fast"))
        ),
    )
    summary = pipeline.run()
    print(
        "Evaluation complete: "
        f"jobs={summary['jobs']} executed={summary['executed']} "
        f"reused={summary['reused']} failed={summary['failed']}"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(cast(Callable[[], int], main)())

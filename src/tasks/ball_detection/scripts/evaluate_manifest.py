"""Evaluate multiple ball-detector checkpoints and generate comparison reports.

Usage:
    python -m src.tasks.ball_detection.scripts.evaluate_manifest
    python -m src.tasks.ball_detection.scripts.evaluate_manifest manifest_path=path/to/manifest.yaml

Notes:
    - Hydra loads CLI settings from `src/tasks/ball_detection/configs/evaluate_manifest.yaml`.
    - The manifest defines checkpoints, fixed val/test datasets, comparison categories, and resume behavior.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.ball_detection import configuration as _configuration  # noqa: F401
from src.tasks.ball_detection.configuration import BallRuntimePaths
from src.tasks.ball_detection.evaluation import (
    EvaluationPipeline,
    load_evaluation_manifest,
)
from src.tasks.ball_detection.evaluation.adapters import resolve_evaluation_device
from src.tasks.ball_detection.evaluation.evaluator import DefaultJobEvaluator
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="evaluate_manifest",
    version_base="1.3",
    validation_boundary="ball.evaluate_manifest",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Run all evaluation jobs in the configured manifest."""
    paths = BallRuntimePaths.from_config(cfg)
    manifest = load_evaluation_manifest(
        paths.project(str(cfg.manifest_path)), resolver=paths.resolver
    )
    pipeline = EvaluationPipeline(
        manifest,
        evaluator=DefaultJobEvaluator(
            device=resolve_evaluation_device(manifest.device)
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
    raise SystemExit(main())

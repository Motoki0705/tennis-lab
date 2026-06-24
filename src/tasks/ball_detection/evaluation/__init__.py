"""Automated, reproducible evaluation for ball-detection checkpoints."""

from src.tasks.ball_detection.evaluation.contracts import (
    DatasetSpec,
    EvaluationManifest,
    ModelSpec,
    load_evaluation_manifest,
)
from src.tasks.ball_detection.evaluation.runner import EvaluationPipeline

__all__ = [
    "DatasetSpec",
    "EvaluationManifest",
    "EvaluationPipeline",
    "ModelSpec",
    "load_evaluation_manifest",
]

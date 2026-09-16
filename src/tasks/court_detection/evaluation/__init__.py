"""CPU-only cross-model court keypoint and alignment benchmark."""

from __future__ import annotations

from src.tasks.court_detection.evaluation.contracts import (
    DOMAIN_NAMES,
    KEYPOINT_COUNT,
    MODEL_NAMES,
    DomainName,
    KeypointPrediction,
    LoadedSample,
    ModelName,
    ModelPrediction,
    SampleRef,
)
from src.tasks.court_detection.evaluation.metrics import (
    SampleEvaluation,
    aggregate_evaluations,
    evaluate_sample,
)
from src.tasks.court_detection.evaluation.settings import BenchmarkSettings

__all__ = [
    "DOMAIN_NAMES",
    "KEYPOINT_COUNT",
    "MODEL_NAMES",
    "BenchmarkSettings",
    "DomainName",
    "KeypointPrediction",
    "LoadedSample",
    "ModelName",
    "ModelPrediction",
    "SampleEvaluation",
    "SampleRef",
    "aggregate_evaluations",
    "evaluate_sample",
]

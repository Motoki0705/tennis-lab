"""Court detection models."""

from __future__ import annotations

from torch import nn

from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel

__all__ = [
    "CourtHierarchicalModel",
    "build_court_detection_model",
]


def build_court_detection_model(config: object) -> nn.Module:
    """Build a court detection model from config.

    The number of output channels comes from ``config.model.num_classes``.
    A mismatch against the selected data task is rejected early.
    """
    runtime = CourtTrainingConfig.from_config(config)
    return CourtHierarchicalModel.from_config(runtime.model)

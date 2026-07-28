"""Configuration-selectable court camera and label algorithms."""

from __future__ import annotations

from src.synthetic_data_generation.dataset.algorithms import (
    AlgorithmDefinition,
    AlgorithmRegistry,
)

CAMERA_SAMPLING_ALGORITHMS = AlgorithmRegistry(
    namespace="court.camera_sampling",
    definitions=(
        AlgorithmDefinition(
            name="sfm_neighborhood",
            implementation="sfm_neighborhood",
            description=(
                "Sample support-bounded perturbations around captured SfM cameras."
            ),
        ),
        AlgorithmDefinition(
            name="inward_orbit",
            implementation="inward_orbit",
            description=(
                "Sample inward-looking circle and ellipse families at configurable "
                "scales, heights, and court targets."
            ),
        ),
    ),
)

LABEL_ALGORITHMS = AlgorithmRegistry(
    namespace="court.labels",
    definitions=(
        AlgorithmDefinition(
            name="symmetric_seven_channel",
            implementation="symmetric_seven_channel",
            description=(
                "Collapse fourteen physical points to seven near/far-symmetric "
                "multi-peak heatmap channels while preserving court instances."
            ),
        ),
    ),
)

__all__ = ["CAMERA_SAMPLING_ALGORITHMS", "LABEL_ALGORITHMS"]

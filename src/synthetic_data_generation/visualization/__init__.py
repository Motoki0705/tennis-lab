"""Canonical 3DGS-backed generated-dataset visualization API."""

from src.synthetic_data_generation.visualization.contracts import (
    VISUALIZATION_METADATA_SCHEMA,
    DatasetVisualizationDomain,
    DatasetVisualizationRequest,
    DatasetVisualizationResult,
)
from src.synthetic_data_generation.visualization.renderer import visualize_dataset

__all__ = [
    "DatasetVisualizationDomain",
    "DatasetVisualizationRequest",
    "DatasetVisualizationResult",
    "VISUALIZATION_METADATA_SCHEMA",
    "visualize_dataset",
]

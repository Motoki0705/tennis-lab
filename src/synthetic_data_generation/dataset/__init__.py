"""Extensible 3DGS-native synthetic-dataset pipelines."""

from src.synthetic_data_generation.dataset.registry import (
    DatasetPipelineDefinition,
    available_dataset_pipelines,
    get_dataset_pipeline,
)

__all__ = [
    "DatasetPipelineDefinition",
    "available_dataset_pipelines",
    "get_dataset_pipeline",
]

"""Generic path-driven synthetic-dataset pipeline."""

from src.synthetic_data_generation.dataset.execution import execute_pipeline
from src.synthetic_data_generation.dataset.pipeline import PathPipelineManifest

__all__ = [
    "PathPipelineManifest",
    "execute_pipeline",
]

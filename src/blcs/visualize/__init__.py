"""BLCS visualization module.

This module provides unified visualization functionality for BLCS:
- Scene loading and saving
- Animation rendering (2d_camera, 3d views)
- Prediction execution and comparison
- Single entry point for all visualization tasks

Public API:
    - visualize_scene: Visualize ground-truth scenes
    - visualize_prediction: Run model prediction and visualize
    - VisualizationConfig: Configuration dataclass
"""

from src.blcs.visualize.config import VisualizationConfig
from src.blcs.visualize.usecases import visualize_prediction, visualize_scene

__all__ = [
    "VisualizationConfig",
    "visualize_scene",
    "visualize_prediction",
]

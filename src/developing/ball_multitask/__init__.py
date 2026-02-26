"""Ball multi-task models for UV completion, event detection, and 3D trajectory.

This package provides a shared Transformer backbone with task-specific heads
for offline inference and training across UV/3D tasks.
"""

from src.developing.ball_multitask.models.multitask_model import BallMultitaskModel

__all__ = ["BallMultitaskModel"]
__version__ = "0.1.0"

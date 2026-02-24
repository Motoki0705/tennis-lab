"""Reusable model components for ball detection."""

from src.tasks.ball_detection.models.components.downsample import BasicBlock
from src.tasks.ball_detection.models.components.heads import VisibilityHead, XYHead

__all__ = [
    "BasicBlock",
    "XYHead",
    "VisibilityHead",
]

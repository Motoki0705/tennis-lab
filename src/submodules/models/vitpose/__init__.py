"""2D pose model (ViTPose-H)."""

from src.submodules.models.vitpose.pose2d import (
    DEFAULT_VITPOSE_CHECKPOINT,
    Pose2DRequest,
    Pose2DResult,
    ViTPosePose2D,
)

__all__ = [
    "DEFAULT_VITPOSE_CHECKPOINT",
    "Pose2DRequest",
    "Pose2DResult",
    "ViTPosePose2D",
]

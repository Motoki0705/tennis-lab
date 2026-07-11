"""SLCS model components: output heads and the DINOv3 token encoder."""

from src.tasks.slcs.models.components.dino_adapter import DinoTokenEncoder
from src.tasks.slcs.models.components.heads import (
    BallPositionHead,
    LogScaleHead,
    PlayerPositionHead,
    PlayerRotationHead,
)

__all__ = [
    "BallPositionHead",
    "DinoTokenEncoder",
    "LogScaleHead",
    "PlayerPositionHead",
    "PlayerRotationHead",
]

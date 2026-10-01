"""Inference with checkpoints exported by the player-detection task."""

from src.submodules.models.dino.person_detector import (
    PersonDetectionRequest as PlayerDetectionRequest,
)
from src.submodules.models.dino.person_detector import (
    PersonDetectionResult as PlayerDetectionResult,
)
from src.tasks.player_detection.inference.checkpoint import (
    PlayerCheckpointInfo,
    inspect_player_checkpoint,
)
from src.tasks.player_detection.inference.predictor import DinoPlayerDetector

__all__ = [
    "DinoPlayerDetector",
    "PlayerCheckpointInfo",
    "PlayerDetectionRequest",
    "PlayerDetectionResult",
    "inspect_player_checkpoint",
]

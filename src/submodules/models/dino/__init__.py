"""DINO person detector public API."""

from src.submodules.models.dino.person_detector import (
    DEFAULT_DINO_CHECKPOINT,
    DinoPersonDetector,
    PersonDetectionRequest,
    PersonDetectionResult,
)

__all__ = [
    "DEFAULT_DINO_CHECKPOINT",
    "DinoPersonDetector",
    "PersonDetectionRequest",
    "PersonDetectionResult",
]

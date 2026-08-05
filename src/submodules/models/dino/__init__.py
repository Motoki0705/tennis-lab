"""DINO person detector public API."""

from src.submodules.models.dino.person_detector import (
    DinoPersonDetector,
    PersonDetectionRequest,
    PersonDetectionResult,
)

__all__ = [
    "DinoPersonDetector",
    "PersonDetectionRequest",
    "PersonDetectionResult",
]

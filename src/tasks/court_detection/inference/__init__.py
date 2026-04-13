"""Inference components for court detection."""

from src.tasks.court_detection.inference.predictor import (
    CourtDetectionPredictor,
    CourtKeypointPredictor,
)

__all__ = ["CourtDetectionPredictor", "CourtKeypointPredictor"]  # noqa: F401

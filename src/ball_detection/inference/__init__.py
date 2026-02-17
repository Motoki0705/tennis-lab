"""Inference entry points for ball_detection."""

from src.ball_detection.inference.ensemble_predictor import BallEnsemblePredictor
from src.ball_detection.inference.predictor import BallPredictor

__all__ = ["BallPredictor", "BallEnsemblePredictor"]

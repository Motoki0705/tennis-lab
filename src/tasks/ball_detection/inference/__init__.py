"""Inference interfaces for ball detection."""

from src.tasks.ball_detection.inference.predictor import BallDetectionPredictor
from src.tasks.ball_detection.inference.trajectory_gate import (
    TrajectoryGateConfig,
    TrajectoryGateDiagnostics,
    TrajectoryGateRejection,
    apply_trajectory_gate,
)

__all__ = [
    "BallDetectionPredictor",
    "TrajectoryGateConfig",
    "TrajectoryGateDiagnostics",
    "TrajectoryGateRejection",
    "apply_trajectory_gate",
]

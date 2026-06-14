"""Input adapters for ball-detection visualization prediction."""

from src.tasks.ball_detection.visualization.adapters.predict_inputs import (
    build_window_starts,
    chunked,
    iter_window_batches,
)

__all__ = ["build_window_starts", "chunked", "iter_window_batches"]

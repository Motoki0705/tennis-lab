"""Input adapters for event detection visualization prediction."""

from src.tasks.event_detection.visualization.adapters.predict_inputs import (
    Traj3DPredictInputs,
    UVPredictInputs,
    build_traj3d_predict_inputs,
    build_uv_predict_inputs,
)

__all__ = [
    "Traj3DPredictInputs",
    "UVPredictInputs",
    "build_uv_predict_inputs",
    "build_traj3d_predict_inputs",
]

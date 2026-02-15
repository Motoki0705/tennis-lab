"""Prediction API for trajectory completion visualization."""

from __future__ import annotations

from src.trajectory_completion.visualization.adapters.predict_inputs import (
    UVCompletionPredictInputs,
)
from src.trajectory_completion.inference.uv_predictor import UVTrajectoryCompletionPredictor


def predict_uv_completion(
    *,
    checkpoint_path: str,
    device: str,
    inputs: UVCompletionPredictInputs,
) -> dict[str, object]:
    """Run trajectory completion model prediction and normalize outputs."""
    predictor = UVTrajectoryCompletionPredictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    outputs = predictor.predict(
        ball_uv_in=inputs.ball_uv_in,
        ball_obs_mask=inputs.ball_obs_mask,
        court_kp=inputs.court_kp,
        court_vis=inputs.court_vis,
        merge_observed=True,
    )

    pred_uv = outputs["ball_uv_pred"].squeeze(0).cpu().numpy()
    completed_uv = (
        outputs.get("ball_uv_completed", outputs["ball_uv_pred"]).squeeze(0).cpu().numpy()
    )

    return {
        "raw": outputs,
        "pred_uv": pred_uv,
        "completed_uv": completed_uv,
    }

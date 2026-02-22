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
    merge_observed: bool,
    in_frame_threshold: float,
    cut_out_of_frame: bool,
) -> dict[str, object]:
    """Run trajectory completion model prediction and normalize outputs."""
    predictor = UVTrajectoryCompletionPredictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    outputs = predictor.predict(
        ball_uv=inputs.ball_uv,
        court_kp=inputs.court_kp,
        ball_vis=inputs.ball_vis,
        ball_mask=inputs.ball_mask,
        court_vis=inputs.court_vis,
        merge_observed=bool(merge_observed),
        in_frame_threshold=float(in_frame_threshold),
        cut_out_of_frame=bool(cut_out_of_frame),
    )

    pred_uv = outputs["ball_uv_pred"].squeeze(0).cpu().numpy()
    completed_uv = (
        outputs.get("ball_uv_completed", outputs["ball_uv_pred"]).squeeze(0).cpu().numpy()
    )
    in_frame_probs = outputs["in_frame_probs"].squeeze(0).cpu().numpy()
    in_frame_pred = outputs["in_frame_pred"].squeeze(0).cpu().numpy()

    return {
        "raw": outputs,
        "pred_uv": pred_uv,
        "completed_uv": completed_uv,
        "in_frame_probs": in_frame_probs,
        "in_frame_pred": in_frame_pred,
    }

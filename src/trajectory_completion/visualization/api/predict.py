"""Prediction API for trajectory completion visualization."""

from __future__ import annotations

import numpy as np
import torch

from src.trajectory_completion.inference.uv_predictor import UVTrajectoryCompletionPredictor
from src.trajectory_completion.visualization.types import TrajectoryInputs


def predict_uv_completion(
    *,
    checkpoint_path: str,
    device: str,
    inputs: TrajectoryInputs,
) -> dict[str, object]:
    """Run trajectory completion model prediction and normalize outputs."""
    predictor = UVTrajectoryCompletionPredictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    outputs = predictor.predict(
        ball_uv_in=torch.from_numpy(inputs.ball_uv_in).float(),
        ball_obs_mask=torch.from_numpy(inputs.ball_obs_mask.astype(np.float32)),
        court_kp=torch.from_numpy(inputs.court_kp).float(),
        court_vis=torch.from_numpy(inputs.court_vis.astype(np.float32)),
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

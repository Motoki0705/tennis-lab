"""Prediction API for event detection visualization."""

from __future__ import annotations

from typing import Any

from src.event_detection.visualization.adapters.predict_inputs import (
    Traj3DPredictInputs,
    UVPredictInputs,
)
from src.event_detection.inference.traj3d_predictor import Traj3DEventPredictor
from src.event_detection.inference.uv_predictor import UVEventPredictor


def predict_uv_events(
    *,
    checkpoint_path: str,
    device: str,
    inputs: UVPredictInputs,
    threshold: float,
    min_distance: int,
    top_k: int | None,
) -> dict[str, Any]:
    """Run UV event detector prediction and normalize outputs for rendering."""
    predictor = UVEventPredictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    outputs = predictor.predict(
        ball_uv=inputs.ball_uv,
        court_kp=inputs.court_kp,
        ball_vis=inputs.ball_vis,
        court_vis=inputs.court_vis,
        threshold=float(threshold),
        min_distance=int(min_distance),
        top_k=top_k,
    )

    return {
        "raw": outputs,
        "probs": outputs["event_probs"].squeeze(0).cpu().numpy(),
        "peaks": outputs["event_peaks"][0],
        "scores": outputs["event_peak_scores"][0],
        "names": outputs.get("event_names"),
    }


def predict_traj3d_events(
    *,
    checkpoint_path: str,
    device: str,
    inputs: Traj3DPredictInputs,
    threshold: float,
    min_distance: int,
    top_k: int | None,
) -> dict[str, Any]:
    """Run 3D event detector prediction and normalize outputs for rendering."""
    predictor = Traj3DEventPredictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    outputs = predictor.predict(
        ball_pos_world=inputs.ball_pos_world,
        threshold=float(threshold),
        min_distance=int(min_distance),
        top_k=top_k,
    )

    return {
        "raw": outputs,
        "probs": outputs["event_probs"].squeeze(0).cpu().numpy(),
        "peaks": outputs["event_peaks"][0],
        "scores": outputs["event_peak_scores"][0],
        "names": outputs.get("event_names"),
    }

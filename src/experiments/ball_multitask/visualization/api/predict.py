"""Prediction API for ball multitask visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.ball_multitask.inference.predictor import BallMultitaskPredictor
from src.experiments.ball_multitask.visualization.adapters.predict_inputs import BallMultitaskPredictInputs


def load_predictor(*, checkpoint_path: Path, device: str) -> BallMultitaskPredictor:
    """Load ball multitask predictor from checkpoint."""
    return BallMultitaskPredictor.load_from_checkpoint(checkpoint_path=checkpoint_path, device=device)


def predict_scene(
    *,
    predictor: BallMultitaskPredictor,
    inputs: BallMultitaskPredictInputs,
    threshold: float,
    min_distance: int,
    top_k: int | None,
    denormalize: bool,
    in_frame_threshold: float,
    cut_out_of_frame: bool,
) -> dict[str, Any]:
    """Run sequence prediction and return numpy-friendly outputs."""
    outputs = predictor.predict(
        inputs.ball_uv,
        inputs.court_kp,
        ball_vis=inputs.ball_vis,
        court_vis=inputs.court_vis,
        seq_len=inputs.seq_len,
        threshold=threshold,
        min_distance=min_distance,
        top_k=top_k,
        denormalize=denormalize,
        in_frame_threshold=in_frame_threshold,
        cut_out_of_frame=cut_out_of_frame,
    )

    peaks = outputs["event_peaks"]
    peak_scores = outputs["event_peak_scores"]
    if peaks and isinstance(peaks[0], list):
        peaks = peaks[0]
    if peak_scores and isinstance(peak_scores[0], list):
        peak_scores = peak_scores[0]

    return {
        "uv_completed": outputs["uv_completed"].squeeze(0).numpy(),
        "position_3d": outputs["position_3d"].squeeze(0).numpy(),
        "event_logits": outputs["event_logits"].squeeze(0).numpy(),
        "event_probs": outputs["event_probs"].squeeze(0).numpy(),
        "in_frame_logits": outputs["in_frame_logits"].squeeze(0).numpy(),
        "in_frame_probs": outputs["in_frame_probs"].squeeze(0).numpy(),
        "in_frame_pred": outputs["in_frame_pred"].squeeze(0).numpy(),
        "event_peaks": peaks,
        "event_peak_scores": peak_scores,
        "event_names": list(outputs["event_names"]),
        "raw": outputs,
    }

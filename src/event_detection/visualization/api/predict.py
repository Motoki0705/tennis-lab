"""Prediction API for event detection visualization."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.event_detection.inference.traj3d_predictor import Traj3DEventPredictor
from src.event_detection.inference.uv_predictor import UVEventPredictor
from src.event_detection.visualization.types import Traj3DEventInputs, UVEventInputs


def predict_uv_events(
    *,
    checkpoint_path: str,
    device: str,
    inputs: UVEventInputs,
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
        ball_uv=torch.from_numpy(inputs.ball_uv).float(),
        court_kp=torch.from_numpy(inputs.court_kp).float(),
        ball_vis=torch.from_numpy(inputs.ball_vis.astype(np.float32)),
        court_vis=torch.from_numpy(inputs.court_vis.astype(np.float32)),
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
    inputs: Traj3DEventInputs,
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
        ball_pos_world=torch.from_numpy(inputs.ball_pos_world).float(),
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

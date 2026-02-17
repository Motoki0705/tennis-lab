"""Trajectory completion wrapper for pseudo-label refinement."""

from __future__ import annotations

import torch
from torch import Tensor

from src.trajectory_completion.inference import UVTrajectoryCompletionPredictor


class TrajectoryRefiner:
    """Refine sparse UV detections into temporally completed trajectories."""

    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu") -> None:
        self.predictor: UVTrajectoryCompletionPredictor | None = None
        if checkpoint_path:
            self.predictor = UVTrajectoryCompletionPredictor.load_from_checkpoint(
                checkpoint_path,
                device=device,
            )

    def refine(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        *,
        ball_vis: Tensor,
        ball_mask: Tensor,
        court_vis: Tensor | None = None,
    ) -> Tensor:
        """Return completed trajectory; fall back to input when predictor is absent."""
        if self.predictor is None:
            return ball_uv

        outputs = self.predictor.predict(
            ball_uv=ball_uv,
            court_kp=court_kp,
            ball_vis=ball_vis,
            ball_mask=ball_mask,
            court_vis=court_vis,
            merge_observed=True,
        )
        if "ball_uv_completed" in outputs:
            return outputs["ball_uv_completed"]
        return outputs["ball_uv_pred"]

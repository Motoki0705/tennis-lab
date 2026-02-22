"""Trajectory completion wrapper for pseudo-label refinement."""

from __future__ import annotations

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
        court_kp: Tensor | None = None,
        *,
        ball_vis: Tensor,
        ball_mask: Tensor,
        court_vis: Tensor | None = None,
    ) -> Tensor:
        """Return completed trajectory; fall back to input when predictor is absent.

        Args:
            ball_uv: Input ball UV sequence.
            court_kp: Optional court keypoints. Required only for court-aware models.
            ball_vis: Ball visibility/observation mask.
            ball_mask: Valid-length mask.
            court_vis: Optional court visibility mask.
        """
        if self.predictor is None:
            return ball_uv

        outputs = self.predictor.predict(
            ball_uv=ball_uv,
            court_kp=court_kp,
            ball_vis=ball_vis,
            ball_mask=ball_mask,
            court_vis=court_vis,
            merge_observed=False,
            in_frame_threshold=0.5,
            cut_out_of_frame=True,
        )
        return outputs["ball_uv_pred"]

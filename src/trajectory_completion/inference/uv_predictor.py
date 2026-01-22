"""UV trajectory completion inference predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from src.base.api.predictor import BasePredictor
from src.trajectory_completion.models.uv_completion_model import UVTrajectoryCompletionModel
from src.trajectory_completion.training.lightning_module import (
    TrajectoryCompletionLightningModule,
)


class UVTrajectoryCompletionPredictor(BasePredictor):
    """UV trajectory completion inference predictor.

    Args:
        model: UVTrajectoryCompletionModel instance.
        device: Inference device.
    """

    def __init__(self, model: UVTrajectoryCompletionModel, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        """Create a UVTrajectoryCompletionPredictor from a Lightning checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (.ckpt).
            device: Inference device.
            **kwargs: Unused (for compatibility).

        Returns:
            Initialized UVTrajectoryCompletionPredictor instance.
        """
        _ = kwargs
        checkpoints = cls._ensure_checkpoint(checkpoint_path)
        if len(checkpoints) != 1:
            raise ValueError("UVTrajectoryCompletionPredictor expects a single checkpoint path.")
        device = cls._resolve_device(device)
        lightning_module = TrajectoryCompletionLightningModule.load_from_checkpoint(
            checkpoints[0],
            map_location=device,
        )
        return cls(model=lightning_module.model, device=device)

    @torch.no_grad()
    def predict(
        self,
        ball_uv_in: Tensor,
        ball_obs_mask: Tensor,
        court_kp: Tensor,
        court_vis: Tensor | None = None,
        seq_len: Tensor | None = None,
        *,
        merge_observed: bool = True,
    ) -> dict[str, Tensor]:
        """Complete missing ball UV trajectory frames.

        Args:
            ball_uv_in: Corrupted inputs. Shape (B, T, 2) or (T, 2).
            ball_obs_mask: Observed mask (1=observed). Shape (B, T) or (T,).
            court_kp: Court keypoints. Shape (B, 20, 2) or (20, 2).
            court_vis: Court visibility mask. Shape (B, 20) or (20,).
            seq_len: Optional sequence lengths. Shape (B,) or scalar.
            merge_observed: If True, keep observed frames from input.

        Returns:
            Dictionary with:
                - ball_uv_pred: Raw model predictions (B, T, 2)
                - ball_uv_completed: Completed trajectory (B, T, 2) if merge_observed
        """
        if ball_uv_in.dim() == 2:
            ball_uv_in = ball_uv_in.unsqueeze(0)
        if ball_obs_mask.dim() == 1:
            ball_obs_mask = ball_obs_mask.unsqueeze(0)
        if court_kp.dim() == 2:
            court_kp = court_kp.unsqueeze(0)
        if court_vis is not None and court_vis.dim() == 1:
            court_vis = court_vis.unsqueeze(0)
        if seq_len is not None and seq_len.dim() == 0:
            seq_len = seq_len.unsqueeze(0)

        ball_uv_in, ball_obs_mask, court_kp, court_vis = self._to_device(
            self.device, ball_uv_in, ball_obs_mask, court_kp, court_vis
        )
        if seq_len is not None:
            seq_len = seq_len.to(self.device)

        pred = self.model(
            ball_uv_in=ball_uv_in,
            ball_vis=ball_obs_mask,
            court_kp=court_kp,
            court_vis=court_vis,
            seq_len=seq_len,
        )

        result = {"ball_uv_pred": pred.cpu()}
        if merge_observed:
            completed = pred.clone()
            mask = ball_obs_mask > 0
            completed[mask] = ball_uv_in[mask]
            result["ball_uv_completed"] = completed.cpu()

        return result


if __name__ == "__main__":
    torch.manual_seed(0)
    model = UVTrajectoryCompletionModel(hidden_dim=32, num_layers=2, num_heads=4, max_seq_len=16)
    predictor = UVTrajectoryCompletionPredictor(model=model, device=torch.device("cpu"))
    ball_uv_in = torch.rand(1, 16, 2)
    ball_obs_mask = torch.randint(0, 2, (1, 16)).float()
    court_kp = torch.rand(1, 20, 2)
    court_vis = torch.ones(1, 20)
    seq_len = torch.tensor([16])
    out = predictor.predict(
        ball_uv_in,
        ball_obs_mask,
        court_kp,
        court_vis=court_vis,
        seq_len=seq_len,
        merge_observed=True,
    )
    assert out["ball_uv_pred"].shape == (1, 16, 2)
    assert out["ball_uv_completed"].shape == (1, 16, 2)
    print("trajectory_completion.uv_predictor smoke ok")

"""Evaluation metrics for PLCS keypoint-3D models."""

from __future__ import annotations

import torch
from torch import Tensor


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


class PLCSKeypoint3DMetrics:
    """Compute and track keypoint-level 3D errors."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._mpjpe_values: list[Tensor] = []
        self._pelvis_values: list[Tensor] = []

    def update(
        self,
        pred_player_kp_3d: Tensor,
        target_player_kp_3d: Tensor,
        *,
        human_vis: Tensor | None = None,
    ) -> dict[str, float]:
        """Update running metrics.

        Args:
            pred_player_kp_3d: Predicted 3D keypoints, shape (B, J, 3).
            target_player_kp_3d: Target 3D keypoints, shape (B, J, 3).
            human_vis: Optional visibility mask, shape (B, J).

        Returns:
            dict: Batch-level metrics.

        """
        per_joint_error = (pred_player_kp_3d - target_player_kp_3d).norm(dim=-1)
        mpjpe = _masked_mean(per_joint_error, human_vis)
        pelvis_error = (pred_player_kp_3d[:, 0, :] - target_player_kp_3d[:, 0, :]).norm(dim=-1).mean()

        self._mpjpe_values.append(mpjpe.detach().cpu().reshape(1))
        self._pelvis_values.append(pelvis_error.detach().cpu().reshape(1))

        return {
            "mpjpe_m": mpjpe.item(),
            "pelvis_error_m": pelvis_error.item(),
        }

    def compute(self) -> dict[str, float]:
        """Compute aggregated metrics."""
        if not self._mpjpe_values:
            return {
                "mpjpe_m": 0.0,
                "pelvis_error_m": 0.0,
            }
        mpjpe = torch.cat(self._mpjpe_values).mean().item()
        pelvis = torch.cat(self._pelvis_values).mean().item()
        return {
            "mpjpe_m": mpjpe,
            "pelvis_error_m": pelvis,
        }

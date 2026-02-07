"""Loss functions for PLCS keypoint-3D training."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


def _masked_joint_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    """Mean over joints with optional visibility mask.

    Args:
        values: Tensor of shape (B, J) or (B, T, J).
        mask: Visibility mask with matching leading dims (B, J) or (B, T, J).

    """
    if mask is None:
        return values.mean()
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


class PLCSKeypoint3DLoss(nn.Module):
    """Smooth-L1 loss for per-keypoint 3D regression."""

    def __init__(self, keypoint_weight: float = 1.0) -> None:
        super().__init__()
        self.keypoint_weight = float(keypoint_weight)

    def forward(
        self,
        pred_player_kp_3d: Tensor,
        target_player_kp_3d: Tensor,
        *,
        human_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute masked keypoint loss.

        Args:
            pred_player_kp_3d: Predicted 3D keypoints, shape (B, J, 3) or (B, T, J, 3).
            target_player_kp_3d: Target 3D keypoints, same shape as prediction.
            human_vis: Optional visibility mask, shape (B, J) or (B, T, J).

        Returns:
            dict: Dictionary with `total` and `keypoint_3d`.

        """
        per_elem = nn.functional.smooth_l1_loss(
            pred_player_kp_3d,
            target_player_kp_3d,
            reduction="none",
        )
        per_joint = per_elem.mean(dim=-1)
        kp3d_loss = _masked_joint_mean(per_joint, human_vis)
        total = self.keypoint_weight * kp3d_loss
        return {
            "total": total,
            "keypoint_3d": kp3d_loss,
        }

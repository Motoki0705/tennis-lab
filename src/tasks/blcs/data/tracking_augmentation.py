"""Pre-query corruption boundary for physical-width BLCS detections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor

from src.tasks.base.data.observation_tracking import (
    limit_synthetic_false_positive_carriers,
)
from src.tasks.blcs.data.augmentation import BLCSBallObservationAugmentation
from src.tasks.blcs.data.observation_candidates import PhysicalObservationCandidates
from src.tasks.blcs.data.types import BLCSMultiViewSample


class BLCSTrackingDetectionAugmentation:
    """Corrupt ``P`` detection carriers before camera-local Q association."""

    def __init__(self, config: Mapping[str, Any], *, num_slots: int) -> None:
        if type(num_slots) is not int:
            raise TypeError(f"num_slots must be int, got {type(num_slots).__name__}.")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive.")
        self.config = config
        self.num_slots = num_slots
        self.observation = BLCSBallObservationAugmentation(self.config)

    def forward(
        self,
        detections: PhysicalObservationCandidates,
        *,
        court_kp: Tensor,
        court_vis: Tensor,
    ) -> PhysicalObservationCandidates:
        """Return noisy physical carriers with post-corruption provenance."""
        views, frames, carriers, _ = detections.uv.shape
        adapted = {
            "ball_uv": detections.uv.permute(0, 2, 1, 3).reshape(
                views * carriers, frames, 2
            ),
            "ball_vis": detections.vis.permute(0, 2, 1).reshape(
                views * carriers, frames
            ),
            "court_kp": court_kp,
            "court_vis": court_vis,
        }
        augmentation_result = self.observation.forward_with_tracking_provenance(
            cast("BLCSMultiViewSample", adapted)
        )
        augmented = augmentation_result.sample
        noisy_uv = (
            augmented["ball_uv"].reshape(views, carriers, frames, 2).permute(0, 2, 1, 3)
        )
        noisy_vis = (
            augmented["ball_vis"]
            .reshape(views, carriers, frames)
            .permute(0, 2, 1)
            .bool()
        )
        visibility_before_false_positive = (
            augmentation_result.visibility_before_false_positive
        )
        visibility_before_false_positive = (
            visibility_before_false_positive.reshape(views, carriers, frames)
            .permute(0, 2, 1)
            .bool()
        )
        limited_uv, limited_vis = limit_synthetic_false_positive_carriers(
            noisy_uv.unsqueeze(-2),
            noisy_vis.unsqueeze(-1),
            visibility_before_false_positive.unsqueeze(-1),
            num_slots=self.num_slots,
        )
        noisy_uv = limited_uv[..., 0, :]
        noisy_vis = limited_vis[..., 0]
        genuine_observation = noisy_vis & visibility_before_false_positive
        noisy_gt_index = torch.where(
            genuine_observation,
            detections.gt_index,
            torch.full_like(detections.gt_index, -1),
        )
        return PhysicalObservationCandidates(
            uv=torch.where(
                noisy_vis.unsqueeze(-1), noisy_uv, torch.zeros_like(noisy_uv)
            ),
            vis=noisy_vis,
            gt_index=noisy_gt_index,
        )

    def __call__(
        self,
        detections: PhysicalObservationCandidates,
        *,
        court_kp: Tensor,
        court_vis: Tensor,
    ) -> PhysicalObservationCandidates:
        """Delegate callable use to :meth:`forward` with a typed contract."""
        return self.forward(
            detections,
            court_kp=court_kp,
            court_vis=court_vis,
        )


__all__ = ["BLCSTrackingDetectionAugmentation"]

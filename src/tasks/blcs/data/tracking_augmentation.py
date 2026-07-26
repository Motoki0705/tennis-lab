"""Shape adapter for BLCS ID-ordered candidate observation augmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor

from src.tasks.blcs.data.augmentation import BLCSBallObservationAugmentation
from src.utils.tensor_utils import clone_tensor_dict


class BLCSTrackingCandidateAugmentation:
    """Apply single-ball corruption along every candidate's time axis."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = config or {}
        self.observation = BLCSBallObservationAugmentation(self.config)

    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Corrupt only candidate/court inputs and preserve clean GT tensors."""
        output = clone_tensor_dict(sample)
        views, frames, detections, _ = output["ball_uv"].shape
        clean_visible = output["ball_visible"].clone()
        court_keypoints = output["court_kp"].clone()
        court_visible = output["court_vis"].clone()
        adapted = {
            "ball_uv": output["ball_uv"]
            .permute(0, 2, 1, 3)
            .reshape(views * detections, frames, 2),
            "ball_vis": output["ball_visible"]
            .permute(0, 2, 1)
            .reshape(views * detections, frames),
            "court_kp": output["court_kp"],
            "court_vis": output["court_vis"],
        }
        augmented = self.observation.forward(adapted)
        output["ball_uv"] = (
            augmented["ball_uv"]
            .reshape(views, detections, frames, 2)
            .permute(0, 2, 1, 3)
        )
        output["ball_visible"] = (
            augmented["ball_vis"]
            .reshape(views, detections, frames)
            .permute(0, 2, 1)
            .bool()
        )
        # Court input is geometric projection/manual annotation, not a detector
        # confidence stream.
        output["court_kp"] = court_keypoints
        output["court_vis"] = court_visible
        output["candidate_gt_index"] = torch.where(
            output["ball_visible"] & clean_visible,
            output["candidate_gt_index"],
            -1,
        )
        return cast(dict[str, Tensor], output)

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Delegate callable use to :meth:`forward` with a typed contract."""
        return self.forward(sample)


__all__ = ["BLCSTrackingCandidateAugmentation"]

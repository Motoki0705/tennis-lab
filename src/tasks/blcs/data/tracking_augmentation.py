"""Shape adapter for BLCS ID-ordered candidate observation augmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor

from src.tasks.blcs.data.augmentation import BLCSBallObservationAugmentation
from src.tasks.blcs.data.types import BLCSMultiViewSample
from src.utils.tensor_utils import clone_tensor_dict


class BLCSTrackingCandidateAugmentation:
    """Apply single-ball corruption along every candidate's time axis."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.observation = BLCSBallObservationAugmentation(self.config)

    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Corrupt only candidate/court inputs and preserve clean GT tensors."""
        output: dict[str, Tensor] = clone_tensor_dict(sample)
        views, frames, detections, _ = output["ball_uv"].shape
        clean_visible = output["ball_visible"].clone()
        kp7 = "court_peak_uv" in output
        if kp7:
            court_shape = output["court_peak_uv"].shape
            court_keypoints = output["court_peak_uv"].flatten(2, 3).clone()
            court_visible = output["court_peak_valid"].flatten(2, 3).clone()
        else:
            court_keypoints = output["court_kp"].clone()
            court_visible = output["court_vis"].clone()
        adapted = {
            "ball_uv": output["ball_uv"]
            .permute(0, 2, 1, 3)
            .reshape(views * detections, frames, 2),
            "ball_vis": output["ball_visible"]
            .permute(0, 2, 1)
            .reshape(views * detections, frames),
            "court_kp": court_keypoints,
            "court_vis": court_visible,
        }
        augmented = self.observation.forward(cast("BLCSMultiViewSample", adapted))
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
        if kp7:
            output["court_peak_uv"] = court_keypoints.reshape(court_shape)
            output["court_peak_valid"] = court_visible.reshape(
                court_shape[:-1]
            ).bool()
            output["court_peak_score"] = output["court_peak_valid"].to(
                output["court_peak_score"].dtype
            )
        else:
            output["court_kp"] = court_keypoints
            output["court_vis"] = court_visible
        output["candidate_gt_index"] = torch.where(
            output["ball_visible"] & clean_visible,
            output["candidate_gt_index"],
            -1,
        )
        if "ball_score" in output:
            output["ball_score"] = output["ball_visible"].to(
                output["ball_score"].dtype
            )
        return output

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Delegate callable use to :meth:`forward` with a typed contract."""
        return self.forward(sample)


__all__ = ["BLCSTrackingCandidateAugmentation"]

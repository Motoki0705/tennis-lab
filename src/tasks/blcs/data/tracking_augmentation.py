"""Shape adapter for fixed-width BLCS candidate observation augmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

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
        court_keypoints = output["court_kp"].clone()
        court_vis = output["court_vis"].clone()
        adapted = {
            "ball_uv": output["ball_uv"]
            .permute(0, 2, 1, 3)
            .reshape(views * detections, frames, 2),
            "ball_vis": output["ball_vis"]
            .permute(0, 2, 1)
            .reshape(views * detections, frames),
            "court_kp": output["court_kp"],
            "court_vis": output["court_vis"],
        }
        augmented = self.observation.forward(cast("BLCSMultiViewSample", adapted))
        output["ball_uv"] = (
            augmented["ball_uv"]
            .reshape(views, detections, frames, 2)
            .permute(0, 2, 1, 3)
        )
        output["ball_vis"] = (
            augmented["ball_vis"]
            .reshape(views, detections, frames)
            .permute(0, 2, 1)
            .bool()
        )
        # Court input is geometric projection/manual annotation, not a detector
        # confidence stream.
        output["court_kp"] = court_keypoints
        output["court_vis"] = court_vis
        return output

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Delegate callable use to :meth:`forward` with a typed contract."""
        return self.forward(sample)


__all__ = ["BLCSTrackingCandidateAugmentation"]

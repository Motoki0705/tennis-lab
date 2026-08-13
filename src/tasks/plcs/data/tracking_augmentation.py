"""Shape adapter for PLCS ID-ordered detection observation augmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from src.tasks.plcs.data.augmentation import PLCSObservationAugmentation
from src.utils.tensor_utils import clone_tensor_dict


class PLCSTrackingDetectionAugmentation:
    """Apply single-person corruption after flattening detection/joint axes."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.observation = PLCSObservationAugmentation(self.config)

    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Corrupt only detection/court inputs and preserve clean GT tensors."""
        output: dict[str, Tensor] = clone_tensor_dict(sample)
        views, frames, detections, joints, _ = output["human_kp"].shape
        clean_detection = output["detection_mask"].clone()
        kp7 = "court_peak_uv" in output
        if kp7:
            court_shape = output["court_peak_uv"].shape
            court_keypoints = output["court_peak_uv"].flatten(2, 3).clone()
            court_visible = output["court_peak_valid"].flatten(2, 3).clone()
        else:
            court_keypoints = output["court_kp"].clone()
            court_visible = output["court_vis"].clone()
        adapted = {
            "human_kp": output["human_kp"].reshape(
                views, frames, detections * joints, 2
            ),
            "human_vis": output["human_vis"].reshape(
                views, frames, detections * joints
            ),
            "court_kp": court_keypoints,
            "court_vis": court_visible,
        }
        augmented = self.observation.forward(adapted)
        output["human_kp"] = augmented["human_kp"].reshape(
            views, frames, detections, joints, 2
        )
        output["human_vis"] = (
            augmented["human_vis"].reshape(views, frames, detections, joints).bool()
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
        output["detection_mask"] = output["human_vis"].any(-1)
        output["joint_visibility"] = output["human_vis"]
        output["detection_score"] = output["human_vis"].float().mean(-1)
        output["detection_gt_index"] = torch.where(
            output["detection_mask"] & clean_detection,
            output["detection_gt_index"],
            -1,
        )
        return output

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Delegate callable use to :meth:`forward` with a typed contract."""
        return self.forward(sample)


__all__ = ["PLCSTrackingDetectionAugmentation"]

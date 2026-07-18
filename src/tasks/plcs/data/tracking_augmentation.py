"""Shape adapter for PLCS unordered-detection tracking augmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor

from src.tasks.base.data.tracking_augmentation import permute_court_keypoint_sets
from src.tasks.plcs.data.augmentation import PLCSObservationAugmentation
from src.utils.tensor_utils import clone_tensor_dict


class PLCSTrackingDetectionAugmentation:
    """Apply single-person corruption after flattening detection/joint axes."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = config or {}
        self.observation = PLCSObservationAugmentation(self.config)
        self.court_permutation_cfg = self.config.get("court_keypoint_permutation", {})

    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Corrupt only detection/court inputs and preserve clean GT tensors."""
        output = clone_tensor_dict(sample)
        views, frames, detections, joints, _ = output["human_kp"].shape
        clean_detection = output["detection_mask"].clone()
        court_keypoints = output["court_kp"].clone()
        court_visible = output["court_vis"].clone()
        adapted = {
            "human_kp": output["human_kp"].reshape(
                views, frames, detections * joints, 2
            ),
            "human_vis": output["human_vis"].reshape(
                views, frames, detections * joints
            ),
            "court_kp": output["court_kp"],
            "court_vis": output["court_vis"],
        }
        augmented = self.observation.forward(adapted)
        output["human_kp"] = augmented["human_kp"].reshape(
            views, frames, detections, joints, 2
        )
        output["human_vis"] = (
            augmented["human_vis"].reshape(views, frames, detections, joints).bool()
        )
        # Court input is geometric projection/manual annotation, not a detector
        # confidence stream. Keep it intact; only the permutation below applies.
        output["court_kp"] = court_keypoints
        output["court_vis"] = court_visible
        output["detection_mask"] = output["human_vis"].any(-1)
        output["detection_gt_index"] = torch.where(
            output["detection_mask"] & clean_detection,
            output["detection_gt_index"],
            -1,
        )
        output["court_kp"], output["court_vis"] = permute_court_keypoint_sets(
            output["court_kp"],
            output["court_vis"],
            self.court_permutation_cfg,
        )
        self._shuffle_detections(output)
        return cast(dict[str, Tensor], output)

    @staticmethod
    def _shuffle_detections(sample: dict[str, Tensor]) -> None:
        detections = int(sample["human_kp"].shape[2])
        keys = (
            "human_kp",
            "human_vis",
            "detection_mask",
            "detection_gt_index",
        )
        for view_index in range(sample["human_kp"].shape[0]):
            for frame_index in range(sample["human_kp"].shape[1]):
                permutation = torch.randperm(
                    detections, device=sample["human_kp"].device
                )
                for key in keys:
                    sample[key][view_index, frame_index] = sample[key][
                        view_index, frame_index, permutation
                    ]

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Delegate callable use to :meth:`forward` with a typed contract."""
        return self.forward(sample)


__all__ = ["PLCSTrackingDetectionAugmentation"]

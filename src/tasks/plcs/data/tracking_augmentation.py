"""Pre-tracking corruption boundary for physical-width PLCS detections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from src.tasks.base.data import limit_synthetic_false_positive_carriers
from src.tasks.plcs.data.augmentation import PLCSObservationAugmentation
from src.utils.tensor_utils import clone_tensor_dict


class PLCSTrackingDetectionAugmentation:
    """Corrupt physical detection carriers before fixed-Q association.

    The carrier axis is deliberately not a query axis.  Scalar provenance is
    retained only when at least one clean joint survives the corruption; an
    observation made exclusively from injected false-positive joints receives
    provenance ``-1``.
    """

    def __init__(self, config: Mapping[str, Any], *, num_slots: int) -> None:
        if type(num_slots) is not int:
            raise TypeError(
                f"num_slots must be int, got {type(num_slots).__name__}."
            )
        if num_slots <= 0:
            raise ValueError("num_slots must be positive.")
        self.config = config
        self.num_slots = num_slots
        self.observation = PLCSObservationAugmentation(self.config)

    @staticmethod
    def _validate_physical_sample(sample: dict[str, Tensor]) -> None:
        human_keypoints = sample["human_kp"]
        if human_keypoints.ndim != 5 or human_keypoints.shape[-1] != 2:
            raise ValueError("human_kp must have shape (V,T,D,17,2).")
        views, frames, detections, joints, _ = human_keypoints.shape
        if joints != 17:
            raise ValueError(
                "PLCS tracking corruption requires exactly 17 COCO keypoints."
            )
        if sample["human_vis"].shape != (views, frames, detections, joints):
            raise ValueError("human_vis must match human_kp without the UV axis.")
        if sample["human_vis"].dtype != torch.bool:
            raise TypeError("human_vis must have dtype torch.bool.")
        if sample["detection_gt_index"].shape != (views, frames, detections):
            raise ValueError(
                "detection_gt_index must match the physical detection carrier axis."
            )
        if sample["detection_gt_index"].dtype != torch.long:
            raise TypeError("detection_gt_index must have dtype torch.long.")

    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Corrupt a prevalidated physical-width ``(V,T,D,17,2)`` detection set."""
        output: dict[str, Tensor] = clone_tensor_dict(sample)
        views, frames, detections, joints, _ = output["human_kp"].shape
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
        augmentation_result = self.observation.forward_with_tracking_provenance(adapted)
        augmented = augmentation_result.sample
        augmented_human_kp = augmented["human_kp"].reshape(
            views, frames, detections, joints, 2
        )
        augmented_human_vis = augmented["human_vis"].reshape(
            views, frames, detections, joints
        ).bool()
        visibility_before_false_positive = (
            augmentation_result.human_visibility_before_false_positive.reshape(
                views, frames, detections, joints
            ).bool()
        )
        output["human_kp"], output["human_vis"] = (
            limit_synthetic_false_positive_carriers(
                augmented_human_kp,
                augmented_human_vis,
                visibility_before_false_positive,
                num_slots=self.num_slots,
            )
        )
        # Court input is geometric projection/manual annotation, not a detector
        # confidence stream.
        output["court_kp"] = court_keypoints
        output["court_vis"] = court_visible
        genuine_joint_survives = visibility_before_false_positive.any(-1)
        output["detection_gt_index"] = torch.where(
            output["human_vis"].any(-1) & genuine_joint_survives,
            output["detection_gt_index"],
            -1,
        )
        return output

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Validate the public sample contract, then delegate to :meth:`forward`."""
        self._validate_physical_sample(sample)
        return self.forward(sample)


__all__ = ["PLCSTrackingDetectionAugmentation"]

from __future__ import annotations

import torch

from src.tasks.plcs.data.tracking_augmentation import (
    PLCSTrackingDetectionAugmentation,
)


def test_tracking_augmentation_keeps_clean_labels_and_detection_pairs() -> None:
    torch.manual_seed(22)
    human_kp = torch.rand(1, 2, 2, 17, 2)
    sample = {
        "human_kp": human_kp.clone(),
        "human_vis": torch.ones(1, 2, 2, 17, dtype=torch.bool),
        "detection_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "detection_gt_index": torch.tensor([[[0, 1], [0, 1]]]),
        "court_kp": torch.rand(1, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 14, dtype=torch.bool),
        "target_position": torch.rand(2, 2, 3),
        "clean_human_kp": human_kp.clone(),
    }
    augmentation = PLCSTrackingDetectionAugmentation(
        {
            "enabled": False,
            "court_keypoint_permutation": {"enabled": True, "prob": 1.0},
        }
    )

    result = augmentation(sample)

    torch.testing.assert_close(result["target_position"], sample["target_position"])
    torch.testing.assert_close(result["clean_human_kp"], sample["clean_human_kp"])
    torch.testing.assert_close(
        result["court_kp"][..., 0].sort(dim=-1).values,
        sample["court_kp"][..., 0].sort(dim=-1).values,
    )
    for frame_index in range(2):
        for detection_index in range(2):
            source = int(result["detection_gt_index"][0, frame_index, detection_index])
            torch.testing.assert_close(
                result["human_kp"][0, frame_index, detection_index],
                sample["human_kp"][0, frame_index, source],
            )

from __future__ import annotations

import torch

from src.tasks.plcs.data.tracking_augmentation import (
    PLCSTrackingDetectionAugmentation,
)


def test_tracking_augmentation_preserves_id_order_without_permutation() -> None:
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
        }
    )

    result = augmentation(sample)

    torch.testing.assert_close(result["target_position"], sample["target_position"])
    torch.testing.assert_close(result["clean_human_kp"], sample["clean_human_kp"])
    torch.testing.assert_close(result["court_kp"], sample["court_kp"])
    torch.testing.assert_close(result["court_vis"], sample["court_vis"])
    torch.testing.assert_close(
        result["detection_gt_index"], sample["detection_gt_index"]
    )
    torch.testing.assert_close(result["human_kp"], sample["human_kp"])
    torch.testing.assert_close(result["human_vis"], sample["human_vis"])


def test_tracking_noise_changes_keypoints_without_reordering_object_ids() -> None:
    torch.manual_seed(24)
    human_kp = torch.empty(1, 2, 2, 17, 2)
    human_kp[:, :, 0] = 0.2
    human_kp[:, :, 1] = 0.8
    sample = {
        "human_kp": human_kp.clone(),
        "human_vis": torch.ones(1, 2, 2, 17, dtype=torch.bool),
        "detection_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "detection_gt_index": torch.tensor([[[0, 1], [0, 1]]]),
        "court_kp": torch.rand(1, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 14, dtype=torch.bool),
    }
    augmentation = PLCSTrackingDetectionAugmentation(
        {
            "enabled": True,
            "gaussian_noise": {
                "enabled": True,
                "prob": 1.0,
                "human_std": 0.001,
                "court_std": 0.001,
            },
            "uv_scale": {"enabled": False},
            "visibility_dropout": {"enabled": False},
            "temporal_jitter": {"enabled": False},
            "burst_dropout": {"enabled": False},
            "false_positive": {"enabled": False},
            "edge_degradation": {"enabled": False},
            "speed_conditioned": {"enabled": False},
        }
    )

    result = augmentation(sample)

    assert not torch.equal(result["human_kp"], sample["human_kp"])
    assert bool(((result["human_kp"] - sample["human_kp"]).abs() < 0.01).all())
    torch.testing.assert_close(
        result["detection_gt_index"], sample["detection_gt_index"]
    )
    torch.testing.assert_close(result["court_kp"], sample["court_kp"])
    torch.testing.assert_close(result["court_vis"], sample["court_vis"])

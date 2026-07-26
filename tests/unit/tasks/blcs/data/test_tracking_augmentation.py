from __future__ import annotations

import torch

from src.tasks.blcs.data.tracking_augmentation import (
    BLCSTrackingCandidateAugmentation,
)


def test_tracking_augmentation_preserves_id_order_without_permutation() -> None:
    torch.manual_seed(21)
    uv = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]])
    sample = {
        "ball_uv": uv.clone(),
        "ball_visible": torch.ones(1, 2, 2, dtype=torch.bool),
        "candidate_gt_index": torch.tensor([[[0, 1], [0, 1]]]),
        "court_kp": torch.rand(1, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 14, dtype=torch.bool),
        "target_position": torch.rand(2, 2, 3),
        "clean_ball_uv": uv.clone(),
    }
    augmentation = BLCSTrackingCandidateAugmentation(
        {
            "enabled": False,
        }
    )

    result = augmentation(sample)

    torch.testing.assert_close(result["target_position"], sample["target_position"])
    torch.testing.assert_close(result["clean_ball_uv"], sample["clean_ball_uv"])
    torch.testing.assert_close(result["court_kp"], sample["court_kp"])
    torch.testing.assert_close(result["court_vis"], sample["court_vis"])
    torch.testing.assert_close(
        result["candidate_gt_index"], sample["candidate_gt_index"]
    )
    torch.testing.assert_close(result["ball_uv"], sample["ball_uv"])
    torch.testing.assert_close(result["ball_visible"], sample["ball_visible"])


def test_tracking_noise_changes_coordinates_without_reordering_object_ids() -> None:
    torch.manual_seed(23)
    uv = torch.tensor(
        [[[[0.2, 0.2], [0.8, 0.8]], [[0.25, 0.25], [0.75, 0.75]]]]
    )
    sample = {
        "ball_uv": uv.clone(),
        "ball_visible": torch.ones(1, 2, 2, dtype=torch.bool),
        "candidate_gt_index": torch.tensor([[[0, 1], [0, 1]]]),
        "court_kp": torch.rand(1, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 14, dtype=torch.bool),
    }
    augmentation = BLCSTrackingCandidateAugmentation(
        {
            "enabled": True,
            "gaussian_noise": {
                "enabled": True,
                "prob": 1.0,
                "ball_std": 0.001,
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

    assert not torch.equal(result["ball_uv"], sample["ball_uv"])
    assert bool(((result["ball_uv"] - sample["ball_uv"]).abs() < 0.01).all())
    torch.testing.assert_close(
        result["candidate_gt_index"], sample["candidate_gt_index"]
    )
    torch.testing.assert_close(result["court_kp"], sample["court_kp"])
    torch.testing.assert_close(result["court_vis"], sample["court_vis"])

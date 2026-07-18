from __future__ import annotations

import torch

from src.tasks.blcs.data.tracking_augmentation import (
    BLCSTrackingCandidateAugmentation,
)


def test_tracking_augmentation_keeps_clean_labels_and_candidate_pairs() -> None:
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
            "court_keypoint_permutation": {"enabled": True, "prob": 1.0},
        }
    )

    result = augmentation(sample)

    torch.testing.assert_close(result["target_position"], sample["target_position"])
    torch.testing.assert_close(result["clean_ball_uv"], sample["clean_ball_uv"])
    torch.testing.assert_close(
        result["court_kp"][..., 0].sort(dim=-1).values,
        sample["court_kp"][..., 0].sort(dim=-1).values,
    )
    for frame_index in range(2):
        for candidate_index in range(2):
            source = int(result["candidate_gt_index"][0, frame_index, candidate_index])
            torch.testing.assert_close(
                result["ball_uv"][0, frame_index, candidate_index],
                sample["ball_uv"][0, frame_index, source],
            )

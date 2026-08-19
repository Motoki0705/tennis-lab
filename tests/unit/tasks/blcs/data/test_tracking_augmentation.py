from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.blcs.data.tracking_augmentation import (
    BLCSTrackingCandidateAugmentation,
)

_AUGMENTATION_CONFIG = (
    Path(__file__).resolve().parents[5]
    / "src/tasks/blcs/configs/data/_augmentation.yaml"
)


def _augmentation_config(*, enabled: bool, gaussian_noise: bool = False) -> DictConfig:
    config = OmegaConf.load(_AUGMENTATION_CONFIG).augmentation
    if not isinstance(config, DictConfig):
        raise AssertionError("BLCS augmentation config must be a mapping.")
    config.enabled = enabled
    for block_name in (
        "uv_scale",
        "gaussian_noise",
        "visibility_dropout",
        "temporal_jitter",
        "burst_dropout",
        "false_positive",
        "edge_degradation",
        "speed_conditioned",
    ):
        config[block_name].enabled = False
    config.gaussian_noise.enabled = gaussian_noise
    config.gaussian_noise.prob = 1.0
    config.gaussian_noise.ball_std = 0.001
    config.gaussian_noise.court_std = 0.001
    return config


def test_tracking_augmentation_preserves_id_order_without_permutation() -> None:
    torch.manual_seed(21)
    uv = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]])
    sample = {
        "ball_uv": uv.clone(),
        "ball_visible": torch.ones(1, 2, 2, dtype=torch.bool),
        "candidate_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "candidate_gt_index": torch.tensor([[[0, 1], [0, 1]]]),
        "court_kp": torch.rand(1, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 14, dtype=torch.bool),
        "target_position": torch.rand(2, 2, 3),
        "clean_ball_uv": uv.clone(),
    }
    augmentation = BLCSTrackingCandidateAugmentation(
        _augmentation_config(enabled=False)
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
        "candidate_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "candidate_gt_index": torch.tensor([[[0, 1], [0, 1]]]),
        "court_kp": torch.rand(1, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 14, dtype=torch.bool),
    }
    augmentation = BLCSTrackingCandidateAugmentation(
        _augmentation_config(enabled=True, gaussian_noise=True)
    )

    result = augmentation(sample)

    assert not torch.equal(result["ball_uv"], sample["ball_uv"])
    assert bool(((result["ball_uv"] - sample["ball_uv"]).abs() < 0.01).all())
    torch.testing.assert_close(
        result["candidate_gt_index"], sample["candidate_gt_index"]
    )
    torch.testing.assert_close(result["court_kp"], sample["court_kp"])
    torch.testing.assert_close(result["court_vis"], sample["court_vis"])


def test_tracking_augmentation_cannot_activate_padding_candidates() -> None:
    uv = torch.rand(1, 2, 2, 2)
    sample = {
        "ball_uv": uv,
        "ball_visible": torch.tensor([[[True, False], [True, False]]]),
        "candidate_mask": torch.tensor([[[True, False], [True, False]]]),
        "candidate_gt_index": torch.tensor([[[3, -1], [3, -1]]]),
        "court_kp": torch.rand(1, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 14, dtype=torch.bool),
    }
    augmentation = BLCSTrackingCandidateAugmentation(
        _augmentation_config(enabled=True)
    )

    result = augmentation(sample)

    assert not result["ball_visible"][..., 1].any()
    assert not result["ball_uv"][..., 1, :].any()
    torch.testing.assert_close(result["candidate_mask"], sample["candidate_mask"])

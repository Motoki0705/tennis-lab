from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.plcs.data.tracking_augmentation import (
    PLCSTrackingDetectionAugmentation,
)

_AUGMENTATION_CONFIG = (
    Path(__file__).resolve().parents[5]
    / "src/tasks/plcs/configs/data/_augmentation.yaml"
)


def _augmentation_config(*, enabled: bool, gaussian_noise: bool = False) -> DictConfig:
    config = OmegaConf.load(_AUGMENTATION_CONFIG).augmentation
    if not isinstance(config, DictConfig):
        raise AssertionError("PLCS augmentation config must be a mapping.")
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
    config.gaussian_noise.human_std = 0.001
    config.gaussian_noise.court_std = 0.001
    return config


def test_tracking_augmentation_preserves_id_order_without_permutation() -> None:
    torch.manual_seed(22)
    human_kp = torch.rand(1, 2, 2, 17, 2)
    sample = {
        "human_kp": human_kp.clone(),
        "human_vis": torch.ones(1, 2, 2, 17, dtype=torch.bool),
        "detection_gt_index": torch.tensor([[[0, 1], [0, 1]]]),
        "court_kp": torch.rand(1, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 14, dtype=torch.bool),
        "target_position": torch.rand(2, 2, 3),
        "clean_human_kp": human_kp.clone(),
    }
    augmentation = PLCSTrackingDetectionAugmentation(
        _augmentation_config(enabled=False)
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
        "detection_gt_index": torch.tensor([[[0, 1], [0, 1]]]),
        "court_kp": torch.rand(1, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 14, dtype=torch.bool),
    }
    augmentation = PLCSTrackingDetectionAugmentation(
        _augmentation_config(enabled=True, gaussian_noise=True)
    )

    result = augmentation(sample)

    assert not torch.equal(result["human_kp"], sample["human_kp"])
    assert bool(((result["human_kp"] - sample["human_kp"]).abs() < 0.01).all())
    torch.testing.assert_close(
        result["detection_gt_index"], sample["detection_gt_index"]
    )
    torch.testing.assert_close(result["court_kp"], sample["court_kp"])
    torch.testing.assert_close(result["court_vis"], sample["court_vis"])

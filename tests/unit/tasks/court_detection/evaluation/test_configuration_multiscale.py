"""Pose-safe multi-scale configuration regression tests.

``CourtProcessingGeometry(require_pose=True)`` already samples an isotropic
long-side scale from ``train_scales`` for every training sample, so the typed
configuration must accept multi-scale pose-safe schedules instead of rejecting
them while the geometry silently supports them.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.utils.configuration import (
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


def _pose_config(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[
                "data/source=synthetic_court",
                "data/processing=all",
                "data/augmentation=pose_safe",
                "model/encoder=dinov3",
                "model/transformer_encoder=default",
                "model/decoder=dpt",
                "training=pose_frozen",
                "loss=pose",
                *overrides,
            ],
        )


def test_pose_safe_configuration_accepts_a_multi_scale_long_side_schedule() -> None:
    config = _pose_config(
        "data.augmentation.train_scales=[256,320,384,448,512]",
        "data.augmentation.val_short_side=512",
    )

    runtime = CourtTrainingConfig.from_config(config)

    assert runtime.data.augmentation.train_scales == (256, 320, 384, 448, 512)
    assert runtime.data.augmentation.val_short_side == 512
    assert runtime.loss.pose.enabled


def test_the_single_scale_pose_safe_configuration_still_loads() -> None:
    config = _pose_config()

    runtime = CourtTrainingConfig.from_config(config)

    assert runtime.data.augmentation.train_scales == (256,)


@pytest.mark.parametrize(
    "override",
    [
        "data.augmentation.hflip_prob=0.1",
        "data.augmentation.crop_scale=[0.8,1.0]",
        "data.augmentation.crop_ratio=[0.75,1.333]",
        "data.augmentation.affine_degrees=5.0",
        "data.augmentation.affine_shear=1.0",
        "data.augmentation.perspective_prob=0.1",
        "data.augmentation.canvas_size=512",
        "data.augmentation.preserve_fx_fy=false",
    ],
)
def test_multi_scale_acceptance_keeps_every_other_pose_guard(override: str) -> None:
    config = _pose_config(
        "data.augmentation.train_scales=[256,320,384,448,512]",
        "data.augmentation.val_short_side=512",
        override,
    )

    with pytest.raises(SemanticConfigurationError, match="Pose|pose"):
        CourtTrainingConfig.from_config(config)


def test_a_non_positive_scale_is_still_rejected() -> None:
    config = _pose_config("data.augmentation.train_scales=[0]")

    with pytest.raises(
        SemanticConfigurationError, match="must contain positive values"
    ):
        CourtTrainingConfig.from_config(config)


def test_unknown_augmentation_keys_are_still_rejected() -> None:
    config = deepcopy(_pose_config())
    with open_dict(config.data.augmentation):
        config.data.augmentation["unexpected"] = 1

    with pytest.raises(UnknownConfigurationKeyError):
        CourtTrainingConfig.from_config(config)

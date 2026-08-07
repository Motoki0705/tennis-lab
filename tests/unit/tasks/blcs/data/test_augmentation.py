"""Tests for constructor-resolved BLCS observation augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from omegaconf import OmegaConf

from src.tasks.blcs.data.augmentation import BLCSBallObservationAugmentation
from src.tasks.blcs.data.types import BLCSMultiViewSample


def test_default_augmentation_parses_once_and_preserves_clean_targets() -> None:
    config = OmegaConf.load(Path("src/tasks/blcs/configs/data/_augmentation.yaml"))
    augmentation = BLCSBallObservationAugmentation(config.augmentation)
    ball_uv = torch.full((2, 8, 2), 0.5)
    ball_vis = torch.ones(2, 8, dtype=torch.bool)
    sample = cast(
        BLCSMultiViewSample,
        {
            "ball_uv": ball_uv,
            "ball_vis": ball_vis,
            "court_kp": torch.full((2, 8, 14, 2), 0.5),
            "court_vis": torch.ones(2, 8, 14, dtype=torch.bool),
        },
    )

    torch.manual_seed(7)
    result = augmentation.forward(sample)

    torch.testing.assert_close(result["ball_uv_target"], ball_uv)
    torch.testing.assert_close(result["ball_vis_target"], ball_vis)
    assert torch.isfinite(result["ball_uv"]).all()
    assert torch.isfinite(result["court_kp"]).all()
    assert bool(((result["ball_uv"] >= 0.0) & (result["ball_uv"] <= 1.0)).all())
    assert bool(((result["court_kp"] >= 0.0) & (result["court_kp"] <= 1.0)).all())

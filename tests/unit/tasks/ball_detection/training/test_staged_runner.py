"""Unit tests for staged training runner calibration behavior."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

import src.tasks.ball_detection.training.staged_runner as staged_runner_module
from src.tasks.ball_detection.data.staged_datamodule import StagedBallDataModule
from src.tasks.ball_detection.training.staged_runner import (
    StagedBallDetectionTrainingRunner,
)

pytestmark = pytest.mark.unit


def test_fixed_t_calibration_keeps_effective_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = OmegaConf.create(
        {
            "data": {
                "t_max": 8,
                "t_distribution": "fixed",
                "val_num_frames": 8,
                "effective_batch_size": 8,
                "batch_size_by_t": {1: 8, 2: 6, 3: 4, 4: 3, 5: 3, 6: 2, 7: 2, 8: 2},
                "sources": {
                    "tracknet": {"enabled": True, "splits": ["train", "val", "test"]},
                    "web": {"enabled": False, "splits": ["train", "val", "test"]},
                },
            },
            "training": {
                "staged": {
                    "calibration_token_budget": 24,
                    "calibration_safety": 0.9,
                }
            },
        }
    )
    datamodule = StagedBallDataModule(config)
    seen_t_values: list[int] = []

    def fake_probe_batch_size_by_t(
        _config: Any,
        t_values: list[int],
        *,
        device: torch.device,
        token_budget: int,
        safety: float,
    ) -> dict[int, int]:
        _ = device
        _ = token_budget
        _ = safety
        seen_t_values.extend(t_values)
        return {8: 5}

    monkeypatch.setattr(staged_runner_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        staged_runner_module,
        "probe_batch_size_by_t",
        fake_probe_batch_size_by_t,
    )

    StagedBallDetectionTrainingRunner()._calibrate(config, datamodule)

    assert seen_t_values == [8]
    assert datamodule.batch_size_by_t[8] == 5
    assert datamodule.effective_batch_size == 8

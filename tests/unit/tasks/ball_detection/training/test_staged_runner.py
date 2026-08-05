"""Unit tests for staged training runner calibration behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir

import src.tasks.ball_detection.training.staged_runner as staged_runner_module
from src.tasks.ball_detection.data.staged_datamodule import StagedBallDataModule
from src.tasks.ball_detection.training.staged_runner import (
    StagedBallDetectionTrainingRunner,
)

pytestmark = pytest.mark.unit

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/ball_detection/configs"


def test_fixed_t_calibration_keeps_effective_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_staged", overrides=["model=stunet"])
    config.data.t_max = 8
    config.data.t_distribution = "fixed"
    config.data.val_num_frames = 8
    config.data.effective_batch_size = 8
    config.data.batch_size_by_t = {
        1: 8,
        2: 6,
        3: 4,
        4: 3,
        5: 3,
        6: 2,
        7: 2,
        8: 2,
    }
    config.data.sources.tracknet.enabled = True
    config.data.sources.tracknet.splits = ["train", "val", "test"]
    config.data.sources.web.enabled = False
    config.data.sources.web.splits = ["train", "val", "test"]
    config.training.staged.calibration_token_budget = 24
    config.training.staged.calibration_safety = 0.9
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

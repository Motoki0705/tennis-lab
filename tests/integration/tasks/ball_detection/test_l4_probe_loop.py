"""Exercise the complete calibration loop on CPU, including the full-run warmup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import pytorch_lightning as pl

from src.tasks.ball_detection.data.multiview_datamodule import MultiviewBallDataModule
from src.tasks.ball_detection.training import l4_calibration as calibration
from tests.support.tasks.ball_detection.multiview import make_multiview_config


@pytest.mark.parametrize("accumulation", [1, 4])
def test_probe_runs_three_updates_validation_and_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accumulation: int,
) -> None:
    config = make_multiview_config(tmp_path)
    config.model.dims = [4, 8, 16, 32]
    config.model.depth = 1
    config.training.compile.enabled = False
    config.training.trainer.precision = "32-true"
    config.training.trainer.accumulate_grad_batches = accumulation
    # The small fixture has 12 windows: at accumulation=4 it needs >66 epochs
    # for the unchanged 200-update warmup to fit inside the full-run schedule.
    config.training.trainer.max_epochs = 80
    assert config.training.warmup_steps == 200
    dm = MultiviewBallDataModule(config)
    dm.prepare_data()
    monkeypatch.setattr(calibration, "require_l4", lambda: "NVIDIA L4")
    monkeypatch.setattr(calibration.torch.cuda, "is_available", lambda: False)
    for name in (
        "reset_peak_memory_stats",
        "synchronize",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(calibration.torch.cuda, name, lambda: 0)
    original_init = pl.Trainer.__init__
    trainers = []

    def cpu_init(self: pl.Trainer, **kwargs: Any) -> None:
        assert kwargs["accelerator"] == "gpu"
        kwargs["accelerator"] = "cpu"
        original_init(self, **kwargs)
        trainers.append(self)

    monkeypatch.setattr(pl.Trainer, "__init__", cpu_init)
    report = calibration.run_trial(config, batch_size=1)
    assert report["status"] == "fits"
    assert trainers[0].global_step == 3
    assert report["training_microbatches"] == 3 * accumulation
    assert report["effective_batch_size"] == accumulation
    assert trainers[0].num_training_batches == 3 * accumulation
    assert trainers[0].lightning_module.steps_per_epoch == 12 // accumulation
    assert "val/loss" in trainers[0].callback_metrics
    assert list((tmp_path / "output").rglob("qualitative/epoch_0000"))

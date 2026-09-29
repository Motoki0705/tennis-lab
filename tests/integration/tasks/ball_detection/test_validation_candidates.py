"""Validation hooks and checkpoint retention on CPU, without optimizer updates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from src.tasks.ball_detection.data.store_datamodule import BallStoreDataModule
from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)
from src.tasks.ball_detection.training.runner import BallDetectionTrainingRunner
from src.tasks.ball_detection.training.staged_lightning_module import (
    StagedBallDetectionLightningModule,
)
from tests.support.tasks.ball_detection.store import (
    ball,
    frame,
    store_config,
    write_store_clip,
)


class NativeGrid(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        values = torch.full((1, 1, 2, 11, 21), 1e-6)
        values[:, :, :, 5, 5] = 0.1
        values[:, :, :, 5, 20] = 0.9
        self.register_buffer("logits", torch.logit(values))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.logits).expand(images.shape[0], -1, -1, -1, -1)


def test_native_grid_metric_is_logged_each_epoch_and_staged_shares_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = store_config(tmp_path)
    config.model.dims = [2, 4, 8, 16]
    config.model.depth = 1
    config.data.heatmap_size = [2, 2]
    for source in ("meiji", "tracknet"):
        write_store_clip(tmp_path / "ball_detection/test-v1", f"{source}/val/clip",
                         [frame(0, ball(xy=(16, 24))), frame(1, ball(xy=(16, 24)))],
                         source=source, split="val")
    data = BallStoreDataModule(config)
    data.setup("validate")
    assert data.val_dataset is not None
    inputs = next(iter(DataLoader(data.val_dataset, batch_size=2)))
    module = BallDetectionLightningModule(config)
    module.model = NativeGrid()
    logged: dict[str, float] = {}

    def log(name: str, value: Any, **_: Any) -> None:
        logged[name] = float(value)

    monkeypatch.setattr(module, "log", log)
    for epoch in range(3):
        module.on_validation_epoch_start()
        if epoch == 2:
            cast(Any, module.model).logits.fill_(-10)
        with torch.no_grad():
            module.validation_step(inputs, 0)
        module.on_validation_epoch_end()
        expected = 0.0 if epoch == 2 else 1.0
        assert logged["val/candidate_recall_at_8_20px"] == expected
        assert logged["val/meiji/candidate_recall_at_8_20px"] == expected
        assert logged["val/candidate_observed"] == 4
        assert not module.val_candidate_metrics.frames
    assert StagedBallDetectionLightningModule.on_validation_epoch_end is BallDetectionLightningModule.on_validation_epoch_end
    assert StagedBallDetectionLightningModule._compute_supervised_result is BallDetectionLightningModule._compute_supervised_result


def test_epoch_checkpoints_survive_worsening_metric_and_callback_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TENNIS_REPRO_DIR", raising=False)
    monkeypatch.delenv("TENNIS_LAB_COLAB_PROGRESS_PATH", raising=False)
    config = store_config(tmp_path)
    config.paths.output_root = str(tmp_path / "outputs")
    config.run.output_dir = "retention"
    config.training.qualitative_logging.enabled = False
    config.training.lr_monitor.enabled = False
    runner = BallDetectionTrainingRunner()
    log_dir = tmp_path / "outputs/retention/logs/run"

    def callbacks() -> list[ModelCheckpoint]:
        return [callback for callback in runner.build_callbacks(
            config, cast(Any, None), cast(Any, SimpleNamespace(log_dir=str(log_dir))),
        ) if isinstance(callback, ModelCheckpoint)]

    all_epochs, latest = callbacks()
    assert all_epochs.save_top_k == -1
    assert all_epochs._every_n_epochs == 1
    assert latest._every_n_epochs == 1
    trainer = SimpleNamespace(
        global_step=0, is_global_zero=True, loggers=[],
        strategy=SimpleNamespace(reduce_boolean_decision=lambda value, **_: value),
    )

    def save_checkpoint(path: str, _: bool) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": trainer.global_step - 1}, path)

    trainer.save_checkpoint = save_checkpoint
    for epoch, score in enumerate([0.8, 0.3, 0.2]):
        if epoch == 2:
            state = all_epochs.state_dict()
            all_epochs, latest = callbacks()
            all_epochs.load_state_dict(state)
        trainer.global_step = epoch + 1
        metrics = {"epoch": torch.tensor(epoch), "step": torch.tensor(epoch + 1),
                   str(all_epochs.monitor): torch.tensor(score)}
        all_epochs._save_topk_checkpoint(cast(Any, trainer), metrics)
        latest._save_topk_checkpoint(cast(Any, trainer), metrics)
    directory = log_dir / "checkpoints"
    assert {path.name for path in directory.glob("*.ckpt")} == {
        "ball-detection-epoch=00.ckpt", "ball-detection-epoch=01.ckpt",
        "ball-detection-epoch=02.ckpt", "last.ckpt",
    }
    assert torch.load(directory / "last.ckpt", weights_only=True)["epoch"] == 2
    assert all_epochs.best_model_path.endswith("epoch=00.ckpt")
    assert len(all_epochs.best_k_models) == 3

"""Verify Colab observations on a real Lightning CPU training loop."""

import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.tasks.base.training.colab_progress import ColabProgressCallback


class TinyModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = (self.weight * batch[0]).square().mean()
        self.log("train/loss", loss, on_step=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_progress_tracks_actual_optimizer_steps(tmp_path: Path) -> None:
    path = tmp_path / "training-progress.json"
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[ColabProgressCallback(path)],
    )
    trainer.fit(TinyModel(), DataLoader(TensorDataset(torch.ones(3, 1)), batch_size=1))
    progress = json.loads(path.read_text())
    assert progress["global_step"] == 3
    assert progress["phase"] == "fit_finished"
    assert progress["max_epochs"] == 1
    assert 0 < progress["metrics"]["train/loss"] < 1
    assert not list(tmp_path.glob("*.tmp"))

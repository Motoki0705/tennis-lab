"""Qualitative schedules, real GIFs and checkpoint resume through Lightning on CPU."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

from src.tasks.base.training.qualitative_callback import QualitativeLoggingCallback
from src.tasks.base.training.qualitative_saving import save_qualitative_clip

pytestmark = pytest.mark.integration


class _ScheduleModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.rendered_epochs: list[int] = []

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.weight.square().sum()

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        return {"prediction": self.weight * batch[0]}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def render_qualitative_samples(
        self,
        batches: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        artifact_dir: Path,
        tb_writer: Any,
        global_step: int,
        epoch: int,
    ) -> None:
        self.rendered_epochs.append(epoch + 1)
        save_qualitative_clip(
            frames_rgb=[np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 255, np.uint8)],
            artifact_dir=artifact_dir,
            name="sample",
            tb_writer=tb_writer,
            tag="sample",
            global_step=global_step,
        )


def _fit(
    directory: Path,
    *,
    max_epochs: int,
    validate_every: int,
    save_every: int,
    ckpt_path: str | Path | None = None,
    val_check_interval: float = 1.0,
) -> tuple[pl.Trainer, _ScheduleModule, QualitativeLoggingCallback]:
    callback = QualitativeLoggingCallback(
        every_n_epochs=save_every,
        num_samples=1,
        enabled=True,
        selection_mode="fixed_indices",
        selected_indices=[0],
    )
    checkpoint = ModelCheckpoint(
        dirpath=directory / "checkpoints",
        save_top_k=0,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        check_val_every_n_epoch=validate_every,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=2,
        logger=TensorBoardLogger(str(directory), name="", version="logs"),
        callbacks=[callback, checkpoint],
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    module = _ScheduleModule()
    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)
    trainer.fit(module, loader, loader, ckpt_path=ckpt_path)
    return trainer, module, callback


@pytest.mark.parametrize(
    ("validate_every", "save_every", "max_epochs", "expected"),
    [
        pytest.param(5, 10, 200, list(range(10, 201, 10)), id="blcs-reported"),
        pytest.param(10, 20, 40, [20, 40], id="blcs-default"),
        pytest.param(10, 10, 30, [10, 20, 30], id="plcs-enabled"),
        pytest.param(3, 5, 20, [6, 12, 15], id="court-dense-enabled"),
        pytest.param(1, 5, 12, [5, 10], id="ball-detection"),
        pytest.param(5, 2, 15, [5, 10, 15], id="multiple-overdue-intervals"),
    ],
)
def test_sparse_validation_writes_gifs_at_training_epoch_thresholds(
    tmp_path: Path,
    validate_every: int,
    save_every: int,
    max_epochs: int,
    expected: list[int],
) -> None:
    _, module, _ = _fit(
        tmp_path,
        max_epochs=max_epochs,
        validate_every=validate_every,
        save_every=save_every,
    )
    assert module.rendered_epochs == expected
    artifacts = sorted((tmp_path / "logs/qualitative").glob("*/sample.gif"))
    assert [int(path.parent.name.removeprefix("epoch_")) + 1 for path in artifacts] == expected
    for path in artifacts:
        with Image.open(path) as gif:
            assert gif.n_frames == 2


@pytest.mark.parametrize("split_epoch", [6, 10])
def test_resume_preserves_schedule_and_skips_sanity(
    tmp_path: Path, split_epoch: int
) -> None:
    _, full, _ = _fit(tmp_path / "full", max_epochs=20, validate_every=3, save_every=5)
    trainer, first, callback = _fit(
        tmp_path / "resumed", max_epochs=split_epoch, validate_every=3, save_every=5
    )
    checkpoint = trainer.checkpoint_callback
    assert isinstance(checkpoint, ModelCheckpoint)
    path = checkpoint.last_model_path
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["callbacks"][callback.state_key] == callback.state_dict()
    _, resumed, _ = _fit(
        tmp_path / "resumed",
        max_epochs=20,
        validate_every=3,
        save_every=5,
        ckpt_path=path,
    )
    assert first.rendered_epochs + resumed.rendered_epochs == full.rendered_epochs == [6, 12, 15]


def test_resume_from_legacy_checkpoint_coalesces_overdue_intervals(tmp_path: Path) -> None:
    trainer, _, callback = _fit(tmp_path, max_epochs=7, validate_every=1, save_every=5)
    checkpoint = trainer.checkpoint_callback
    assert isinstance(checkpoint, ModelCheckpoint)
    path = checkpoint.last_model_path
    state = torch.load(path, map_location="cpu", weights_only=False)
    del state["callbacks"][callback.state_key]
    legacy = tmp_path / "legacy.ckpt"
    torch.save(state, legacy)
    _, resumed, _ = _fit(
        tmp_path, max_epochs=10, validate_every=1, save_every=5, ckpt_path=legacy
    )
    assert resumed.rendered_epochs == [8, 10]


def test_multiple_validations_in_one_epoch_do_not_overwrite_gifs(tmp_path: Path) -> None:
    _, module, _ = _fit(
        tmp_path, max_epochs=3, validate_every=1, save_every=1, val_check_interval=0.5
    )
    assert module.rendered_epochs == [1, 2, 3]

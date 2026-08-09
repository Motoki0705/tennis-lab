"""Unit tests for the chunk-rotation callback."""

from __future__ import annotations

from typing import cast

import pytest
import pytorch_lightning as pl

from src.tasks.base.training.chunk_rotation_callback import ChunkRotationCallback

pytestmark = pytest.mark.unit


class _DM:
    def __init__(self) -> None:
        self.calls = 0

    def on_train_epoch_end(self) -> None:
        self.calls += 1


class _Trainer:
    def __init__(
        self,
        datamodule,
        *,
        current_epoch: int = 0,
        max_epochs: int = 100,
    ) -> None:
        self.datamodule = datamodule
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs


def _run_epoch_end(trainer: _Trainer) -> None:
    ChunkRotationCallback().on_train_epoch_end(
        cast(pl.Trainer, trainer),
        pl_module=pl.LightningModule(),
    )


def test_forwards_to_datamodule_hook() -> None:
    dm = _DM()
    _run_epoch_end(_Trainer(dm))
    assert dm.calls == 1


def test_noop_when_datamodule_none() -> None:
    # Must not raise.
    _run_epoch_end(_Trainer(None))


def test_noop_when_hook_absent() -> None:
    class _Bare:
        pass

    # Datamodule without on_train_epoch_end -> no error.
    _run_epoch_end(_Trainer(_Bare()))


def test_noop_on_last_scheduled_epoch() -> None:
    dm = _DM()
    trainer = _Trainer(dm, current_epoch=99, max_epochs=100)
    _run_epoch_end(trainer)
    assert dm.calls == 0

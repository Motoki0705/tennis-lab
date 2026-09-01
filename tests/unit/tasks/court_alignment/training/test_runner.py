"""Unit tests for court-alignment post-fit checkpoint selection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from pytorch_lightning.callbacks import ModelCheckpoint

from src.tasks.court_alignment.training.runner import CourtAlignmentTrainingRunner


def _runtime(*, checkpoint_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(checkpoint=SimpleNamespace(enabled=checkpoint_enabled))
    )


def test_test_checkpoint_path_selects_nonempty_validation_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CourtAlignmentTrainingRunner()
    monkeypatch.setattr(
        runner,
        "validate_runtime_config",
        lambda _: _runtime(checkpoint_enabled=True),
    )
    checkpoint = ModelCheckpoint()
    checkpoint.best_model_path = "/tmp/best.ckpt"
    trainer = SimpleNamespace(checkpoint_callback=checkpoint)

    assert runner.test_checkpoint_path({}, cast(Any, trainer)) == "best"


@pytest.mark.parametrize(
    "trainer",
    [
        SimpleNamespace(checkpoint_callback=None),
        SimpleNamespace(checkpoint_callback=ModelCheckpoint()),
    ],
)
def test_test_checkpoint_path_rejects_missing_validation_best(
    trainer: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CourtAlignmentTrainingRunner()
    monkeypatch.setattr(
        runner,
        "validate_runtime_config",
        lambda _: _runtime(checkpoint_enabled=True),
    )

    with pytest.raises(RuntimeError, match="[Cc]heckpoint"):
        runner.test_checkpoint_path({}, cast(Any, trainer))


def test_test_checkpoint_path_rejects_disabled_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CourtAlignmentTrainingRunner()
    monkeypatch.setattr(
        runner,
        "validate_runtime_config",
        lambda _: _runtime(checkpoint_enabled=False),
    )

    with pytest.raises(RuntimeError, match="without checkpointing"):
        runner.test_checkpoint_path({}, cast(Any, SimpleNamespace()))

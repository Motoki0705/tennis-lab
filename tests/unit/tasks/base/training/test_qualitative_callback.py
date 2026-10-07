"""Unit tests for QualitativeLoggingCallback selection/gating logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from src.tasks.base.training.qualitative_callback import (
    QualitativeLoggingCallback,
    _detach_to_cpu,
)

pytestmark = pytest.mark.unit


def _callback(**overrides: object) -> QualitativeLoggingCallback:
    config: dict[str, object] = {
        "every_n_epochs": 1,
        "num_samples": 1,
        "enabled": True,
        "selection_mode": "random",
        "selected_indices": None,
    }
    config.update(overrides)
    return QualitativeLoggingCallback(**config)  # type: ignore[arg-type]


def test_init_rejects_non_positive_intervals() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _callback(every_n_epochs=0, num_samples=0)


def test_select_random_subset_bounded() -> None:
    cb = _callback(num_samples=3, selection_mode="random")
    selected = cb._select_batch_indices(total=10)
    assert len(selected) == 3
    assert selected <= set(range(10))


def test_select_random_caps_at_total() -> None:
    cb = _callback(num_samples=20, selection_mode="random")
    selected = cb._select_batch_indices(total=5)
    assert selected == set(range(5))


def test_select_zero_total_empty() -> None:
    cb = _callback(num_samples=4)
    assert cb._select_batch_indices(total=0) == set()


def test_fixed_indices_mode() -> None:
    cb = _callback(
        selection_mode="fixed_indices", selected_indices=[0, 2, 4]
    )
    assert cb._select_batch_indices(total=6) == {0, 2, 4}


def test_fixed_indices_requires_list() -> None:
    cb = _callback(selection_mode="fixed_indices", selected_indices=None)
    with pytest.raises(ValueError, match="non-empty selected_indices"):
        cb._select_batch_indices(total=6)


def test_fixed_indices_out_of_range_raises() -> None:
    cb = _callback(
        selection_mode="fixed_indices", selected_indices=[0, 7]
    )
    with pytest.raises(ValueError, match="out-of-range"):
        cb._select_batch_indices(total=5)


def test_unknown_selection_mode_raises() -> None:
    cb = _callback(selection_mode="bogus")
    with pytest.raises(ValueError, match="must be 'random' or"):
        cb._select_batch_indices(total=5)


class _Trainer:
    def __init__(self, *, enabled_epoch: int = 0, sanity: bool = False) -> None:
        self.current_epoch = enabled_epoch
        self.sanity_checking = sanity
        self.global_rank = 0
        self.global_step = 1
        self.val_dataloaders: list[list[dict[str, Any]]] = [[{}]]
        self.logger = None


def _as_lightning_trainer(trainer: _Trainer) -> pl.Trainer:
    return cast(pl.Trainer, trainer)


def test_should_log_respects_enabled_flag() -> None:
    cb = _callback(enabled=False)
    assert cb._should_log(_as_lightning_trainer(_Trainer())) is False


def test_should_log_skips_sanity_check() -> None:
    cb = _callback(enabled=True)
    assert cb._should_log(_as_lightning_trainer(_Trainer(sanity=True))) is False


def test_should_log_every_n_epochs() -> None:
    cb = _callback(enabled=True, every_n_epochs=3)
    assert cb._should_log(_as_lightning_trainer(_Trainer(enabled_epoch=0))) is False
    assert cb._should_log(_as_lightning_trainer(_Trainer(enabled_epoch=2))) is True
    assert cb._should_log(_as_lightning_trainer(_Trainer(enabled_epoch=3))) is True
    assert cb._should_log(_as_lightning_trainer(_Trainer(enabled_epoch=1))) is False


class _Renderer(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.epochs: list[int] = []

    def render_qualitative_samples(
        self,
        batches: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        artifact_dir: Path,
        tb_writer: Any,
        global_step: int,
        epoch: int,
    ) -> None:
        assert len(batches) == len(outputs) == 1
        assert batches[0]["x"].device.type == "cpu"
        assert not outputs[0]["prediction"].requires_grad
        self.epochs.append(epoch + 1)


def _validation(
    callback: QualitativeLoggingCallback,
    trainer: _Trainer,
    module: pl.LightningModule,
    *,
    collect: bool = True,
) -> None:
    wrapped = _as_lightning_trainer(trainer)
    callback.on_validation_epoch_start(wrapped, module)
    if collect:
        callback.on_validation_batch_end(
            wrapped,
            module,
            {"prediction": torch.ones(1, requires_grad=True)},
            {"x": torch.ones(1)},
            0,
        )
    callback.on_validation_epoch_end(wrapped, module)
    assert not callback._collected_batches
    assert not callback._collected_outputs


@pytest.fixture
def scheduled_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> QualitativeLoggingCallback:
    callback = _callback(every_n_epochs=5)
    monkeypatch.setattr(
        callback, "_resolve_artifact_dir", lambda trainer, epoch: tmp_path / str(epoch)
    )
    return callback


def test_state_roundtrip_skips_already_rendered_interval(
    scheduled_callback: QualitativeLoggingCallback,
) -> None:
    module = _Renderer()
    _validation(scheduled_callback, _Trainer(enabled_epoch=5), module)
    assert module.epochs == [6]
    restored = _callback(every_n_epochs=5)
    restored.load_state_dict(scheduled_callback.state_dict())
    for epoch in (5, 8):
        assert not restored._should_log(_as_lightning_trainer(_Trainer(enabled_epoch=epoch)))
    assert restored._should_log(_as_lightning_trainer(_Trainer(enabled_epoch=11)))


@pytest.mark.parametrize("epoch", [-1, True, 1.5, "5"])
def test_invalid_checkpoint_state_is_rejected(epoch: object) -> None:
    with pytest.raises(ValueError, match="last_logged_epoch"):
        _callback().load_state_dict({"last_logged_epoch": epoch})


def test_sanity_check_does_not_consume_due_interval(
    scheduled_callback: QualitativeLoggingCallback,
) -> None:
    module = _Renderer()
    trainer = _Trainer(enabled_epoch=5, sanity=True)
    before = scheduled_callback.state_dict()
    _validation(scheduled_callback, trainer, module)
    assert scheduled_callback.state_dict() == before
    assert module.epochs == []
    trainer.sanity_checking = False
    _validation(scheduled_callback, trainer, module)
    assert module.epochs == [6]


def test_disabled_logging_does_not_collect_or_advance(
    scheduled_callback: QualitativeLoggingCallback,
) -> None:
    scheduled_callback.enabled = False
    module = _Renderer()
    before = scheduled_callback.state_dict()
    _validation(scheduled_callback, _Trainer(enabled_epoch=5), module)
    assert scheduled_callback.state_dict() == before
    assert module.epochs == []


def test_empty_validation_keeps_interval_due(
    scheduled_callback: QualitativeLoggingCallback,
) -> None:
    module = _Renderer()
    _validation(scheduled_callback, _Trainer(enabled_epoch=5), module, collect=False)
    _validation(scheduled_callback, _Trainer(enabled_epoch=8), module)
    assert module.epochs == [9]


def test_nonzero_rank_advances_schedule_without_rendering(
    scheduled_callback: QualitativeLoggingCallback,
) -> None:
    module = _Renderer()
    trainer = _Trainer(enabled_epoch=5)
    trainer.global_rank = 1
    _validation(scheduled_callback, trainer, module)
    assert module.epochs == []
    assert not scheduled_callback._should_log(_as_lightning_trainer(trainer))


def test_missing_renderer_is_not_silently_ignored(
    scheduled_callback: QualitativeLoggingCallback,
) -> None:
    with pytest.raises(TypeError, match="requires render_qualitative_samples"):
        _validation(scheduled_callback, _Trainer(enabled_epoch=5), pl.LightningModule())


def test_detach_to_cpu_recurses_structures() -> None:
    t = torch.ones(2, requires_grad=True)
    data = {"a": t, "b": [t, t], "c": ("x", 3)}
    out = _detach_to_cpu(data)
    assert out["a"].requires_grad is False
    assert out["a"].device.type == "cpu"
    assert isinstance(out["b"], list)
    assert isinstance(out["c"], tuple)
    assert out["c"] == ("x", 3)


def test_detach_to_cpu_passthrough_non_tensor() -> None:
    assert _detach_to_cpu(5) == 5
    assert _detach_to_cpu("hello") == "hello"
    arr = np.zeros(3)
    assert _detach_to_cpu(arr) is arr

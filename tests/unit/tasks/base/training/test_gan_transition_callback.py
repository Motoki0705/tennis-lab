"""Unit tests for the GAN transition callback's deterministic logic.

Tests cover config parsing, improvement detection, early-stopping-style
patience, the supervised->GAN switch, warmup weight ramping, and state
(de)serialization. PyTorch Lightning is not driven end-to-end; lightweight fake
trainer/module stand-ins exercise the pure decision logic.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.tasks.base.training.gan_transition_callback import GANTransitionCallback

pytestmark = pytest.mark.unit


class _FakeTrainer:
    def __init__(self, current_epoch: int = 0, metrics: dict[str, Any] | None = None) -> None:
        self.current_epoch = current_epoch
        self.callback_metrics = metrics or {}


class _FakeModule:
    """Records calls the callback makes into the LightningModule."""

    def __init__(self) -> None:
        self.weights: list[float] = []
        self.activated_at: int | None = None

    def set_gan_weight(self, w: float) -> None:
        self.weights.append(w)

    def activate_gan_phase(self, epoch: int) -> None:
        self.activated_at = epoch


def _cfg(**gan: Any) -> dict[str, Any]:
    return {"training": {"gan": gan}}


def test_disabled_by_default() -> None:
    cb = GANTransitionCallback({})
    assert cb.enabled is False
    assert cb.has_switched_to_gan is False


def test_config_parsing() -> None:
    cb = GANTransitionCallback(
        _cfg(
            enabled=True,
            target_weight=0.25,
            warmup_epochs=4,
            transition={
                "monitor": "val/acc",
                "mode": "max",
                "min_delta": 0.01,
                "patience": 2,
                "min_supervised_epochs": 3,
            },
        )
    )
    assert cb.enabled is True
    assert cb.monitor == "val/acc"
    assert cb.mode == "max"
    assert cb.min_delta == pytest.approx(0.01)
    assert cb.patience == 2
    assert cb.min_supervised_epochs == 3
    assert cb.gan_target_weight == pytest.approx(0.25)
    assert cb.gan_warmup_epochs == 4


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="transition mode"):
        GANTransitionCallback(_cfg(enabled=True, transition={"mode": "bogus"}))


def test_is_improvement_min_mode() -> None:
    cb = GANTransitionCallback(_cfg(enabled=True, transition={"mode": "min", "min_delta": 0.1}))
    cb.best_score = 1.0
    assert cb._is_improvement(0.85) is True  # below best - delta
    assert cb._is_improvement(0.95) is False  # within delta band
    assert cb._is_improvement(1.2) is False


def test_is_improvement_max_mode() -> None:
    cb = GANTransitionCallback(_cfg(enabled=True, transition={"mode": "max", "min_delta": 0.1}))
    cb.best_score = 1.0
    assert cb._is_improvement(1.2) is True
    assert cb._is_improvement(1.05) is False


def _validation_step(cb: GANTransitionCallback, module: _FakeModule, epoch: int, value: float) -> None:
    trainer = _FakeTrainer(current_epoch=epoch, metrics={cb.monitor: torch.tensor(value)})
    cb.on_validation_epoch_end(trainer, module)


def test_patience_triggers_switch() -> None:
    cb = GANTransitionCallback(
        _cfg(
            enabled=True,
            transition={"monitor": "val/loss", "mode": "min", "patience": 2, "min_delta": 0.0},
        )
    )
    module = _FakeModule()

    _validation_step(cb, module, epoch=0, value=1.0)  # sets best
    assert cb.best_score == pytest.approx(1.0)
    _validation_step(cb, module, epoch=1, value=1.0)  # no improvement -> bad 1
    assert cb.bad_epochs == 1
    assert cb.has_switched_to_gan is False
    _validation_step(cb, module, epoch=2, value=1.0)  # no improvement -> bad 2 -> switch
    assert cb.has_switched_to_gan is True
    assert cb.switch_epoch == 3  # current_epoch + 1
    assert module.activated_at == 3


def test_improvement_resets_bad_epochs() -> None:
    cb = GANTransitionCallback(
        _cfg(
            enabled=True,
            transition={"monitor": "val/loss", "mode": "min", "patience": 2, "min_delta": 0.0},
        )
    )
    module = _FakeModule()
    _validation_step(cb, module, 0, 1.0)
    _validation_step(cb, module, 1, 1.0)  # bad 1
    assert cb.bad_epochs == 1
    _validation_step(cb, module, 2, 0.5)  # improvement -> reset
    assert cb.bad_epochs == 0
    assert cb.best_score == pytest.approx(0.5)


def test_min_supervised_epochs_gates_switch() -> None:
    cb = GANTransitionCallback(
        _cfg(
            enabled=True,
            transition={
                "monitor": "val/loss",
                "mode": "min",
                "patience": 0,
                "min_supervised_epochs": 3,
            },
        )
    )
    module = _FakeModule()
    # patience<=0 would switch immediately, but min_supervised_epochs gates it.
    _validation_step(cb, module, epoch=0, value=1.0)  # epoch+1=1 < 3 -> ignored
    assert cb.has_switched_to_gan is False
    _validation_step(cb, module, epoch=2, value=1.0)  # epoch+1=3 -> allowed, patience<=0 -> switch
    assert cb.has_switched_to_gan is True


def test_missing_monitor_metric_is_noop() -> None:
    cb = GANTransitionCallback(
        _cfg(enabled=True, transition={"monitor": "val/loss", "patience": 0})
    )
    module = _FakeModule()
    trainer = _FakeTrainer(current_epoch=5, metrics={})  # monitor absent
    cb.on_validation_epoch_end(trainer, module)
    assert cb.has_switched_to_gan is False


def test_on_fit_start_zeroes_weight() -> None:
    cb = GANTransitionCallback(_cfg(enabled=True))
    module = _FakeModule()
    cb.on_fit_start(_FakeTrainer(), module)
    assert module.weights == [0.0]


def test_warmup_weight_ramp() -> None:
    cb = GANTransitionCallback(
        _cfg(enabled=True, target_weight=0.2, warmup_epochs=4, transition={"patience": 0})
    )
    cb.has_switched_to_gan = True
    cb.switch_epoch = 2
    module = _FakeModule()

    # epoch 2 -> since_switch = 1 -> progress 1/4 -> 0.05
    cb.on_train_epoch_start(_FakeTrainer(current_epoch=2), module)
    assert module.weights[-1] == pytest.approx(0.2 * 0.25)
    # epoch 5 -> since_switch = 4 -> progress 1.0 -> full target
    cb.on_train_epoch_start(_FakeTrainer(current_epoch=5), module)
    assert module.weights[-1] == pytest.approx(0.2)
    # beyond warmup stays clamped at target
    cb.on_train_epoch_start(_FakeTrainer(current_epoch=10), module)
    assert module.weights[-1] == pytest.approx(0.2)


def test_warmup_le_one_sets_target_immediately() -> None:
    cb = GANTransitionCallback(
        _cfg(enabled=True, target_weight=0.3, warmup_epochs=1)
    )
    cb.has_switched_to_gan = True
    cb.switch_epoch = 0
    module = _FakeModule()
    cb.on_train_epoch_start(_FakeTrainer(current_epoch=0), module)
    assert module.weights[-1] == pytest.approx(0.3)


def test_state_dict_roundtrip() -> None:
    cb = GANTransitionCallback(_cfg(enabled=True))
    cb.best_score = 0.42
    cb.bad_epochs = 2
    cb.has_switched_to_gan = True
    cb.switch_epoch = 7

    restored = GANTransitionCallback(_cfg(enabled=True))
    restored.load_state_dict(cb.state_dict())
    assert restored.best_score == pytest.approx(0.42)
    assert restored.bad_epochs == 2
    assert restored.has_switched_to_gan is True
    assert restored.switch_epoch == 7

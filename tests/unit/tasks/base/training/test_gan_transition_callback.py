"""Unit tests for the GAN transition callback's deterministic epoch logic.

The transition is deterministic: the GAN phase activates at a fixed configured
epoch (`training.gan.transition.start_epoch`) rather than on loss convergence.
Tests cover config parsing/validation, the fit-start max-epochs guard, the
one-time supervised->GAN switch, warmup weight ramping, and resume behaviour.
PyTorch Lightning is not driven end-to-end; lightweight fake trainer/module
stand-ins exercise the pure decision logic.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.tasks.base.training.gan_transition_callback import GANTransitionCallback

pytestmark = pytest.mark.unit


class _FakeTrainer:
    def __init__(self, current_epoch: int = 0, max_epochs: int | None = None) -> None:
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs


class _FakeModule:
    """Records the calls the callback makes into the LightningModule."""

    def __init__(self) -> None:
        self.weights: list[float] = []
        self.activated_at: list[int] = []

    def set_gan_weight(self, w: float) -> None:
        self.weights.append(w)

    def activate_gan_phase(self, epoch: int) -> None:
        self.activated_at.append(epoch)


def _cfg(**gan: Any) -> dict[str, Any]:
    return {"training": {"gan": gan}}


def _run_epoch(cb: GANTransitionCallback, module: _FakeModule, epoch: int) -> None:
    cb.on_train_epoch_start(_FakeTrainer(current_epoch=epoch), module)


# --------------------------------------------------------------------------- #
# Config parsing / validation
# --------------------------------------------------------------------------- #
def test_disabled_by_default() -> None:
    cb = GANTransitionCallback({})
    assert cb.enabled is False
    assert cb.has_switched_to_gan is False
    assert cb.start_epoch == 0


def test_config_parsing() -> None:
    cb = GANTransitionCallback(
        _cfg(
            enabled=True,
            target_weight=0.25,
            warmup_epochs=4,
            transition={"start_epoch": 50},
        )
    )
    assert cb.enabled is True
    assert cb.start_epoch == 50
    assert cb.gan_target_weight == pytest.approx(0.25)
    assert cb.gan_warmup_epochs == 4


def test_missing_start_epoch_when_enabled_raises() -> None:
    with pytest.raises(ValueError, match="start_epoch is required"):
        GANTransitionCallback(_cfg(enabled=True))


def test_missing_start_epoch_when_disabled_defaults_to_zero() -> None:
    cb = GANTransitionCallback(_cfg(enabled=False))
    assert cb.start_epoch == 0


def test_negative_start_epoch_raises() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        GANTransitionCallback(_cfg(enabled=True, transition={"start_epoch": -1}))


# --------------------------------------------------------------------------- #
# on_fit_start
# --------------------------------------------------------------------------- #
def test_on_fit_start_zeroes_weight() -> None:
    cb = GANTransitionCallback(_cfg(enabled=True, transition={"start_epoch": 5}))
    module = _FakeModule()
    cb.on_fit_start(_FakeTrainer(max_epochs=10), module)
    assert module.weights == [0.0]


def test_on_fit_start_rejects_start_epoch_ge_max_epochs() -> None:
    cb = GANTransitionCallback(_cfg(enabled=True, transition={"start_epoch": 100}))
    module = _FakeModule()
    with pytest.raises(ValueError, match="never activates"):
        cb.on_fit_start(_FakeTrainer(max_epochs=100), module)


def test_on_fit_start_disabled_still_zeroes_and_skips_guard() -> None:
    # Disabled: even a start_epoch >= max_epochs must not raise (no GAN at all).
    cb = GANTransitionCallback(_cfg(enabled=False, transition={"start_epoch": 100}))
    module = _FakeModule()
    cb.on_fit_start(_FakeTrainer(max_epochs=100), module)
    assert module.weights == [0.0]


# --------------------------------------------------------------------------- #
# Deterministic switch behaviour
# --------------------------------------------------------------------------- #
def test_no_switch_before_start_epoch() -> None:
    cb = GANTransitionCallback(_cfg(enabled=True, transition={"start_epoch": 10}))
    module = _FakeModule()
    for epoch in range(10):
        _run_epoch(cb, module, epoch)
    assert cb.has_switched_to_gan is False
    assert module.activated_at == []
    assert module.weights == []  # weight untouched during supervised phase


def test_switch_at_start_epoch() -> None:
    cb = GANTransitionCallback(
        _cfg(
            enabled=True,
            target_weight=0.2,
            warmup_epochs=4,
            transition={"start_epoch": 10},
        )
    )
    module = _FakeModule()
    _run_epoch(cb, module, 10)
    assert cb.has_switched_to_gan is True
    assert module.activated_at == [10]  # activated at the actual current epoch
    # since_switch = 10 - 10 + 1 = 1 -> progress 1/4 -> 0.05
    assert module.weights[-1] == pytest.approx(0.2 * 0.25)


def test_activation_happens_exactly_once() -> None:
    cb = GANTransitionCallback(
        _cfg(enabled=True, transition={"start_epoch": 2}, warmup_epochs=4)
    )
    module = _FakeModule()
    for epoch in range(2, 8):
        _run_epoch(cb, module, epoch)
    assert module.activated_at == [2]  # only one activation despite many epochs


def test_disabled_callback_is_noop_on_epoch_start() -> None:
    cb = GANTransitionCallback(_cfg(enabled=False, transition={"start_epoch": 5}))
    module = _FakeModule()
    _run_epoch(cb, module, 100)
    assert cb.has_switched_to_gan is False
    assert module.activated_at == []
    assert module.weights == []


# --------------------------------------------------------------------------- #
# Warmup ramp
# --------------------------------------------------------------------------- #
def test_warmup_weight_ramp() -> None:
    cb = GANTransitionCallback(
        _cfg(
            enabled=True,
            target_weight=0.2,
            warmup_epochs=4,
            transition={"start_epoch": 2},
        )
    )
    module = _FakeModule()

    # epoch 2 -> since_switch = 1 -> progress 1/4 -> 0.05
    _run_epoch(cb, module, 2)
    assert module.weights[-1] == pytest.approx(0.2 * 0.25)
    # epoch 5 -> since_switch = 4 -> progress 1.0 -> full target
    _run_epoch(cb, module, 5)
    assert module.weights[-1] == pytest.approx(0.2)
    # beyond warmup stays clamped at target
    _run_epoch(cb, module, 10)
    assert module.weights[-1] == pytest.approx(0.2)


def test_warmup_le_one_sets_target_immediately() -> None:
    cb = GANTransitionCallback(
        _cfg(
            enabled=True,
            target_weight=0.3,
            warmup_epochs=1,
            transition={"start_epoch": 0},
        )
    )
    module = _FakeModule()
    _run_epoch(cb, module, 0)
    assert module.weights[-1] == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #
def test_resume_into_gan_phase_reactivates() -> None:
    # Fresh callback (as on process restart) resumed at an epoch already inside
    # the GAN phase: it must re-activate the module and pick the correct ramp
    # point, using the *actual* epoch for LR-schedule scaling.
    cb = GANTransitionCallback(
        _cfg(
            enabled=True,
            target_weight=0.2,
            warmup_epochs=4,
            transition={"start_epoch": 10},
        )
    )
    module = _FakeModule()
    _run_epoch(cb, module, 13)
    assert cb.has_switched_to_gan is True
    assert module.activated_at == [13]  # scaled from the resumed epoch
    # ramp anchored to start_epoch: since_switch = 13 - 10 + 1 = 4 -> full target
    assert module.weights[-1] == pytest.approx(0.2)

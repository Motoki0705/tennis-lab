"""Unit tests for ManualGANTrainingStrategy's pure decision logic.

The optimizer/backward machinery requires a real LightningModule + Trainer and
is covered by the integration smoke suite. Here we test the deterministic state
logic: weight clamping, phase activation, step routing, and epoch-end scheduler
stepping (with fake schedulers).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.tasks.base.training.gan_training import ManualGANTrainingStrategy

pytestmark = pytest.mark.unit


def _strategy(scheduler_interval: str = "step") -> ManualGANTrainingStrategy:
    return ManualGANTrainingStrategy(
        generator_gradient_clip_val=None,
        discriminator_gradient_clip_val=None,
        scheduler_interval=scheduler_interval,
    )


def test_initial_state() -> None:
    s = _strategy()
    assert s.phase_active is False
    assert s.start_epoch is None
    assert s.current_weight == 0.0
    assert s.supervised_only_step_count == 0
    assert s.hybrid_gan_step_count == 0


def test_activate_phase_sets_flags() -> None:
    s = _strategy()
    s.activate_phase(5)
    assert s.phase_active is True
    assert s.start_epoch == 5


def test_set_weight_clamps_negative_to_zero() -> None:
    s = _strategy()
    s.set_weight(0.3)
    assert s.current_weight == pytest.approx(0.3)
    s.set_weight(-1.0)
    assert s.current_weight == 0.0


def test_shared_step_routes_supervised_when_inactive() -> None:
    """When the GAN phase is inactive, training routes through the supervised step."""
    s = _strategy()
    calls: list[str] = []

    class _Module:
        def _supervised_step(self, batch, stage):
            calls.append(stage)
            return ("loss", {"m": 1})

        def _gan_step(self, batch, stage):  # pragma: no cover - should not run
            calls.append("gan")
            return ("loss", {})

    # Strategy.shared_step calls the strategy's own _supervised_step/_gan_step,
    # which delegate into the module. Patch those to observe routing.
    s._supervised_step = lambda module, batch, stage: ("sup", stage)  # type: ignore
    s._gan_step = lambda module, batch, stage: ("gan", stage)  # type: ignore

    assert s.shared_step(_Module(), None, "val") == ("sup", "val")
    assert s.shared_step(_Module(), None, "train") == ("sup", "train")  # inactive
    s.activate_phase(0)
    assert s.shared_step(_Module(), None, "train") == ("gan", "train")  # active
    assert s.shared_step(_Module(), None, "val") == ("sup", "val")  # non-train always sup


def test_on_train_epoch_end_steps_schedulers_epoch_interval() -> None:
    s = _strategy(scheduler_interval="epoch")

    class _Sched:
        def __init__(self) -> None:
            self.steps = 0

        def step(self) -> None:
            self.steps += 1

    gen, disc = _Sched(), _Sched()

    class _Module:
        def lr_schedulers(self) -> Any:
            return [gen, disc]

    # Inactive phase -> only generator scheduler steps.
    s.on_train_epoch_end(_Module())
    assert gen.steps == 1
    assert disc.steps == 0

    # Active phase -> both step.
    s.activate_phase(0)
    s.on_train_epoch_end(_Module())
    assert gen.steps == 2
    assert disc.steps == 1


def test_on_train_epoch_end_noop_for_step_interval() -> None:
    s = _strategy(scheduler_interval="step")

    class _Sched:
        def __init__(self) -> None:
            self.steps = 0

        def step(self) -> None:  # pragma: no cover
            self.steps += 1

    sched = _Sched()

    class _Module:
        def lr_schedulers(self):
            return sched

    s.on_train_epoch_end(_Module())
    assert sched.steps == 0  # step-interval schedulers are not advanced at epoch end


def test_manual_optimizers_requires_exactly_two() -> None:
    s = _strategy()

    class _Module:
        def optimizers(self):
            return ["only_one"]

    with pytest.raises(RuntimeError, match="generator and discriminator"):
        s._manual_optimizers(_Module())


def test_manual_schedulers_normalizes_to_list() -> None:
    s = _strategy()

    class _ModuleNone:
        def lr_schedulers(self):
            return None

    class _ModuleSingle:
        def lr_schedulers(self):
            return "sched"

    class _ModuleList:
        def lr_schedulers(self):
            return ["a", "b"]

    assert s._manual_schedulers(_ModuleNone()) == []
    assert s._manual_schedulers(_ModuleSingle()) == ["sched"]
    assert s._manual_schedulers(_ModuleList()) == ["a", "b"]


def test_unwrap_optimizer() -> None:
    s = _strategy()

    class _Wrapped:
        optimizer = "inner"

    assert s._unwrap_optimizer(_Wrapped()) == "inner"
    assert s._unwrap_optimizer("plain") == "plain"

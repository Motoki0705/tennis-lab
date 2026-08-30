"""Unit tests for ManualGANTrainingStrategy's pure decision logic.

The optimizer/backward machinery requires a real LightningModule + Trainer and
is covered by the integration smoke suite. Here we test the deterministic state
logic: weight validation, phase activation, step routing, and epoch-end scheduler
stepping (with fake schedulers).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from src.tasks.base.training.gan_loss import LSGANLoss
from src.tasks.base.training.gan_training import (
    ManualGANSupportMixin,
    ManualGANTrainingStrategy,
    loss_component_metrics,
)
from src.tasks.base.training.metric_logging import (
    WeightedMetricAccumulator,
    uniform_metric_logging_contract,
)

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


def test_gan_mixin_exposes_discriminator_as_additional_compile_target() -> None:
    discriminator = nn.Linear(3, 1)
    mixin = ManualGANSupportMixin()
    mixin.gan_enabled = True
    mixin.discriminator = discriminator

    assert mixin.additional_compilation_targets() == {"discriminator": discriminator}


def test_gan_mixin_has_no_additional_target_when_disabled() -> None:
    mixin = ManualGANSupportMixin()
    mixin.gan_enabled = False
    mixin.discriminator = None

    assert mixin.additional_compilation_targets() == {}


def test_loss_component_metrics_omits_exact_total_alias() -> None:
    assert loss_component_metrics(
        {
            "total": torch.tensor(1.5),
            "position": torch.tensor(1.0),
            "reprojection": torch.tensor(0.5),
        }
    ) == {
        "loss_position": 1.0,
        "loss_reprojection": 0.5,
    }


def test_shared_gan_logger_keeps_ball_compatible_enabled_metrics() -> None:
    class _Module(ManualGANSupportMixin):
        def __init__(self) -> None:
            self.gan_enabled = True
            self.gan_training = _strategy()
            self.gan_training.activate_phase(0)
            self.gan_training.set_weight(0.2)
            self.names: list[str] = []

        def log(self, name: str, value: Any, **kwargs: Any) -> None:
            del value, kwargs
            self.names.append(name)

    module = _Module()

    module._log_gan_metrics(
        "train",
        {
            "loss_gan_generator": torch.tensor(0.1),
            "loss_gan_discriminator": torch.tensor(0.3),
        },
    )

    assert module.names == [
        "train/gan_weight",
        "train/gan_phase_active",
        "train/loss_gan_generator",
        "train/loss_gan_discriminator",
    ]


def test_standard_test_artifact_uses_headlines_and_separate_diagnostics() -> None:
    class _Tracker:
        def __init__(self) -> None:
            self.reset_count = 0

        def compute(self) -> dict[str, float]:
            return {
                "position_error_m": 0.2,
                "position_accuracy_0.3m": 0.8,
                "endpoint_error_m": 0.4,
                "x_error_m": 0.1,
            }

        def reset(self) -> None:
            self.reset_count += 1

    class _Module(ManualGANSupportMixin):
        metric_logging_contract = uniform_metric_logging_contract(
            "fixture",
            headline_keys=(
                "position_error_m",
                "position_accuracy_0.3m",
                "endpoint_error_m",
            ),
            progress_bar_keys=("position_error_m",),
        )

        def __init__(self) -> None:
            self.tracker = _Tracker()
            self.saved: dict[str, Any] = {}
            self._test_metric_diagnostic_accumulator = WeightedMetricAccumulator()

        def _metric_tracker_for_stage(self, stage: str) -> _Tracker:
            assert stage == "test"
            return self.tracker

        def save_test_predictions(self, **kwargs: Any) -> None:
            self.saved = kwargs

        def log(self, *args: Any, **kwargs: Any) -> None:
            pass

    module = _Module()
    module._test_metric_diagnostic_accumulator.update(
        {
            "position_error_m": 0.3,
            "reference_index_1_position_error_m": 0.25,
            "loss_position": 0.05,
        },
        weight=2,
    )

    module.on_test_epoch_end()

    assert module.saved["metrics"] == {
        "position_error_m": 0.2,
        "position_accuracy_0.3m": 0.8,
        "endpoint_error_m": 0.4,
    }
    assert module.saved["diagnostic_metrics"] == {
        "x_error_m": 0.1,
        "reference_index_1_position_error_m": 0.25,
        "loss_position": 0.05,
    }


def test_epoch_flush_logs_only_the_fixed_headline_contract() -> None:
    class _Tracker:
        def compute(self) -> dict[str, float]:
            return {
                "position_error_m": 0.2,
                "position_accuracy_0.3m": 0.8,
                "endpoint_error_m": 0.4,
                "x_error_m": 0.1,
            }

        def reset(self) -> None:
            pass

    class _Module(ManualGANSupportMixin):
        metric_logging_contract = uniform_metric_logging_contract(
            "fixture",
            headline_keys=(
                "position_error_m",
                "position_accuracy_0.3m",
                "endpoint_error_m",
            ),
            progress_bar_keys=("position_error_m",),
        )

        def __init__(self) -> None:
            self.tracker = _Tracker()
            self.names: list[str] = []

        def _metric_tracker_for_stage(self, stage: str) -> _Tracker:
            assert stage in {"train", "val", "test"}
            return self.tracker

        def log(self, name: str, value: Any, **kwargs: Any) -> None:
            del value, kwargs
            self.names.append(name)

    module = _Module()

    for stage in ("train", "val", "test"):
        module._flush_stage_metrics(stage)
        assert module.names == [
            f"{stage}/position_error_m",
            f"{stage}/position_accuracy_0.3m",
            f"{stage}/endpoint_error_m",
        ]
        module.names.clear()


def test_activate_phase_sets_flags() -> None:
    s = _strategy()
    s.activate_phase(5)
    assert s.phase_active is True
    assert s.start_epoch == 5


def test_activate_phase_rejects_negative_epoch() -> None:
    with pytest.raises(ValueError, match="start_epoch"):
        _strategy().activate_phase(-1)


def test_strategy_rejects_invalid_scheduler_interval() -> None:
    with pytest.raises(ValueError, match="scheduler_interval"):
        _strategy("batch")


@pytest.mark.parametrize("clip", [-1.0, float("inf"), float("nan")])
def test_strategy_rejects_invalid_gradient_clip(clip: float) -> None:
    with pytest.raises(ValueError, match="finite value >= 0"):
        ManualGANTrainingStrategy(
            generator_gradient_clip_val=clip,
            discriminator_gradient_clip_val=None,
            scheduler_interval="step",
        )


def test_set_weight_rejects_negative_value() -> None:
    s = _strategy()
    s.set_weight(0.3)
    assert s.current_weight == pytest.approx(0.3)
    with pytest.raises(ValueError, match="finite value >= 0"):
        s.set_weight(-1.0)


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

    shared_step: Any = s.shared_step
    assert shared_step(_Module(), None, "val") == ("sup", "val")
    assert shared_step(_Module(), None, "train") == ("sup", "train")  # inactive
    s.activate_phase(0)
    assert shared_step(_Module(), None, "train") == ("gan", "train")  # active
    assert shared_step(_Module(), None, "val") == ("sup", "val")  # non-train always sup


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


class _RecordingDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.padding_masks: list[torch.Tensor] = []

    def forward(
        self,
        sequence: torch.Tensor,
        *,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.padding_masks.append(padding_mask.detach().clone())
        valid = (~padding_mask).unsqueeze(-1)
        masked = torch.where(valid, sequence, torch.zeros_like(sequence))
        denominator = valid.sum(dim=1).clamp_min(1)
        return (masked.sum(dim=1) / denominator).mean(dim=-1) * self.scale


def test_gan_step_passes_named_padding_mask_to_discriminator() -> None:
    strategy = _strategy()
    strategy.activate_phase(0)
    strategy.set_weight(0.2)
    generator_value = nn.Parameter(torch.tensor(0.25))
    discriminator = _RecordingDiscriminator()
    generator_optimizer = torch.optim.SGD([generator_value], lr=0.01)
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.01)
    gan_padding_mask = torch.tensor([[False, True, False]])

    class _Module:
        gan_loss_fn = LSGANLoss()

        def __init__(self) -> None:
            self.discriminator = discriminator

        def _compute_supervised_result(
            self,
            batch: object,
            stage: str,
        ) -> dict[str, object]:
            del batch, stage
            return {
                "loss": generator_value.square(),
                "metrics": {},
                "gan_fake": generator_value.reshape(1, 1, 1).expand(1, 3, 1),
                "gan_real": torch.tensor([[[0.1], [0.2], [0.3]]]),
                "gan_padding_mask": gan_padding_mask,
            }

        def optimizers(self) -> list[torch.optim.Optimizer]:
            return [generator_optimizer, discriminator_optimizer]

        def lr_schedulers(self) -> list[object]:
            return []

        def toggle_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
            del optimizer

        def untoggle_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
            del optimizer

        def manual_backward(self, loss: torch.Tensor) -> None:
            loss.backward()

    loss, metrics = strategy._gan_step(_Module(), batch=None, stage="train")

    assert torch.isfinite(loss)
    assert "loss_hybrid_total" not in metrics
    assert "loss_gan_generator" in metrics
    assert "loss_gan_discriminator" in metrics
    assert strategy.hybrid_gan_step_count == 1
    assert len(discriminator.padding_masks) == 3
    assert all(
        torch.equal(mask, gan_padding_mask) for mask in discriminator.padding_masks
    )

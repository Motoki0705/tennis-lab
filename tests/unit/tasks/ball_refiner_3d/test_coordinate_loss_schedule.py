"""Schedules are explicit opt-ins; default position/event coefficients stay equal."""

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.tasks.ball_refiner_3d.config import (
    ReconstructionConfig,
    parse_gan,
    parse_section,
    training_config,
)
from src.tasks.ball_refiner_3d.losses import (
    generator_objective,
    reconstruction_weight_at,
)
from src.tasks.base.training.gan_schedule import gan_weight_at
from src.utils.paths import PROJECT_ROOT


def configuration(overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "src/tasks/ball_refiner_3d/configs")):
        return compose(config_name="train_coordinates", overrides=["training.reconstruction.enabled=true", "training.reconstruction.final_weight=0", "training.reconstruction.start_step=2000", "training.reconstruction.decay_steps=1000", "training.gan.enabled=true", "training.gan.schedule_enabled=true", *overrides])


@pytest.mark.parametrize("step,position,adversarial", [
    (1, 1.0, 0.0), (500, 1.0, 0.0), (501, 1.0, 0.001), (1500, 1.0, 1.0),
    (2000, 1.0, 1.0), (2001, 0.999, 1.0), (2500, 0.5, 1.0),
    (2999, 0.001, 1.0), (3000, 0.0, 1.0), (3001, 0.0, 1.0), (4000, 0.0, 1.0),
])
def test_requested_update_boundaries(step, position, adversarial):
    raw, _, _ = training_config(configuration())
    reconstruction = parse_section(ReconstructionConfig, raw["training"]["reconstruction"])
    gan, _ = parse_gan(raw["training"]["gan"])
    assert reconstruction_weight_at(step - 1, reconstruction) == pytest.approx(position)
    assert gan_weight_at(step - 1, start=gan.start_step, warmup=gan.warmup_steps, target=gan.target_weight) == pytest.approx(adversarial)


@pytest.mark.parametrize("target_value", [-100.0, 100.0])
def test_zero_position_weight_removes_target_gradient_but_keeps_gan_gradient(target_value):
    prediction = torch.tensor([0.2, 0.7], requires_grad=True)
    target = torch.full_like(prediction, target_value, requires_grad=True)
    reconstruction = torch.nn.functional.smooth_l1_loss(prediction, target, beta=0.02)
    adversarial = (prediction.square().sum() - 1).square()
    expected = torch.autograd.grad(adversarial, prediction, retain_graph=True)[0]
    loss = generator_objective(reconstruction, adversarial, reconstruction_weight=0.0, gan_weight=1.0)
    loss.backward()
    assert target.grad is None
    assert prediction.grad.abs().sum() > 0
    torch.testing.assert_close(prediction.grad, expected)
    assert loss.item() == adversarial.item()


@pytest.mark.parametrize("overrides,message", [
    (["training.reconstruction.final_weight=-1"], "Reconstruction weights"),
    (["training.reconstruction.initial_weight=0"], "Reconstruction weights"),
    (["training.reconstruction.decay_steps=0"], "decay schedule"),
    (["training.reconstruction.start_step=3500"], "within training.steps"),
    (["training.gan.enabled=false"], "active GAN"),
    (["training.gan.transition.start_step=3000"], "active GAN"),
    (["training.reconstruction.final_weight=nan"], "Nonfinite"),
])
def test_invalid_or_objectiveless_schedules_fail_before_training(overrides, message):
    with pytest.raises((ValueError, TypeError), match=message):
        training_config(configuration(overrides))

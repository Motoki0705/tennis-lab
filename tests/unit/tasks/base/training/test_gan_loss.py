"""Unit tests for the least-squares GAN loss helpers."""

from __future__ import annotations

import pytest
import torch

from src.tasks.base.training.gan_loss import LSGANLoss

pytestmark = pytest.mark.unit


def test_generator_loss_matches_formula() -> None:
    loss = LSGANLoss()
    fake = torch.tensor([0.0, 0.5, 1.0])
    # 0.5 * mean((fake - 1)^2) = 0.5 * mean([1, 0.25, 0]) = 0.5 * 0.4166...
    expected = 0.5 * ((fake - 1.0) ** 2).mean()
    assert torch.allclose(loss.generator_loss(fake), expected)


def test_generator_loss_zero_when_fake_classified_real() -> None:
    loss = LSGANLoss()
    fake = torch.ones(4, 1)  # already at real_label
    assert loss.generator_loss(fake).item() == pytest.approx(0.0)


def test_discriminator_loss_matches_formula() -> None:
    loss = LSGANLoss()
    real = torch.tensor([1.0, 0.8])
    fake = torch.tensor([0.0, 0.3])
    real_term = ((real - 1.0) ** 2).mean()
    fake_term = ((fake - 0.0) ** 2).mean()
    expected = 0.5 * (real_term + fake_term)
    assert torch.allclose(loss.discriminator_loss(real, fake), expected)


def test_discriminator_loss_zero_when_perfectly_separated() -> None:
    loss = LSGANLoss()
    real = torch.ones(3, 1)
    fake = torch.zeros(3, 1)
    assert loss.discriminator_loss(real, fake).item() == pytest.approx(0.0)


def test_custom_labels_respected() -> None:
    loss = LSGANLoss(real_label=0.9, fake_label=0.1)
    assert loss.real_label == pytest.approx(0.9)
    assert loss.fake_label == pytest.approx(0.1)
    real = torch.full((2, 1), 0.9)
    fake = torch.full((2, 1), 0.1)
    assert loss.discriminator_loss(real, fake).item() == pytest.approx(0.0)


def test_losses_are_scalar_and_finite() -> None:
    loss = LSGANLoss()
    fake = torch.randn(5, 3)
    real = torch.randn(5, 3)
    g = loss.generator_loss(fake)
    d = loss.discriminator_loss(real, fake)
    assert g.ndim == 0 and d.ndim == 0
    assert torch.isfinite(g) and torch.isfinite(d)


def test_gradients_flow_to_inputs() -> None:
    loss = LSGANLoss()
    fake = torch.randn(4, 1, requires_grad=True)
    loss.generator_loss(fake).backward()
    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()

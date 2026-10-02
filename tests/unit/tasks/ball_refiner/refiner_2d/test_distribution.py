"""Density, covariance and pixel units checked against independent formulas."""

from dataclasses import replace

import pytest
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D


def distribution() -> BallGMM2D:
    return BallGMM2D(
        means=torch.tensor([[[[0.2, 0.4], [0.8, 0.6]]]], dtype=torch.float64),
        scale_tril=torch.tensor(
            [[[[[0.1, 0], [0.03, 0.2]], [[0.2, 0], [-0.05, 0.1]]]]],
            dtype=torch.float64,
        ),
        mixture_logits=torch.tensor([[[1.0, -0.3]]], dtype=torch.float64),
        presence_logits=torch.tensor([[0.7]], dtype=torch.float64),
    )


def test_density_and_gradients_match_torch_mixture() -> None:
    actual = distribution()
    for value in (actual.means, actual.scale_tril, actual.mixture_logits):
        value.requires_grad_()
    uv = torch.tensor([[[0.35, 0.45]]], dtype=torch.float64)
    reference = MixtureSameFamily(
        Categorical(logits=actual.mixture_logits),
        MultivariateNormal(actual.means, scale_tril=actual.scale_tril),
    ).log_prob(uv)
    result = actual.log_prob(uv)
    torch.testing.assert_close(result, reference)
    tensors = (actual.means, actual.scale_tril, actual.mixture_logits)
    gradients = torch.autograd.grad(result.sum(), tensors, retain_graph=True)
    expected = torch.autograd.grad(reference.sum(), tensors)
    for gradient, reference_gradient in zip(gradients, expected, strict=True):
        torch.testing.assert_close(gradient, reference_gradient)


def test_component_order_has_no_effect_on_density() -> None:
    original = distribution()
    permuted = replace(
        original,
        means=original.means.flip(-2),
        scale_tril=original.scale_tril.flip(-3),
        mixture_logits=original.mixture_logits.flip(-1),
    )
    uv = torch.tensor([[[0.5, 0.5]]], dtype=torch.float64)
    torch.testing.assert_close(original.log_prob(uv), permuted.log_prob(uv))
    torch.testing.assert_close(
        original.weights.sum(-1), torch.ones(1, 1, dtype=torch.float64)
    )
    torch.testing.assert_close(
        original.presence_probability, original.presence_logits.sigmoid()
    )


def test_pixel_covariance_includes_anisotropic_cross_term_and_density_jacobian() -> (
    None
):
    value = distribution()
    sizes = torch.tensor([[1920, 1080]], dtype=torch.float64)
    means, covariance = value.pixel_moments(sizes)
    scale = torch.tensor([1919, 1079], dtype=torch.float64)
    expected_covariance = torch.diag(scale) @ value.covariance @ torch.diag(scale)
    torch.testing.assert_close(means, value.means * scale)
    torch.testing.assert_close(covariance, expected_covariance)
    uv = torch.tensor([[[0.21, 0.41]]], dtype=torch.float64)
    pixel_log_prob = MixtureSameFamily(
        Categorical(logits=value.mixture_logits),
        MultivariateNormal(means, covariance_matrix=covariance),
    ).log_prob(uv * scale)
    torch.testing.assert_close(pixel_log_prob, value.log_prob(uv) - scale.log().sum())


@pytest.mark.parametrize(
    "field,change",
    [
        ("means", lambda x: torch.full_like(x, float("nan"))),
        ("means", lambda x: torch.full_like(x, 1.01)),
        ("means", lambda x: x[..., :0, :]),
        ("scale_tril", lambda x: torch.zeros_like(x)),
        ("scale_tril", lambda x: -x),
        ("scale_tril", lambda x: torch.ones_like(x)),
        ("mixture_logits", lambda x: torch.full_like(x, float("inf"))),
        ("mixture_logits", lambda x: x.float()),
        ("presence_logits", lambda x: x.unsqueeze(-1)),
    ],
)
def test_invalid_distribution_fails_instead_of_repairing(field, change) -> None:
    value = distribution()
    with pytest.raises(ValueError):
        replace(value, **{field: change(getattr(value, field))})


@pytest.mark.parametrize(
    "sizes", [[[1, 1080]], [[1920, 0]], [[1920.5, 1080]], [[float("nan"), 1080]]]
)
def test_invalid_pixel_sizes_fail(sizes) -> None:
    with pytest.raises(ValueError):
        distribution().pixel_moments(torch.tensor(sizes))

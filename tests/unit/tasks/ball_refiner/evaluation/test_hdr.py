"""Analytic regions, separated mixture modes, and Monte Carlo reproducibility."""

import math
from dataclasses import replace
from typing import Any

import pytest
import torch

from src.tasks.ball_refiner.evaluation.hdr import highest_density_regions
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D


def gaussian(frames=1):
    return BallGMM2D(
        means=torch.full((1, frames, 1, 2), .5, dtype=torch.float64),
        scale_tril=torch.tensor([[.08, 0], [.03, .04]], dtype=torch.float64).expand(1, frames, 1, 2, 2),
        mixture_logits=torch.zeros(1, frames, 1, dtype=torch.float64),
        presence_logits=torch.zeros(1, frames, dtype=torch.float64),
    )


def test_single_correlated_gaussian_matches_analytic_threshold_area_and_coverage():
    prediction = gaussian(3)
    white = torch.tensor([[[1., 0], [2., 0], [math.sqrt(8), 0]]], dtype=torch.float64)
    target = prediction.means[..., 0, :] + (prediction.scale_tril[..., 0, :, :] @ white[..., None]).squeeze(-1)
    levels = (.5, .9, .95)
    result = highest_density_regions(prediction, target, levels=levels, samples=32768, seed=17, chunk_size=2)
    mass = torch.tensor(levels, dtype=torch.float64)
    expected_threshold = -math.log(2 * math.pi * .08 * .04) + (1 - mass).log()
    expected_area = -2 * math.pi * .08 * .04 * (1 - mass).log()
    torch.testing.assert_close(result.log_threshold, expected_threshold.expand(1, 3, 3), atol=.05, rtol=0)
    torch.testing.assert_close(result.area_uv2, expected_area.expand(1, 3, 3), atol=0, rtol=.035)
    assert result.covered.tolist() == [[[True, True, True], [False, True, True], [False, False, False]]]
    assert bool((result.area_mc_standard_error_uv2 > 0).all())


def test_mixture_hdr_is_not_union_of_individual_component_ellipses():
    prediction = BallGMM2D(
        means=torch.tensor([[[[.2, .2], [.8, .8]]]], dtype=torch.float64),
        scale_tril=torch.diag_embed(torch.tensor([[[[.02, .02], [.2, .2]]]], dtype=torch.float64)),
        mixture_logits=torch.tensor([[[.9, .1]]], dtype=torch.float64).log(),
        presence_logits=torch.zeros(1, 1, dtype=torch.float64),
    )
    result = highest_density_regions(prediction, torch.tensor([[[.8, .8]]], dtype=torch.float64),
                                     levels=(.5, .9, .95), samples=32768, seed=1, chunk_size=1)
    assert not result.covered[0, 0, 0]  # broad, low-density component's own centre is outside 50% HDR
    assert result.area_uv2[0, 0, 0] < .01
    assert bool((result.log_threshold[..., 1:] < result.log_threshold[..., :-1]).all())
    assert bool((result.area_uv2[..., 1:] > result.area_uv2[..., :-1]).all())


def test_chunk_size_target_and_presence_do_not_change_regions():
    prediction = gaussian(5)
    target = prediction.means[..., 0, :]
    first = highest_density_regions(prediction, target, chunk_size=1, levels=(.5, .9), samples=512, seed=31)
    second = highest_density_regions(replace(prediction, presence_logits=prediction.presence_logits + 10),
                                     target + 1, chunk_size=4, levels=(.5, .9), samples=512, seed=31)
    for name in ("log_threshold", "area_uv2", "area_mc_standard_error_uv2"):
        torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=1e-13, atol=1e-13)
    assert bool(first.covered.all()) and not bool(second.covered.any())


def test_diagnostics_do_not_retain_training_gradients():
    prediction = gaussian()
    prediction.means.requires_grad_()
    target = torch.full((1, 1, 2), .5, requires_grad=True)
    result = highest_density_regions(prediction, target, levels=(.5, .9), samples=128, seed=3, chunk_size=1)
    assert not result.log_threshold.requires_grad
    assert not result.area_uv2.requires_grad
    assert not result.area_mc_standard_error_uv2.requires_grad


@pytest.mark.parametrize("overrides", [{"levels": (.9, .5)}, {"levels": (.5, .5)}, {"levels": (1.,)},
                                       {"samples": 2}, {"chunk_size": 0}, {"seed": -1}])
def test_invalid_monte_carlo_settings_fail(overrides):
    options: dict[str, Any] = dict(levels=(.5, .9), samples=128, seed=31, chunk_size=2)
    options.update(overrides)
    with pytest.raises(ValueError):
        highest_density_regions(gaussian(), torch.zeros(1, 1, 2), **options)

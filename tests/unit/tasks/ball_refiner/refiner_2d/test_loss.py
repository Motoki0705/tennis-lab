"""Unknown, absent and amodal-present targets have different likelihoods."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from src.tasks.ball_refiner.refiner_2d import BallGMM2D, Refiner2DTarget, refiner_2d_nll


def prediction() -> BallGMM2D:
    return BallGMM2D(
        means=torch.full((1, 4, 2, 2), 0.5, requires_grad=True),
        scale_tril=(torch.eye(2).expand(1, 4, 2, 2, 2) * 0.1).clone().requires_grad_(),
        mixture_logits=torch.zeros(1, 4, 2, requires_grad=True),
        presence_logits=torch.tensor([[0.4, -0.7, 0.2, 0.9]], requires_grad=True),
    )


def target() -> Refiner2DTarget:
    return Refiner2DTarget(
        uv=torch.tensor(
            [
                [
                    [0.4, 0.6],
                    [float("nan"), float("nan")],
                    [float("nan"), float("nan")],
                    [float("nan"), float("nan")],
                ]
            ]
        ),
        position_valid=torch.tensor([[True, False, False, False]]),
        presence=torch.tensor([[True, False, False, True]]),
        presence_valid=torch.tensor([[True, True, False, True]]),
        weight=torch.tensor([[1.0, 2.0, 100.0, 1.0]]),
    )


def test_joint_nll_known_absent_unknown_and_presence_only() -> None:
    p, y = prediction(), target()
    result = refiner_2d_nll(p, y)
    spatial = -torch.distributions.MultivariateNormal(
        p.means[0, 0, 0],
        scale_tril=p.scale_tril[0, 0, 0],
    ).log_prob(y.uv[0, 0])
    bernoulli = F.binary_cross_entropy_with_logits(
        p.presence_logits[0, [0, 1, 3]],
        y.presence[0, [0, 1, 3]].float(),
        reduction="none",
    ) @ torch.tensor([1.0, 2.0, 1.0])
    torch.testing.assert_close(result.loss, (spatial + bernoulli) / 4)
    assert result.position_weight == 1
    assert result.presence_weight == 4
    result.loss.backward()
    for tensor in (p.means, p.scale_tril, p.mixture_logits, p.presence_logits):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert (
            torch.count_nonzero(tensor.grad[:, 2]) == 0
        )  # Unknown, even with weight 100.
    assert (
        torch.count_nonzero(p.means.grad[:, 1:]) == 0
    )  # No fake absent/presence-only UV.


def test_presence_cannot_hide_position_error() -> None:
    y = target()
    gradients = []
    spatial = []
    for logit in (-80.0, 80.0):
        p = replace(
            prediction(), presence_logits=torch.full((1, 4), logit, requires_grad=True)
        )
        result = refiner_2d_nll(p, y)
        result.loss.backward()
        gradients.append(p.means.grad)
        spatial.append(result.position_nll_sum)
    torch.testing.assert_close(spatial[0], spatial[1])
    torch.testing.assert_close(gradients[0], gradients[1])


def test_all_absent_trains_only_presence() -> None:
    p = prediction()
    y = replace(
        target(),
        uv=torch.full((1, 4, 2), float("nan")),
        position_valid=torch.zeros(1, 4, dtype=torch.bool),
        presence=torch.zeros(1, 4, dtype=torch.bool),
        presence_valid=torch.ones(1, 4, dtype=torch.bool),
        weight=torch.ones(1, 4),
    )
    result = refiner_2d_nll(p, y)
    result.loss.backward()
    assert result.position_weight == 0
    assert result.position_nll_sum == 0
    assert torch.count_nonzero(p.means.grad) == 0
    assert torch.count_nonzero(p.presence_logits.grad) == 4


def test_unknown_tiny_covariance_never_enters_density_arithmetic() -> None:
    p = prediction()
    scales = p.scale_tril.detach().clone()
    scales[:, 2] *= 1e-25
    p = replace(p, scale_tril=scales.requires_grad_())
    result = refiner_2d_nll(p, target())
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert torch.isfinite(p.scale_tril.grad).all()
    assert torch.count_nonzero(p.scale_tril.grad[:, 2]) == 0


def test_additive_totals_reproduce_global_nll_across_batches() -> None:
    p, y = prediction(), target()
    full = refiner_2d_nll(p, y)
    pieces = []
    for sl in (slice(0, 2), slice(2, 4)):
        sliced_p = BallGMM2D(
            means=p.means[:, sl],
            scale_tril=p.scale_tril[:, sl],
            mixture_logits=p.mixture_logits[:, sl],
            presence_logits=p.presence_logits[:, sl],
        )
        sliced_y = Refiner2DTarget(
            uv=y.uv[:, sl],
            position_valid=y.position_valid[:, sl],
            presence=y.presence[:, sl],
            presence_valid=y.presence_valid[:, sl],
            weight=y.weight[:, sl],
        )
        pieces.append(refiner_2d_nll(sliced_p, sliced_y))
    total = sum(r.position_nll_sum + r.presence_bce_sum for r in pieces)
    denominator = sum(r.presence_weight for r in pieces)
    torch.testing.assert_close(full.loss, total / denominator)


@pytest.mark.parametrize(
    "change",
    [
        {"presence_valid": torch.zeros(1, 4, dtype=torch.bool)},
        {"presence": torch.zeros(1, 4, dtype=torch.bool)},
        {"uv": torch.full((1, 4, 2), float("nan"))},
        {"weight": torch.zeros(1, 4)},
        {"weight": -torch.ones(1, 4)},
        {"weight": torch.full((1, 4), float("inf"))},
        {"position_valid": torch.ones(1, 4)},
    ],
)
def test_invalid_or_empty_supervision_is_rejected(change) -> None:
    with pytest.raises(ValueError):
        refiner_2d_nll(prediction(), replace(target(), **change))

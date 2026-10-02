"""Proper joint Bernoulli/conditional-GMM negative log likelihood."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from src.tasks.ball_refiner.refiner_2d.contracts import Refiner2DTarget
from src.tasks.ball_refiner.refiner_2d.distribution import (
    FLOAT_DTYPES,
    BallGMM2D,
    conditional_log_density,
)
from src.tasks.base.model_io.tensors import TensorSpec


@dataclass(frozen=True)
class Refiner2DLoss:
    """Additive totals support epoch aggregation without averaging batch means."""

    loss: Tensor
    position_nll_sum: Tensor
    presence_bce_sum: Tensor
    position_weight: Tensor
    presence_weight: Tensor


def refiner_2d_nll(prediction: BallGMM2D, target: Refiner2DTarget) -> Refiner2DLoss:
    """Mean joint NLL over weighted presence-known frames.

    Spatial density is conditional on presence and is never weighted by the
    predicted presence probability. Absent frames train only the Bernoulli.
    Unknown/padded targets contribute neither term. Empty supervision fails.
    """
    shape = tuple(prediction.presence_logits.shape)
    for name, value, expected, dtypes in (
        ("uv", target.uv, (*shape, 2), FLOAT_DTYPES),
        ("position_valid", target.position_valid, shape, frozenset({torch.bool})),
        ("presence", target.presence, shape, frozenset({torch.bool})),
        ("presence_valid", target.presence_valid, shape, frozenset({torch.bool})),
        ("weight", target.weight, shape, FLOAT_DTYPES),
    ):
        TensorSpec(expected, dtypes).validate(name, value)
        if value.device != prediction.means.device:
            raise ValueError(f"{name} must share the prediction device")
    if bool((target.position_valid & ~(target.presence_valid & target.presence)).any()):
        raise ValueError("Known positions require known positive presence")
    if not bool(torch.isfinite(target.weight).all()) or bool((target.weight < 0).any()):
        raise ValueError("Supervision weights must be finite and nonnegative")
    located_uv = target.uv[target.position_valid]
    if not bool(torch.isfinite(located_uv).all()) or bool(
        ((located_uv < 0) | (located_uv > 1)).any()
    ):
        raise ValueError("Known target positions must be finite normalized UV")

    weight = target.weight.to(prediction.means.dtype)
    presence_weights = weight * target.presence_valid
    position_weights = weight * target.position_valid
    presence_weight = presence_weights.sum()
    if not bool(presence_weight > 0):
        raise ValueError("Batch has no positive-weight supervision")
    # Select known positions BEFORE arithmetic. Even valid but very narrow
    # distributions on unknown frames must not produce inf * 0 or NaN gradients.
    located = target.position_valid & (weight > 0)
    position_nll_sum = (
        -conditional_log_density(
            target.uv[located].to(prediction.means.dtype),
            prediction.means[located],
            prediction.scale_tril[located],
            prediction.mixture_logits[located],
        )
        * position_weights[located]
    ).sum()
    known = target.presence_valid & (weight > 0)
    presence_bce_sum = (
        F.binary_cross_entropy_with_logits(
            prediction.presence_logits[known],
            target.presence[known].to(prediction.presence_logits.dtype),
            reduction="none",
        )
        * presence_weights[known]
    ).sum()
    loss = (position_nll_sum + presence_bce_sum) / presence_weight
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite refiner NLL; inspect GMM scales and targets")
    return Refiner2DLoss(
        loss=loss,
        position_nll_sum=position_nll_sum,
        presence_bce_sum=presence_bce_sum,
        position_weight=position_weights.sum(),
        presence_weight=presence_weight,
    )

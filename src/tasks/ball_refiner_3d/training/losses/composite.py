"""Compose supervised and optional adversarial objectives."""

from __future__ import annotations

import torch


def generator_objective(
    reconstruction: torch.Tensor,
    adversarial: torch.Tensor,
    *,
    reconstruction_weight: float,
    gan_weight: float,
) -> torch.Tensor:
    """A zero reconstruction coefficient removes its graph from the objective."""
    if reconstruction_weight == 0:
        return gan_weight * adversarial
    return reconstruction_weight * reconstruction + gan_weight * adversarial

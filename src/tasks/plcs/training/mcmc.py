"""Stochastic Gradient Langevin Dynamics (SGLD) noise injection for PLCS.

This is the "MCMC as a training strategy" option from issue #519. The PLCS
rotation head sits at a flat saddle when the predicted yaw is exactly 180deg
from the target: the ``1 - cos`` rotation loss has gradient ``sin(theta)`` which
vanishes at the antipode, so plain gradient descent cannot escape the flip.

SGLD turns the optimiser into an approximate MCMC sampler by perturbing the
parameters with Gaussian noise after each update::

    theta <- theta - lr * grad(L)  +  N(0, 2 * lr * temperature)

The injected noise gives every step a non-zero probability of moving uphill,
which lets training tunnel out of the 180deg basin even though the loss
gradient there is ~0. ``noise_scale`` and ``temperature`` tune how aggressive
the exploration is; ``decay`` anneals the noise toward 0 so the run can still
settle into a sharp minimum at the end (simulated-annealing style).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.utils.configuration import (
    ConfigurationTypeError,
    SemanticConfigurationError,
)


@dataclass
class MCMCConfig:
    """Configuration for SGLD / Langevin-dynamics noise injection."""

    enabled: bool
    # Multiplier on the base Langevin noise std (sqrt(2 * lr * temperature)).
    noise_scale: float
    # SGLD temperature; >1 explores more, <1 stays closer to MAP.
    temperature: float
    # Skip noise for the first ``warmup_epochs`` epochs (let the model settle
    # into a sensible region before exploring).
    warmup_epochs: int
    # Noise annealing schedule over training progress: "none" | "cosine" | "linear".
    decay: str
    # Which parameters receive noise: "all" or "rotation" (rotation-related only).
    target: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCMCConfig:
        expected = {
            "enabled",
            "noise_scale",
            "temperature",
            "warmup_epochs",
            "decay",
            "target",
        }
        if set(data) != expected:
            raise ValueError(
                f"Invalid training.mcmc keys: missing={sorted(expected - set(data))}, "
                f"unknown={sorted(set(data) - expected)}."
            )
        if type(data["enabled"]) is not bool:
            raise ConfigurationTypeError("training.mcmc.enabled must be exact bool.")
        for key in {"noise_scale", "temperature"}:
            value = data[key]
            if type(value) not in {float, int}:
                raise ConfigurationTypeError(
                    f"training.mcmc.{key} must be exact float | int."
                )
        if type(data["warmup_epochs"]) is not int:
            raise ConfigurationTypeError(
                "training.mcmc.warmup_epochs must be exact int."
            )
        for key in {"decay", "target"}:
            if type(data[key]) is not str:
                raise ConfigurationTypeError(f"training.mcmc.{key} must be exact str.")
        noise_scale = float(data["noise_scale"])
        temperature = float(data["temperature"])
        if not math.isfinite(noise_scale) or noise_scale < 0.0:
            raise SemanticConfigurationError(
                "training.mcmc.noise_scale must be finite and non-negative."
            )
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise SemanticConfigurationError(
                "training.mcmc.temperature must be finite and positive."
            )
        enabled = data["enabled"]
        if enabled and noise_scale == 0.0:
            raise SemanticConfigurationError(
                "training.mcmc.noise_scale must be positive when MCMC is enabled."
            )
        warmup_epochs = data["warmup_epochs"]
        if warmup_epochs < 0:
            raise SemanticConfigurationError(
                "training.mcmc.warmup_epochs must be non-negative."
            )
        decay = data["decay"]
        if decay not in {"none", "cosine", "linear"}:
            raise SemanticConfigurationError(
                "training.mcmc.decay must be 'none', 'cosine', or 'linear'."
            )
        target = data["target"]
        if target not in {"all", "rotation"}:
            raise SemanticConfigurationError(
                "training.mcmc.target must be 'all' or 'rotation'."
            )
        return cls(
            enabled=enabled,
            noise_scale=noise_scale,
            temperature=temperature,
            warmup_epochs=warmup_epochs,
            decay=decay,
            target=target,
        )


class LangevinNoiseInjector:
    """Injects SGLD noise into model parameters after each optimizer step."""

    # Substrings identifying rotation-related parameters when target="rotation".
    _ROTATION_KEYS = ("rotation", "rot_")

    def __init__(self, config: MCMCConfig) -> None:
        self.config = config

    def _selected_parameters(self, model: nn.Module) -> Iterator[torch.Tensor]:
        if self.config.target == "rotation":
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(key in name for key in self._ROTATION_KEYS):
                    yield param
            return
        for param in model.parameters():
            if param.requires_grad:
                yield param

    def _decay_factor(self, progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        decay = self.config.decay
        if decay == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if decay == "linear":
            return 1.0 - progress
        return 1.0

    def inject(
        self,
        model: nn.Module,
        *,
        lr: float,
        epoch: int,
        progress: float,
    ) -> float:
        """Perturb parameters in-place; returns the noise std actually used."""
        if not self.config.enabled or epoch < self.config.warmup_epochs:
            return 0.0
        base_std = self.config.noise_scale * math.sqrt(
            max(2.0 * lr * self.config.temperature, 0.0)
        )
        std = base_std * self._decay_factor(progress)
        if std <= 0.0:
            return 0.0
        with torch.no_grad():
            for param in self._selected_parameters(model):
                param.add_(torch.randn_like(param) * std)
        return std

"""Reusable Low-Rank Adaptation (LoRA) helpers for ``nn.Linear`` modules.

LoRA (Hu et al., 2021) freezes a pretrained linear weight ``W`` and learns a
low-rank update ``B @ A`` so that the effective transform becomes
``W x + (alpha / rank) * B (A x)``. Only the small ``A`` and ``B`` factors are
trained, which makes it a cheap way to adapt a frozen backbone such as DINOv3.

This module is domain-agnostic: it wraps any ``torch.nn.Linear`` whose local
attribute name matches the requested targets, leaving everything else untouched.
Downstream tasks compose it through ``src.utils.models.loading.dinov3``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class LoRAConfig:
    """Hyper-parameters that describe a LoRA adaptation.

    Attributes:
        enabled: Whether LoRA should be applied at all.
        rank: Inner dimension ``r`` of the low-rank update. Must be positive.
        alpha: Scaling numerator; the update is scaled by ``alpha / rank``.
        dropout: Dropout probability applied to the LoRA input branch.
        target_modules: Local attribute names of ``nn.Linear`` modules to wrap
            (for example ``("qkv", "proj")`` for attention projections).
    """

    enabled: bool
    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.enabled:
            if self.rank <= 0:
                raise ValueError("LoRA rank must be positive when enabled.")
            if self.alpha <= 0:
                raise ValueError("LoRA alpha must be positive when enabled.")
            if not 0.0 <= self.dropout < 1.0:
                raise ValueError("LoRA dropout must be in [0, 1).")
            if not self.target_modules:
                raise ValueError("LoRA target_modules must be non-empty when enabled.")

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
    ) -> LoRAConfig:
        """Build a config from a complete, validated Hydra/OmegaConf mapping."""
        expected = {"enabled", "rank", "alpha", "dropout", "target_modules"}
        missing = expected - set(mapping)
        unknown = set(mapping) - expected
        if missing or unknown:
            raise ValueError(
                "Invalid LoRA config keys: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        enabled = mapping["enabled"]
        rank = mapping["rank"]
        alpha = mapping["alpha"]
        dropout = mapping["dropout"]
        target_modules = mapping["target_modules"]
        if type(enabled) is not bool:
            raise TypeError("LoRA enabled must be exactly bool.")
        if type(rank) is not int:
            raise TypeError("LoRA rank must be exactly int.")
        if type(alpha) is not float:
            raise TypeError("LoRA alpha must be exactly float.")
        if type(dropout) is not float:
            raise TypeError("LoRA dropout must be exactly float.")
        if not isinstance(target_modules, Sequence) or isinstance(
            target_modules, str | bytes
        ):
            raise TypeError("LoRA target_modules must be a sequence of strings.")
        if any(type(name) is not str for name in target_modules):
            raise TypeError("Every LoRA target module must be exactly str.")
        return cls(
            enabled=enabled,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=tuple(target_modules),
        )


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable low-rank update."""

    def __init__(
        self,
        base_linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear modules.")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1).")

        self.base = base_linear
        self.base.requires_grad_(False)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        weight = base_linear.weight
        self.lora_dropout: nn.Module = (
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )
        self.lora_a = nn.Parameter(
            torch.empty(self.rank, self.in_features, dtype=weight.dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(self.out_features, self.rank, dtype=weight.dtype)
        )
        # Kaiming init on A and zeros on B keep the initial update at zero, so
        # the wrapped layer starts as an exact copy of the pretrained linear.
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self._align_lora_to(weight.device)

    def _align_lora_to(self, device: torch.device) -> None:
        self.lora_a.data = self.lora_a.data.to(device)
        self.lora_b.data = self.lora_b.data.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        lora_input = self.lora_dropout(inputs)
        update = torch.nn.functional.linear(lora_input, self.lora_a)
        update = torch.nn.functional.linear(update, self.lora_b)
        result: torch.Tensor = base_output + self.scaling * update
        return result

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )


def apply_lora(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Sequence[str],
) -> list[str]:
    """Replace matching ``nn.Linear`` children with :class:`LoRALinear` in place.

    A linear layer is wrapped when its local attribute name (the final
    dotted component of its qualified name) appears in ``target_modules``.
    Already-wrapped layers are skipped so the call is idempotent.

    Returns the fully-qualified names of the wrapped linear layers.
    """
    targets = set(target_modules)
    if not targets:
        raise ValueError("target_modules must be non-empty.")

    wrapped: list[str] = []
    for parent_name, parent in module.named_modules():
        for child_name, child in list(parent.named_children()):
            if child_name not in targets:
                continue
            if isinstance(child, LoRALinear):
                continue
            if not isinstance(child, nn.Linear):
                continue
            setattr(
                parent,
                child_name,
                LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
            )
            qualified = f"{parent_name}.{child_name}" if parent_name else child_name
            wrapped.append(qualified)

    if not wrapped:
        raise ValueError(
            "apply_lora matched no nn.Linear modules for target_modules="
            f"{sorted(targets)}."
        )
    return wrapped


def iter_lora_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    """Yield the trainable LoRA parameters contained in ``module``."""
    for submodule in module.modules():
        if isinstance(submodule, LoRALinear):
            yield submodule.lora_a
            yield submodule.lora_b


def mark_only_lora_as_trainable(module: nn.Module) -> None:
    """Freeze every parameter in ``module`` except the LoRA factors."""
    module.requires_grad_(False)
    for parameter in iter_lora_parameters(module):
        parameter.requires_grad_(True)


__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "apply_lora",
    "iter_lora_parameters",
    "mark_only_lora_as_trainable",
]

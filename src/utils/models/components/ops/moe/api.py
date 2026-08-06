"""Construction-time resolution for MoE dispatch/combine implementations."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch

from src.utils.models.components.ops.loader import require_moe_cuda_extension
from src.utils.models.components.ops.moe._autograd import (
    cuda_moe_combine,
    cuda_moe_dispatch,
)
from src.utils.models.components.ops.moe.reference import (
    DropPolicy,
    MoEDispatchResult,
    reference_moe_combine,
    reference_moe_dispatch,
)

MoEDispatch = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], MoEDispatchResult]
MoECombine = Callable[[torch.Tensor, MoEDispatchResult], torch.Tensor]
MoECapacity = Callable[[torch.Tensor], int]


@dataclass(frozen=True)
class MoEOperations:
    """Fixed dispatch/combine callables selected before tensor execution."""

    dispatch: MoEDispatch
    combine: MoECombine


def resolve_moe_operations(
    *,
    backend: Literal["reference", "cuda"],
    num_experts: int,
    capacity_factor: float | None,
    drop_policy: DropPolicy,
) -> MoEOperations:
    """Resolve one MoE backend and bind its static configuration."""
    capacity_for = _resolve_capacity_policy(
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        drop_policy=drop_policy,
    )
    if backend == "reference":

        def dispatch_reference(
            tokens: torch.Tensor,
            expert_indices: torch.Tensor,
            expert_weights: torch.Tensor,
        ) -> MoEDispatchResult:
            return reference_moe_dispatch(
                tokens,
                expert_indices,
                expert_weights,
                num_experts=num_experts,
                capacity=capacity_for(expert_indices),
            )

        return MoEOperations(
            dispatch=dispatch_reference,
            combine=reference_moe_combine,
        )
    if backend != "cuda":
        raise ValueError(f"Unsupported MoE backend: {backend!r}.")

    extension = require_moe_cuda_extension()

    def dispatch_cuda(
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> MoEDispatchResult:
        return cuda_moe_dispatch(
            tokens,
            expert_indices,
            expert_weights,
            num_experts=num_experts,
            capacity=capacity_for(expert_indices),
            extension=extension,
        )

    return MoEOperations(
        dispatch=dispatch_cuda,
        combine=partial(cuda_moe_combine, extension=extension),
    )


def _resolve_capacity_policy(
    *,
    num_experts: int,
    capacity_factor: float | None,
    drop_policy: DropPolicy,
) -> MoECapacity:
    if type(num_experts) is not int or num_experts <= 0:
        raise ValueError(f"num_experts must be a positive int, got {num_experts!r}.")
    if drop_policy == "none":
        return lambda expert_indices: expert_indices.numel()
    if drop_policy != "capacity":
        raise ValueError(f"Unsupported drop_policy={drop_policy!r}.")
    if capacity_factor is None or capacity_factor <= 0.0:
        raise ValueError(
            "capacity_factor must be positive when drop_policy='capacity'."
        )

    def capacity_with_drops(expert_indices: torch.Tensor) -> int:
        total_assignments = int(expert_indices.numel())
        if total_assignments == 0:
            return 0
        average_assignments = total_assignments / num_experts
        return max(1, math.ceil(average_assignments * capacity_factor))

    return capacity_with_drops


__all__ = ["MoECombine", "MoEDispatch", "MoEOperations", "resolve_moe_operations"]

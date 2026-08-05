from __future__ import annotations

from typing import Literal

import torch

from src.utils.configuration import OperationEnvironmentConfig, operation_environment
from src.utils.models.components.ops.loader import get_moe_cuda_extension
from src.utils.models.components.ops.moe._autograd import (
    cuda_moe_combine,
    cuda_moe_dispatch,
)
from src.utils.models.components.ops.moe.reference import (
    MoEDispatchResult,
    compute_moe_capacity,
    reference_moe_combine,
    reference_moe_dispatch,
    validate_moe_routing,
)

DropPolicy = Literal["none", "capacity"]


def moe_dispatch(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    *,
    num_experts: int,
    capacity_factor: float | None = None,
    drop_policy: DropPolicy = "none",
    capacity: int | None = None,
    use_cuda: bool | None = None,
) -> MoEDispatchResult:
    environment = operation_environment()
    expert_indices, expert_weights = validate_moe_routing(
        tokens,
        expert_indices,
        expert_weights,
        num_experts=num_experts,
    )
    capacity_value = compute_moe_capacity(
        expert_indices,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        drop_policy=drop_policy,
        capacity=capacity,
    )
    if _should_use_cuda(tokens, use_cuda, environment):
        return cuda_moe_dispatch(
            tokens,
            expert_indices,
            expert_weights,
            num_experts=num_experts,
            capacity=capacity_value,
        )
    return reference_moe_dispatch(
        tokens,
        expert_indices,
        expert_weights,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        drop_policy=drop_policy,
        capacity=capacity_value,
    )


def moe_combine(
    expert_outputs: torch.Tensor,
    dispatch_result: MoEDispatchResult,
    *,
    use_cuda: bool | None = None,
) -> torch.Tensor:
    environment = operation_environment()
    _validate_combine_inputs(expert_outputs, dispatch_result)
    if _should_use_cuda(expert_outputs, use_cuda, environment):
        return cuda_moe_combine(expert_outputs, dispatch_result)
    return reference_moe_combine(expert_outputs, dispatch_result)


def _should_use_cuda(
    tensor: torch.Tensor,
    use_cuda: bool | None,
    environment: OperationEnvironmentConfig,
) -> bool:
    force_reference = environment.force_moe_reference
    if use_cuda is False or force_reference:
        if use_cuda is True and force_reference:
            raise RuntimeError("TENNIS_LAB_FORCE_MOE_REFERENCE is set")
        return False
    if not tensor.is_cuda:
        if use_cuda is True:
            raise RuntimeError("use_cuda=True requires CUDA tensors")
        return False
    if get_moe_cuda_extension() is None:
        if use_cuda is True:
            raise RuntimeError("MoE CUDA extension is not available")
        return False
    return True


def _validate_combine_inputs(
    expert_outputs: torch.Tensor,
    dispatch_result: MoEDispatchResult,
) -> None:
    if expert_outputs.ndim != 3:
        raise ValueError(
            f"expert_outputs must have shape [num_experts, capacity, hidden], "
            f"got {tuple(expert_outputs.shape)}"
        )
    if expert_outputs.shape[:2] != dispatch_result.expert_inputs.shape[:2]:
        raise ValueError(
            "expert_outputs first two dimensions must match dispatch_result.expert_inputs"
        )
    if expert_outputs.device != dispatch_result.expert_indices.device:
        raise ValueError(
            "expert_outputs and dispatch_result must be on the same device"
        )

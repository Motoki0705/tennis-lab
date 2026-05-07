from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

DropPolicy = Literal["none", "capacity"]


@dataclass(frozen=True)
class MoEDispatchResult:
    """Output of `moe_dispatch` shared by reference and CUDA backends."""

    expert_inputs: torch.Tensor
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    locations: torch.Tensor
    combine_mask: torch.Tensor
    expert_counts: torch.Tensor
    capacity: int


def validate_moe_routing(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 2:
        raise ValueError(
            f"tokens must have shape [tokens, hidden], got {tuple(tokens.shape)}"
        )
    if expert_indices.shape != expert_weights.shape:
        raise ValueError(
            "expert_indices and expert_weights must have the same shape, got "
            f"{tuple(expert_indices.shape)} and {tuple(expert_weights.shape)}"
        )
    if expert_indices.ndim != 2:
        raise ValueError(
            "expert_indices and expert_weights must have shape [tokens, top_k], "
            f"got {tuple(expert_indices.shape)}"
        )
    if expert_indices.shape[0] != tokens.shape[0]:
        raise ValueError("routing tensors must have the same token dimension as tokens")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if expert_indices.device != tokens.device or expert_weights.device != tokens.device:
        raise ValueError(
            "tokens, expert_indices, and expert_weights must be on the same device"
        )
    if expert_weights.dtype != tokens.dtype:
        expert_weights = expert_weights.to(dtype=tokens.dtype)
    if expert_indices.dtype != torch.long:
        expert_indices = expert_indices.to(dtype=torch.long)
    invalid = (expert_indices < 0) | (expert_indices >= num_experts)
    if bool(invalid.any().item()):
        raise ValueError("expert_indices contains values outside [0, num_experts)")
    return expert_indices.contiguous(), expert_weights.contiguous()


def compute_moe_capacity(
    expert_indices: torch.Tensor,
    *,
    num_experts: int,
    capacity_factor: float | None = None,
    drop_policy: DropPolicy = "none",
    capacity: int | None = None,
) -> int:
    if capacity is not None:
        if capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {capacity}")
        return int(capacity)
    if drop_policy not in ("none", "capacity"):
        raise ValueError(f"Unsupported drop_policy={drop_policy}")

    total_assignments = int(expert_indices.numel())
    if total_assignments == 0:
        return 0

    if (
        drop_policy == "capacity"
        and capacity_factor is not None
        and capacity_factor > 0
    ):
        average_assignments = total_assignments / num_experts
        return max(1, int(math.ceil(average_assignments * capacity_factor)))

    flat_indices = expert_indices.reshape(-1)
    counts = torch.bincount(flat_indices, minlength=num_experts)
    return int(counts.max().item())


def reference_moe_dispatch(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    *,
    num_experts: int,
    capacity_factor: float | None = None,
    drop_policy: DropPolicy = "none",
    capacity: int | None = None,
) -> MoEDispatchResult:
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

    num_tokens, hidden_dim = tokens.shape
    top_k = expert_indices.shape[1]
    locations = torch.full(
        (num_tokens, top_k),
        -1,
        dtype=torch.long,
        device=tokens.device,
    )
    combine_mask = torch.zeros(
        (num_tokens, top_k),
        dtype=torch.bool,
        device=tokens.device,
    )
    expert_counts = torch.zeros(num_experts, dtype=torch.long, device=tokens.device)

    for token_idx in range(num_tokens):
        for top_idx in range(top_k):
            expert_idx = int(expert_indices[token_idx, top_idx].item())
            slot = int(expert_counts[expert_idx].item())
            if slot < capacity_value:
                locations[token_idx, top_idx] = slot
                combine_mask[token_idx, top_idx] = True
                expert_counts[expert_idx] += 1

    expert_inputs_flat = tokens.new_zeros((num_experts * capacity_value, hidden_dim))
    token_ids, flat_slots = _active_token_and_slot_indices(
        expert_indices,
        locations,
        combine_mask,
        capacity=capacity_value,
    )
    if token_ids.numel() > 0:
        expert_inputs_flat.index_copy_(0, flat_slots, tokens.index_select(0, token_ids))

    expert_inputs = expert_inputs_flat.reshape(num_experts, capacity_value, hidden_dim)
    return MoEDispatchResult(
        expert_inputs=expert_inputs,
        expert_indices=expert_indices,
        expert_weights=expert_weights,
        locations=locations,
        combine_mask=combine_mask,
        expert_counts=expert_counts,
        capacity=capacity_value,
    )


def reference_moe_combine(
    expert_outputs: torch.Tensor,
    dispatch_result: MoEDispatchResult,
) -> torch.Tensor:
    if expert_outputs.ndim != 3:
        raise ValueError(
            f"expert_outputs must have shape [num_experts, capacity, hidden], "
            f"got {tuple(expert_outputs.shape)}"
        )
    if expert_outputs.shape[:2] != dispatch_result.expert_inputs.shape[:2]:
        raise ValueError(
            "expert_outputs first two dimensions must match dispatch_result.expert_inputs"
        )

    num_tokens = dispatch_result.expert_indices.shape[0]
    hidden_dim = expert_outputs.shape[-1]
    output = expert_outputs.new_zeros((num_tokens, hidden_dim))
    token_ids, flat_slots = _active_token_and_slot_indices(
        dispatch_result.expert_indices,
        dispatch_result.locations,
        dispatch_result.combine_mask,
        capacity=dispatch_result.capacity,
    )
    if token_ids.numel() == 0:
        return output

    active_weights = dispatch_result.expert_weights[dispatch_result.combine_mask]
    flat_expert_outputs = expert_outputs.reshape(-1, hidden_dim)
    weighted_outputs = (
        flat_expert_outputs.index_select(0, flat_slots) * active_weights[:, None]
    )
    output.index_add_(0, token_ids, weighted_outputs)
    return output


def _active_token_and_slot_indices(
    expert_indices: torch.Tensor,
    locations: torch.Tensor,
    combine_mask: torch.Tensor,
    *,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if capacity == 0:
        empty = torch.empty(0, dtype=torch.long, device=expert_indices.device)
        return empty, empty

    num_tokens, top_k = expert_indices.shape
    token_grid = torch.arange(
        num_tokens, device=expert_indices.device
    ).repeat_interleave(top_k)
    active_mask = combine_mask.reshape(-1)
    token_ids = token_grid[active_mask]
    expert_ids = expert_indices.reshape(-1)[active_mask]
    active_locations = locations.reshape(-1)[active_mask]
    flat_slots = expert_ids * capacity + active_locations
    return token_ids.to(dtype=torch.long), flat_slots.to(dtype=torch.long)

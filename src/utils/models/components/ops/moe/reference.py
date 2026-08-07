from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

DropPolicy = Literal["none", "capacity"]


@dataclass(frozen=True)
class MoEDispatchResult:
    """Prepared dispatch state shared by reference and CUDA backends."""

    expert_inputs: torch.Tensor
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    locations: torch.Tensor
    combine_mask: torch.Tensor
    expert_counts: torch.Tensor
    capacity: int


def reference_moe_dispatch(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    *,
    num_experts: int,
    capacity: int,
) -> MoEDispatchResult:
    num_tokens, hidden_dim = tokens.shape
    top_k = expert_indices.shape[1]
    flat_experts = expert_indices.reshape(-1)
    assignment_by_expert = F.one_hot(
        flat_experts,
        num_classes=num_experts,
    ).to(dtype=torch.long)
    flat_locations = (
        (assignment_by_expert.cumsum(dim=0) - 1)
        .gather(1, flat_experts.unsqueeze(1))
        .squeeze(1)
    )
    locations = flat_locations.reshape(num_tokens, top_k)
    combine_mask = locations < capacity
    expert_counts = assignment_by_expert.sum(dim=0).clamp_max(capacity)

    expert_inputs_flat = tokens.new_zeros((num_experts * capacity, hidden_dim))
    token_ids, flat_slots = _active_token_and_slot_indices(
        expert_indices,
        locations,
        combine_mask,
        capacity=capacity,
    )
    expert_inputs_flat.index_copy_(0, flat_slots, tokens.index_select(0, token_ids))

    expert_inputs = expert_inputs_flat.reshape(num_experts, capacity, hidden_dim)
    return MoEDispatchResult(
        expert_inputs=expert_inputs,
        expert_indices=expert_indices,
        expert_weights=expert_weights,
        locations=locations,
        combine_mask=combine_mask,
        expert_counts=expert_counts,
        capacity=capacity,
    )


def reference_moe_combine(
    expert_outputs: torch.Tensor,
    dispatch_result: MoEDispatchResult,
) -> torch.Tensor:
    num_tokens = dispatch_result.expert_indices.shape[0]
    hidden_dim = expert_outputs.shape[-1]
    output = expert_outputs.new_zeros((num_tokens, hidden_dim))
    token_ids, flat_slots = _active_token_and_slot_indices(
        dispatch_result.expert_indices,
        dispatch_result.locations,
        dispatch_result.combine_mask,
        capacity=dispatch_result.capacity,
    )
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

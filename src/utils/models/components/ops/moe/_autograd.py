from __future__ import annotations

from types import ModuleType
from typing import Any, cast

import torch

from src.utils.models.components.ops.moe.reference import MoEDispatchResult


class _CudaMoEDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        num_experts: int,
        capacity: int,
        extension: ModuleType,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expert_inputs, locations, combine_mask, expert_counts = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            extension.moe_dispatch_forward(
                tokens.contiguous(),
                expert_indices.contiguous(),
                num_experts,
                capacity,
            ),
        )
        ctx.save_for_backward(expert_indices.contiguous(), locations, combine_mask)
        ctx.num_tokens = tokens.shape[0]
        ctx.extension = extension
        return expert_inputs, locations, combine_mask, expert_counts

    @staticmethod
    def backward(
        ctx: Any,
        grad_expert_inputs: torch.Tensor,
        grad_locations: torch.Tensor | None,
        grad_combine_mask: torch.Tensor | None,
        grad_expert_counts: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        del grad_locations, grad_combine_mask, grad_expert_counts
        expert_indices, locations, combine_mask = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor], ctx.saved_tensors
        )
        extension = cast(ModuleType, ctx.extension)
        grad_tokens = cast(
            torch.Tensor,
            extension.moe_dispatch_backward(
                grad_expert_inputs.contiguous(),
                expert_indices,
                locations,
                combine_mask,
                int(ctx.num_tokens),
            ),
        )
        return grad_tokens, None, None, None, None, None


class _CudaMoECombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        expert_outputs: torch.Tensor,
        expert_indices: torch.Tensor,
        locations: torch.Tensor,
        expert_weights: torch.Tensor,
        combine_mask: torch.Tensor,
        extension: ModuleType,
    ) -> torch.Tensor:
        output = cast(
            torch.Tensor,
            extension.moe_combine_forward(
                expert_outputs.contiguous(),
                expert_indices.contiguous(),
                locations.contiguous(),
                expert_weights.contiguous(),
                combine_mask.contiguous(),
            ),
        )
        ctx.save_for_backward(
            expert_outputs.contiguous(),
            expert_indices.contiguous(),
            locations.contiguous(),
            expert_weights.contiguous(),
            combine_mask.contiguous(),
        )
        ctx.extension = extension
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, torch.Tensor, None, None]:
        expert_outputs, expert_indices, locations, expert_weights, combine_mask = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            ctx.saved_tensors,
        )
        extension = cast(ModuleType, ctx.extension)
        grad_expert_outputs, grad_expert_weights = cast(
            tuple[torch.Tensor, torch.Tensor],
            extension.moe_combine_backward(
                grad_output.contiguous(),
                expert_outputs,
                expert_indices,
                locations,
                expert_weights,
                combine_mask,
            ),
        )
        return grad_expert_outputs, None, None, grad_expert_weights, None, None


def cuda_moe_dispatch(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    *,
    num_experts: int,
    capacity: int,
    extension: ModuleType,
) -> MoEDispatchResult:
    expert_inputs, locations, combine_mask, expert_counts = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        _CudaMoEDispatch.apply(
            tokens,
            expert_indices,
            expert_weights,
            num_experts,
            capacity,
            extension,
        ),
    )
    return MoEDispatchResult(
        expert_inputs=expert_inputs,
        expert_indices=expert_indices,
        expert_weights=expert_weights,
        locations=locations,
        combine_mask=combine_mask,
        expert_counts=expert_counts,
        capacity=capacity,
    )


def cuda_moe_combine(
    expert_outputs: torch.Tensor,
    dispatch_result: MoEDispatchResult,
    *,
    extension: ModuleType,
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        _CudaMoECombine.apply(
            expert_outputs,
            dispatch_result.expert_indices,
            dispatch_result.locations,
            dispatch_result.expert_weights,
            dispatch_result.combine_mask,
            extension,
        ),
    )

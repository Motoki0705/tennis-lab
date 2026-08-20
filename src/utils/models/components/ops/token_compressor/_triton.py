"""Triton autograd implementation of fused post-projection compressor pooling."""

from __future__ import annotations

from typing import Any, cast

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

_COMPRESSION_RATIO = 4
_HEAD_DIM = 64
_BLOCK_DIM = 64
_SUPPORTED_RAW_DTYPES = {torch.bfloat16, torch.float32}


@triton.jit  # type: ignore[untyped-decorator]
def _token_compressor_forward_kernel(  # type: ignore[no-untyped-def]
    raw_kv,
    raw_gate,
    state_valid,
    pooled,
    pooled_valid,
    sequence_length,
    compressed_length,
    RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // compressed_length
    compressed_index = row - batch * compressed_length
    channels = tl.arange(0, BLOCK_DIM)
    channel_valid = channels < HEAD_DIM

    maximum = tl.full((BLOCK_DIM,), -float("inf"), tl.float32)
    any_valid = False
    for source in range(2 * RATIO):
        token = (compressed_index - 1) * RATIO + source
        maximum_branch: tl.constexpr = source // RATIO
        boundary_valid = (token >= 0) & (token < sequence_length)
        safe_token = tl.maximum(0, tl.minimum(token, sequence_length - 1))
        source_valid = boundary_valid & tl.load(
            state_valid + batch * sequence_length + safe_token
        )
        offset = (
            (batch * sequence_length + safe_token) * 2 + maximum_branch
        ) * HEAD_DIM + channels
        gate = tl.load(
            raw_gate + offset,
            mask=channel_valid,
            other=0.0,
        ).to(tl.float32)
        maximum = tl.maximum(maximum, tl.where(source_valid, gate, -float("inf")))
        any_valid = any_valid | source_valid

    maximum = tl.where(any_valid, maximum, 0.0)
    denominator = tl.zeros((BLOCK_DIM,), tl.float32)
    numerator = tl.zeros((BLOCK_DIM,), tl.float32)
    for source in range(2 * RATIO):
        token = (compressed_index - 1) * RATIO + source
        reduction_branch: tl.constexpr = source // RATIO
        boundary_valid = (token >= 0) & (token < sequence_length)
        safe_token = tl.maximum(0, tl.minimum(token, sequence_length - 1))
        source_valid = boundary_valid & tl.load(
            state_valid + batch * sequence_length + safe_token
        )
        offset = (
            (batch * sequence_length + safe_token) * 2 + reduction_branch
        ) * HEAD_DIM + channels
        gate = tl.load(
            raw_gate + offset,
            mask=channel_valid,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            raw_kv + offset,
            mask=channel_valid,
            other=0.0,
        ).to(tl.float32)
        safe_value = tl.where(source_valid, value, 0.0)
        weight = tl.where(source_valid, tl.exp(gate - maximum), 0.0)
        denominator += weight
        numerator += weight * safe_value

    safe_denominator = tl.where(any_valid, denominator, 1.0)
    result = tl.where(any_valid, numerator / safe_denominator, 0.0)
    tl.store(pooled + row * HEAD_DIM + channels, result, mask=channel_valid)
    tl.store(pooled_valid + row, any_valid)


@triton.jit  # type: ignore[untyped-decorator]
def _token_compressor_backward_kernel(  # type: ignore[no-untyped-def]
    grad_pooled,
    raw_kv,
    raw_gate,
    state_valid,
    pooled,
    grad_raw_kv,
    grad_raw_gate,
    sequence_length,
    compressed_length,
    RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // (sequence_length * 2)
    within_batch = row - batch * sequence_length * 2
    token = within_batch // 2
    branch = within_batch - token * 2
    compressed_index = token // RATIO + (1 - branch)
    channels = tl.arange(0, BLOCK_DIM)
    channel_valid = channels < HEAD_DIM
    contributes = (compressed_index < compressed_length) & tl.load(
        state_valid + batch * sequence_length + token
    )

    maximum = tl.full((BLOCK_DIM,), -float("inf"), tl.float32)
    any_source_valid = False
    for source in range(2 * RATIO):
        source_token = (compressed_index - 1) * RATIO + source
        maximum_source_branch: tl.constexpr = source // RATIO
        boundary_valid = (source_token >= 0) & (source_token < sequence_length)
        safe_token = tl.maximum(0, tl.minimum(source_token, sequence_length - 1))
        source_valid = boundary_valid & tl.load(
            state_valid + batch * sequence_length + safe_token
        )
        offset = (
            (batch * sequence_length + safe_token) * 2 + maximum_source_branch
        ) * HEAD_DIM + channels
        gate = tl.load(
            raw_gate + offset,
            mask=channel_valid,
            other=0.0,
        ).to(tl.float32)
        maximum = tl.maximum(maximum, tl.where(source_valid, gate, -float("inf")))
        any_source_valid = any_source_valid | source_valid

    maximum = tl.where(any_source_valid, maximum, 0.0)
    denominator = tl.zeros((BLOCK_DIM,), tl.float32)
    for source in range(2 * RATIO):
        source_token = (compressed_index - 1) * RATIO + source
        denominator_source_branch: tl.constexpr = source // RATIO
        boundary_valid = (source_token >= 0) & (source_token < sequence_length)
        safe_token = tl.maximum(0, tl.minimum(source_token, sequence_length - 1))
        source_valid = boundary_valid & tl.load(
            state_valid + batch * sequence_length + safe_token
        )
        offset = (
            (batch * sequence_length + safe_token) * 2 + denominator_source_branch
        ) * HEAD_DIM + channels
        gate = tl.load(
            raw_gate + offset,
            mask=channel_valid,
            other=0.0,
        ).to(tl.float32)
        denominator += tl.where(source_valid, tl.exp(gate - maximum), 0.0)

    raw_offset = ((batch * sequence_length + token) * 2 + branch) * HEAD_DIM + channels
    gate = tl.load(
        raw_gate + raw_offset,
        mask=channel_valid,
        other=0.0,
    ).to(tl.float32)
    value = tl.load(
        raw_kv + raw_offset,
        mask=channel_valid,
        other=0.0,
    ).to(tl.float32)
    safe_compressed_index = tl.minimum(compressed_index, compressed_length - 1)
    pooled_offset = (
        batch * compressed_length + safe_compressed_index
    ) * HEAD_DIM + channels
    pooled_value = tl.load(
        pooled + pooled_offset,
        mask=channel_valid,
        other=0.0,
    ).to(tl.float32)
    upstream = tl.load(
        grad_pooled + pooled_offset,
        mask=channel_valid,
        other=0.0,
    ).to(tl.float32)
    safe_denominator = tl.where(contributes, denominator, 1.0)
    safe_gate = tl.where(contributes, gate, 0.0)
    weight = tl.where(
        contributes,
        tl.exp(safe_gate - maximum) / safe_denominator,
        0.0,
    )
    safe_value = tl.where(contributes, value, 0.0)
    safe_pooled = tl.where(contributes, pooled_value, 0.0)
    grad_value = tl.where(contributes, upstream * weight, 0.0)
    grad_gate = tl.where(
        contributes,
        upstream * weight * (safe_value - safe_pooled),
        0.0,
    )
    tl.store(
        grad_raw_kv + raw_offset,
        grad_value,
        mask=channel_valid,
    )
    tl.store(
        grad_raw_gate + raw_offset,
        grad_gate,
        mask=channel_valid,
    )


def _validate_inputs(
    raw_kv: Tensor,
    raw_gate: Tensor,
    state_valid: Tensor,
    *,
    compression_ratio: int,
) -> tuple[int, int]:
    if raw_kv.ndim != 4:
        raise ValueError(
            f"raw_kv must have shape [N, T, 2, 64], got {tuple(raw_kv.shape)}"
        )
    if raw_gate.shape != raw_kv.shape:
        raise ValueError("raw_gate shape must equal raw_kv shape")
    n, sequence_length, branches, head_dim = raw_kv.shape
    if n <= 0 or sequence_length <= 0:
        raise ValueError("raw_kv batch and sequence dimensions must be positive")
    if branches != 2 or head_dim != _HEAD_DIM:
        raise ValueError(
            "token-compressor CUDA requires raw shape [N, T, 2, 64], "
            f"got {tuple(raw_kv.shape)}"
        )
    if raw_kv.dtype not in _SUPPORTED_RAW_DTYPES:
        raise TypeError(
            "token-compressor CUDA supports bfloat16 and float32 raw_kv, "
            f"got {raw_kv.dtype}"
        )
    if raw_gate.dtype != torch.float32:
        raise TypeError(
            f"token-compressor CUDA requires float32 raw_gate, got {raw_gate.dtype}"
        )
    if state_valid.shape != (n, sequence_length):
        raise ValueError(
            f"state_valid must have shape {(n, sequence_length)}, "
            f"got {tuple(state_valid.shape)}"
        )
    if state_valid.dtype != torch.bool:
        raise TypeError(f"state_valid must have dtype bool, got {state_valid.dtype}")
    for name, tensor in (
        ("raw_kv", raw_kv),
        ("raw_gate", raw_gate),
        ("state_valid", state_valid),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.device != raw_kv.device:
            raise ValueError(
                f"{name} must be on device {raw_kv.device}, got {tensor.device}"
            )
    if type(compression_ratio) is not int or compression_ratio != _COMPRESSION_RATIO:
        raise ValueError(
            "token-compressor CUDA supports compression_ratio=4, "
            f"got {compression_ratio!r}"
        )
    return n, sequence_length


class _TritonTokenCompressorPool(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        raw_kv: Tensor,
        raw_gate: Tensor,
        state_valid: Tensor,
        compression_ratio: int,
    ) -> tuple[Tensor, Tensor]:
        contiguous_raw_kv = raw_kv.contiguous()
        contiguous_raw_gate = raw_gate.contiguous()
        contiguous_state_valid = state_valid.contiguous()
        n, sequence_length = contiguous_state_valid.shape
        compressed_length = (
            sequence_length + compression_ratio - 1
        ) // compression_ratio
        pooled = torch.empty(
            n,
            compressed_length,
            _HEAD_DIM,
            device=raw_kv.device,
            dtype=torch.float32,
        )
        pooled_valid = torch.empty(
            n,
            compressed_length,
            device=raw_kv.device,
            dtype=torch.bool,
        )
        _token_compressor_forward_kernel[(n * compressed_length,)](
            contiguous_raw_kv,
            contiguous_raw_gate,
            contiguous_state_valid,
            pooled,
            pooled_valid,
            sequence_length,
            compressed_length,
            RATIO=_COMPRESSION_RATIO,
            HEAD_DIM=_HEAD_DIM,
            BLOCK_DIM=_BLOCK_DIM,
            num_warps=4,
        )
        ctx.save_for_backward(
            contiguous_raw_kv,
            contiguous_raw_gate,
            contiguous_state_valid,
            pooled,
        )
        ctx.mark_non_differentiable(pooled_valid)
        ctx.set_materialize_grads(False)
        return pooled, pooled_valid

    @staticmethod
    def backward(
        ctx: Any,
        grad_pooled: Tensor,
        _grad_pooled_valid: Tensor | None,
    ) -> tuple[Tensor, Tensor, None, None]:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "token-compressor CUDA does not support higher-order gradients"
            )
        raw_kv, raw_gate, state_valid, pooled = ctx.saved_tensors
        n, sequence_length, _, _ = raw_kv.shape
        compressed_length = pooled.shape[1]
        grad_raw_kv = torch.empty_like(raw_kv)
        grad_raw_gate = torch.empty_like(raw_gate, dtype=torch.float32)
        _token_compressor_backward_kernel[(n * sequence_length * 2,)](
            grad_pooled.contiguous(),
            raw_kv,
            raw_gate,
            state_valid,
            pooled,
            grad_raw_kv,
            grad_raw_gate,
            sequence_length,
            compressed_length,
            RATIO=_COMPRESSION_RATIO,
            HEAD_DIM=_HEAD_DIM,
            BLOCK_DIM=_BLOCK_DIM,
            num_warps=4,
        )
        return grad_raw_kv, grad_raw_gate, None, None


def cuda_token_compressor_pool(
    raw_kv: Tensor,
    raw_gate: Tensor,
    state_valid: Tensor,
    *,
    compression_ratio: int,
) -> tuple[Tensor, Tensor]:
    """Run fused previous/current masked pooling on its explicit CUDA contract."""
    _validate_inputs(
        raw_kv,
        raw_gate,
        state_valid,
        compression_ratio=compression_ratio,
    )
    result = _TritonTokenCompressorPool.apply(
        raw_kv,
        raw_gate,
        state_valid,
        compression_ratio,
    )
    return cast(tuple[Tensor, Tensor], result)


__all__ = ["cuda_token_compressor_pool"]

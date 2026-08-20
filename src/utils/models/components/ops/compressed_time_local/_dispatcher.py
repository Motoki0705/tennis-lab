"""Opaque dispatcher boundary for compressed time-local CUDA kernels."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from src.utils.models.components.ops import loader

_FORWARD_QUALNAME = "tennis_lab::compressed_time_local_forward"
_BACKWARD_QUALNAME = "tennis_lab::compressed_time_local_backward"
_HIGHER_ORDER_MESSAGE = (
    "compressed time-local CUDA attention does not support higher-order gradients"
)
_FORWARD_ARGUMENTS = (
    "query",
    "key",
    "value",
    "query_valid",
    "key_valid",
    "query_phasors_real",
    "key_phasors_real",
    "compression_ratio",
    "window_radius",
)
_BACKWARD_ARGUMENTS = (
    "grad_output",
    "query",
    "key",
    "value",
    "query_valid",
    "key_valid",
    "logsumexp",
    "query_phasors_real",
    "key_phasors_real",
    "compression_ratio",
    "window_radius",
)


def _empty_compact_nhtd_like(tensor: Tensor) -> Tensor:
    n, heads, sequence_length, head_dim = tensor.shape
    return tensor.new_empty((n, sequence_length, heads, head_dim)).transpose(1, 2)


def _validate_fake_phasor_pair(
    query_phasors_real: Tensor | None,
    key_phasors_real: Tensor | None,
) -> None:
    if (query_phasors_real is None) != (key_phasors_real is None):
        raise RuntimeError(
            "query_phasors_real and key_phasors_real must both be present or absent"
        )


def _cuda_forward_impl(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_valid: Tensor,
    key_valid: Tensor,
    query_phasors_real: Tensor | None,
    key_phasors_real: Tensor | None,
    compression_ratio: int,
    window_radius: int,
) -> tuple[Tensor, Tensor, Tensor]:
    extension = loader.require_compressed_time_local_cuda_extension()
    raw_result = cast(
        list[Tensor],
        extension.forward(
            query,
            key,
            value,
            query_valid,
            key_valid,
            query_phasors_real,
            key_phasors_real,
            compression_ratio,
            window_radius,
        ),
    )
    output, logsumexp, invalid_row = raw_result
    return output, logsumexp, invalid_row


def _cuda_backward_impl(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_valid: Tensor,
    key_valid: Tensor,
    logsumexp: Tensor,
    query_phasors_real: Tensor | None,
    key_phasors_real: Tensor | None,
    compression_ratio: int,
    window_radius: int,
) -> tuple[Tensor, Tensor, Tensor]:
    extension = loader.require_compressed_time_local_cuda_extension()
    raw_result = cast(
        list[Tensor],
        extension.backward(
            grad_output,
            query,
            key,
            value,
            query_valid,
            key_valid,
            logsumexp,
            query_phasors_real,
            key_phasors_real,
            compression_ratio,
            window_radius,
        ),
    )
    grad_query, grad_key, grad_value = raw_result
    return grad_query, grad_key, grad_value


def _fake_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_valid: Tensor,
    key_valid: Tensor,
    query_phasors_real: Tensor | None,
    key_phasors_real: Tensor | None,
    compression_ratio: int,
    window_radius: int,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_fake_phasor_pair(query_phasors_real, key_phasors_real)
    del (
        key,
        value,
        query_valid,
        key_valid,
        query_phasors_real,
        key_phasors_real,
        compression_ratio,
        window_radius,
    )
    output = _empty_compact_nhtd_like(query)
    logsumexp = query.new_empty(query.shape[:3], dtype=torch.float32)
    invalid_row = query.new_empty((1,), dtype=torch.int32)
    return output, logsumexp, invalid_row


def _fake_backward(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_valid: Tensor,
    key_valid: Tensor,
    logsumexp: Tensor,
    query_phasors_real: Tensor | None,
    key_phasors_real: Tensor | None,
    compression_ratio: int,
    window_radius: int,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_fake_phasor_pair(query_phasors_real, key_phasors_real)
    del (
        grad_output,
        query_valid,
        key_valid,
        logsumexp,
        query_phasors_real,
        key_phasors_real,
        compression_ratio,
        window_radius,
    )
    return (
        _empty_compact_nhtd_like(query),
        torch.empty_like(key, memory_format=torch.contiguous_format),
        torch.empty_like(value, memory_format=torch.contiguous_format),
    )


def _setup_forward_context(
    ctx: Any,
    inputs: tuple[object, ...],
    output: tuple[Tensor, Tensor, Tensor],
) -> None:
    (
        query,
        key,
        value,
        query_valid,
        key_valid,
        query_phasors_real,
        key_phasors_real,
        compression_ratio,
        window_radius,
    ) = inputs
    _attention_output, logsumexp, invalid_row = output
    tensors_to_save = (
        query,
        key,
        value,
        query_valid,
        key_valid,
        logsumexp,
    )
    if query_phasors_real is not None and key_phasors_real is not None:
        ctx.save_for_backward(
            *tensors_to_save,
            query_phasors_real,
            key_phasors_real,
        )
        ctx.has_rope = True
    elif query_phasors_real is None and key_phasors_real is None:
        ctx.save_for_backward(*tensors_to_save)
        ctx.has_rope = False
    else:
        raise RuntimeError(
            "query_phasors_real and key_phasors_real must both be present or absent"
        )
    ctx.mark_non_differentiable(logsumexp, invalid_row)
    ctx.compression_ratio = compression_ratio
    ctx.window_radius = window_radius


def _forward_autograd_backward(
    ctx: Any,
    grad_output: Tensor,
    _grad_logsumexp: Tensor | None,
    _grad_invalid_row: Tensor | None,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    None,
    None,
    None,
    None,
    None,
    None,
]:
    query, key, value, query_valid, key_valid, logsumexp = ctx.saved_tensors[:6]
    if ctx.has_rope:
        query_phasors_real, key_phasors_real = ctx.saved_tensors[-2:]
    else:
        query_phasors_real = None
        key_phasors_real = None
    grad_query, grad_key, grad_value = compressed_time_local_backward(
        grad_output,
        query,
        key,
        value,
        query_valid,
        key_valid,
        logsumexp,
        query_phasors_real,
        key_phasors_real,
        ctx.compression_ratio,
        ctx.window_radius,
    )
    return grad_query, grad_key, grad_value, None, None, None, None, None, None


def _reject_higher_order_gradients(
    ctx: Any,
    grad_grad_query: Tensor | None,
    grad_grad_key: Tensor | None,
    grad_grad_value: Tensor | None,
) -> None:
    del ctx, grad_grad_query, grad_grad_key, grad_grad_value
    raise RuntimeError(_HIGHER_ORDER_MESSAGE)


def _lookup_registered_op(qualname: str) -> Any:
    return torch.library._maybe_get_opdef(qualname)


def _validate_registered_schema(
    op: Any,
    *,
    qualname: str,
    expected_arguments: tuple[str, ...],
) -> None:
    overload = getattr(op, "_opoverload", None)
    schema = getattr(overload, "_schema", None)
    argument_names = (
        () if schema is None else tuple(argument.name for argument in schema.arguments)
    )
    return_types = (
        () if schema is None else tuple(str(result.type) for result in schema.returns)
    )
    optional_arguments = (
        ()
        if schema is None
        else tuple(
            argument.name
            for argument in schema.arguments
            if str(argument.type) == "Optional[Tensor]"
        )
    )
    if (
        argument_names != expected_arguments
        or return_types != ("Tensor", "Tensor", "Tensor")
        or optional_arguments != ("query_phasors_real", "key_phasors_real")
    ):
        raise RuntimeError(
            f"stale compressed time-local dispatcher schema for {qualname}: {schema}"
        )


def _register_ops() -> tuple[Any, Any]:
    existing_forward = _lookup_registered_op(_FORWARD_QUALNAME)
    existing_backward = _lookup_registered_op(_BACKWARD_QUALNAME)
    if (existing_forward is None) != (existing_backward is None):
        raise RuntimeError(
            "compressed time-local dispatcher registration is incomplete"
        )
    if existing_forward is not None and existing_backward is not None:
        _validate_registered_schema(
            existing_forward,
            qualname=_FORWARD_QUALNAME,
            expected_arguments=_FORWARD_ARGUMENTS,
        )
        _validate_registered_schema(
            existing_backward,
            qualname=_BACKWARD_QUALNAME,
            expected_arguments=_BACKWARD_ARGUMENTS,
        )
        return existing_forward, existing_backward

    forward_op = torch.library.custom_op(
        _FORWARD_QUALNAME,
        _cuda_forward_impl,
        mutates_args=(),
        device_types="cuda",
    )
    backward_op = torch.library.custom_op(
        _BACKWARD_QUALNAME,
        _cuda_backward_impl,
        mutates_args=(),
        device_types="cuda",
    )
    forward_op.register_fake(_fake_forward)
    backward_op.register_fake(_fake_backward)
    backward_op.register_autograd(_reject_higher_order_gradients)
    forward_op.register_autograd(
        _forward_autograd_backward,
        setup_context=_setup_forward_context,
    )
    _validate_registered_schema(
        forward_op,
        qualname=_FORWARD_QUALNAME,
        expected_arguments=_FORWARD_ARGUMENTS,
    )
    _validate_registered_schema(
        backward_op,
        qualname=_BACKWARD_QUALNAME,
        expected_arguments=_BACKWARD_ARGUMENTS,
    )
    return forward_op, backward_op


compressed_time_local_forward, compressed_time_local_backward = _register_ops()

__all__ = ["compressed_time_local_backward", "compressed_time_local_forward"]

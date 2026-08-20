"""CPU-safe dispatcher and compiler contracts for the optional CUDA op."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensorMode

from src.utils.models.components.ops.compressed_time_local import _dispatcher
from src.utils.models.components.ops.compressed_time_local._autograd import (
    cuda_compressed_time_local_attention,
)


def _is_alias_of(left: Tensor, right: Tensor) -> bool:
    """Call PyTorch's runtime alias probe, which is absent from its type stubs."""
    return bool(torch._C._is_alias_of(left, right))  # type: ignore[attr-defined]


def _meta_inputs(
    dtype: torch.dtype,
    *,
    key_heads: int,
    transpose_query: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if transpose_query:
        query = torch.empty((2, 9, 3, 16), device="meta", dtype=dtype).transpose(1, 2)
    else:
        query = torch.empty((2, 3, 9, 16), device="meta", dtype=dtype)
    key = torch.empty((2, key_heads, 3, 16), device="meta", dtype=dtype)
    value = torch.empty_like(key)
    query_valid = torch.empty((2, 9), device="meta", dtype=torch.bool)
    key_valid = torch.empty((2, 3), device="meta", dtype=torch.bool)
    return query, key, value, query_valid, key_valid


def test_dispatcher_registration_is_idempotent_on_module_reload() -> None:
    original_forward = _dispatcher.compressed_time_local_forward
    original_backward = _dispatcher.compressed_time_local_backward

    reloaded = importlib.reload(_dispatcher)

    assert reloaded.compressed_time_local_forward is original_forward
    assert reloaded.compressed_time_local_backward is original_backward


def test_dispatcher_registration_rejects_a_stale_complete_schema() -> None:
    with pytest.raises(RuntimeError, match="stale compressed time-local"):
        _dispatcher._validate_registered_schema(
            _dispatcher.compressed_time_local_backward,
            qualname="tennis_lab::compressed_time_local_forward",
            expected_arguments=_dispatcher._FORWARD_ARGUMENTS,
        )


def test_dispatcher_fake_kernel_rejects_incomplete_phasor_pair() -> None:
    query, key, value, query_valid, key_valid = _meta_inputs(torch.float32, key_heads=1)

    with pytest.raises(RuntimeError, match="both be present or absent"):
        _dispatcher.compressed_time_local_forward(
            query,
            key,
            value,
            query_valid,
            key_valid,
            torch.empty((1, 9, 1, 8, 2), device="meta"),
            None,
            4,
            2,
        )


def test_dispatcher_schemas_have_fixed_tensor_result_arity() -> None:
    forward_schema = torch.ops.tennis_lab.compressed_time_local_forward.default._schema
    backward_schema = (
        torch.ops.tennis_lab.compressed_time_local_backward.default._schema
    )

    assert len(forward_schema.returns) == 3
    assert len(backward_schema.returns) == 3
    assert len(forward_schema.arguments) == 9
    assert len(backward_schema.arguments) == 11
    assert tuple(argument.name for argument in forward_schema.arguments) == (
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
    assert tuple(argument.name for argument in backward_schema.arguments) == (
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
    assert str(forward_schema.arguments[5].type) == "Optional[Tensor]"
    assert str(forward_schema.arguments[6].type) == "Optional[Tensor]"
    assert all(str(result.type) == "Tensor" for result in forward_schema.returns)
    assert all(str(result.type) == "Tensor" for result in backward_schema.returns)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("key_heads", [1, 3])
@pytest.mark.parametrize("transpose_query", [False, True])
@pytest.mark.parametrize("with_rope", [False, True])
def test_dispatcher_fake_forward_and_backward_shapes_and_dtypes(
    dtype: torch.dtype,
    key_heads: int,
    transpose_query: bool,
    with_rope: bool,
) -> None:
    query, key, value, query_valid, key_valid = _meta_inputs(
        dtype,
        key_heads=key_heads,
        transpose_query=transpose_query,
    )
    query_phasors = torch.empty((1, 9, 1, 8, 2), device="meta") if with_rope else None
    key_phasors = torch.empty((2, 3, 1, 8, 2), device="meta") if with_rope else None
    output, logsumexp, invalid_row = _dispatcher.compressed_time_local_forward(
        query,
        key,
        value,
        query_valid,
        key_valid,
        query_phasors,
        key_phasors,
        4,
        2,
    )
    grad_query, grad_key, grad_value = _dispatcher.compressed_time_local_backward(
        torch.empty_like(output),
        query,
        key,
        value,
        query_valid,
        key_valid,
        logsumexp,
        query_phasors,
        key_phasors,
        4,
        2,
    )

    assert output.shape == query.shape
    assert output.dtype == dtype
    assert output.stride() == (9 * 3 * 16, 16, 3 * 16, 1)
    assert not _is_alias_of(output, query)
    assert logsumexp.shape == query.shape[:3]
    assert logsumexp.dtype == torch.float32
    assert invalid_row.shape == (1,)
    assert invalid_row.dtype == torch.int32
    assert grad_query.shape == query.shape
    assert grad_query.stride() == output.stride()
    assert not _is_alias_of(grad_query, query)
    assert grad_key.shape == key.shape
    assert not _is_alias_of(grad_key, key)
    assert grad_value.shape == value.shape
    assert not _is_alias_of(grad_value, value)
    assert (grad_query.dtype, grad_key.dtype, grad_value.dtype) == (
        dtype,
        dtype,
        dtype,
    )
    if transpose_query:
        round_trip = output.transpose(1, 2)
        flattened = round_trip.reshape(2, 9, 3 * 16)
        assert round_trip.is_contiguous()
        assert (
            flattened.untyped_storage().data_ptr()
            == output.untyped_storage().data_ptr()
        )


def test_forward_autograd_saves_inputs_and_lse_but_not_attention_output() -> None:
    query, key, value, query_valid, key_valid = _meta_inputs(torch.float32, key_heads=1)
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)
    saved_tensors: list[Tensor] = []

    def record_saved_tensor(tensor: Tensor) -> Tensor:
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record_saved_tensor, lambda x: x):
        attention_output, logsumexp, _invalid_row = (
            _dispatcher.compressed_time_local_forward(
                query,
                key,
                value,
                query_valid,
                key_valid,
                None,
                None,
                4,
                2,
            )
        )

    expected_saved = (query, key, value, query_valid, key_valid, logsumexp)
    assert len(saved_tensors) == len(expected_saved)
    assert all(
        actual is expected
        for actual, expected in zip(saved_tensors, expected_saved, strict=True)
    )
    assert not any(
        _is_alias_of(saved_tensor, attention_output) for saved_tensor in saved_tensors
    )


def test_fused_rope_autograd_saves_read_only_views_but_not_attention_output() -> None:
    query, key, value, query_valid, key_valid = _meta_inputs(torch.float32, key_heads=1)
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)
    query_phasors = torch.empty((1, 9, 1, 8, 2), device="meta")
    key_phasors = torch.empty((2, 3, 1, 8, 2), device="meta")
    saved_tensors: list[Tensor] = []

    def record_saved_tensor(tensor: Tensor) -> Tensor:
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record_saved_tensor, lambda x: x):
        attention_output, logsumexp, _invalid_row = (
            _dispatcher.compressed_time_local_forward(
                query,
                key,
                value,
                query_valid,
                key_valid,
                query_phasors,
                key_phasors,
                4,
                2,
            )
        )

    expected_saved = (
        query,
        key,
        value,
        query_valid,
        key_valid,
        logsumexp,
        query_phasors,
        key_phasors,
    )
    assert len(saved_tensors) == len(expected_saved)
    assert all(
        actual is expected
        for actual, expected in zip(saved_tensors, expected_saved, strict=True)
    )
    assert not any(
        _is_alias_of(saved_tensor, attention_output) for saved_tensor in saved_tensors
    )


def test_dispatcher_explicitly_rejects_higher_order_gradients() -> None:
    query, key, value, query_valid, key_valid = _meta_inputs(torch.float32, key_heads=1)
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)
    output, _logsumexp, _invalid_row = _dispatcher.compressed_time_local_forward(
        query,
        key,
        value,
        query_valid,
        key_valid,
        None,
        None,
        4,
        2,
    )
    grad_query = torch.autograd.grad(
        output,
        query,
        torch.ones_like(output),
        create_graph=True,
    )[0]

    with pytest.raises(RuntimeError, match="does not support higher-order gradients"):
        torch.autograd.grad(grad_query.sum(), query)


def test_fake_cuda_public_boundary_is_one_compilable_dispatcher_graph() -> None:
    captured_graphs: list[torch.fx.GraphModule] = []

    def attention_and_projection_input(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_valid: Tensor,
        key_valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        output = cuda_compressed_time_local_attention(
            query,
            key,
            value,
            query_valid=query_valid,
            key_valid=key_valid,
            compression_ratio=4,
            window_radius=2,
        )
        n, heads, query_length, head_dim = output.shape
        projection_input = output.transpose(1, 2).reshape(
            n, query_length, heads * head_dim
        )
        return output, projection_input

    def capture_backend(
        graph_module: torch.fx.GraphModule,
        _example_inputs: list[Tensor],
    ) -> Callable[..., Any]:
        captured_graphs.append(graph_module)

        def run_graph(*args: Any, **kwargs: Any) -> Any:
            return graph_module(*args, **kwargs)

        return run_graph

    compiled = torch.compile(
        attention_and_projection_input,
        backend=capture_backend,
        fullgraph=True,
    )
    with FakeTensorMode():
        query = torch.empty(
            (2, 9, 3, 16), device="cuda", dtype=torch.bfloat16
        ).transpose(1, 2)
        key = torch.empty((2, 1, 3, 16), device="cuda", dtype=torch.bfloat16)
        value = torch.empty_like(key)
        query_valid = torch.ones((2, 9), device="cuda", dtype=torch.bool)
        key_valid = torch.ones((2, 3), device="cuda", dtype=torch.bool)
        output, projection_input = compiled(query, key, value, query_valid, key_valid)

    assert output.shape == query.shape
    assert output.stride() == query.stride()
    assert output.transpose(1, 2).is_contiguous()
    assert not _is_alias_of(output, query)
    assert projection_input.shape == (2, 9, 3 * 16)
    assert _is_alias_of(projection_input, output)
    assert len(captured_graphs) == 1
    call_targets = {
        node.target
        for node in captured_graphs[0].graph.nodes
        if node.op == "call_function"
    }
    assert torch.ops.tennis_lab.compressed_time_local_forward.default in call_targets
    assert torch.ops.aten.clone.default not in call_targets


def test_fake_cuda_aot_autograd_compiles_first_order_backward() -> None:
    def loss(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_valid: Tensor,
        key_valid: Tensor,
        upstream: Tensor,
    ) -> Tensor:
        output = cuda_compressed_time_local_attention(
            query,
            key,
            value,
            query_valid=query_valid,
            key_valid=key_valid,
            compression_ratio=4,
            window_radius=2,
        )
        return (output * upstream).sum()

    compiled = torch.compile(loss, backend="aot_eager", fullgraph=True)
    with FakeTensorMode():
        query = torch.empty(
            (2, 9, 3, 16),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        ).transpose(1, 2)
        key = torch.empty(
            (2, 1, 3, 16),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.empty_like(key, requires_grad=True)
        query_valid = torch.ones((2, 9), device="cuda", dtype=torch.bool)
        key_valid = torch.ones((2, 3), device="cuda", dtype=torch.bool)
        upstream = torch.empty_like(query)
        grad_query, grad_key, grad_value = torch.autograd.grad(
            compiled(query, key, value, query_valid, key_valid, upstream),
            (query, key, value),
        )

    assert grad_query.shape == query.shape
    assert grad_query.stride() == query.stride()
    assert grad_key.shape == key.shape
    assert grad_value.shape == value.shape


def test_fake_cuda_fused_rope_aot_compiles_first_order_backward() -> None:
    def loss(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_valid: Tensor,
        key_valid: Tensor,
        query_freqs_cis: Tensor,
        key_freqs_cis: Tensor,
        upstream: Tensor,
    ) -> Tensor:
        output = cuda_compressed_time_local_attention(
            query,
            key,
            value,
            query_valid=query_valid,
            key_valid=key_valid,
            compression_ratio=4,
            window_radius=2,
            query_freqs_cis=query_freqs_cis,
            key_freqs_cis=key_freqs_cis,
        )
        return (output * upstream).sum()

    compiled = torch.compile(loss, backend="aot_eager", fullgraph=True)
    with FakeTensorMode():
        packed = torch.empty(
            (2, 9, 3 * 16 + 32),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        query = packed[..., : 3 * 16].reshape(2, 9, 3, 16).transpose(1, 2)
        key = torch.empty(
            (2, 1, 3, 16),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.empty_like(key, requires_grad=True)
        query_valid = torch.ones((2, 9), device="cuda", dtype=torch.bool)
        key_valid = torch.ones((2, 3), device="cuda", dtype=torch.bool)
        query_freqs_cis = torch.empty(
            (1, 9, 1, 8), device="cuda", dtype=torch.complex64
        )
        key_freqs_cis = torch.empty((3, 1, 8), device="cuda", dtype=torch.complex64)
        upstream = torch.empty_like(query)
        grad_query, grad_key, grad_value = torch.autograd.grad(
            compiled(
                query,
                key,
                value,
                query_valid,
                key_valid,
                query_freqs_cis,
                key_freqs_cis,
                upstream,
            ),
            (query, key, value),
        )

    assert query.stride() == (9 * (3 * 16 + 32), 16, 3 * 16 + 32, 1)
    assert grad_query.shape == query.shape
    assert grad_query.stride() == (9 * 3 * 16, 16, 3 * 16, 1)
    assert grad_key.shape == key.shape
    assert grad_value.shape == value.shape

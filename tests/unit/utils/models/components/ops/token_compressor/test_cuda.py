"""CUDA tests for fused Triton token-compressor pooling."""

from __future__ import annotations

import os

import pytest
import torch
from torch import Tensor

from src.utils.models.components.compressor import (
    TokenLevelCompressorConfig,
    TokenLevelKVCompressor,
)
from src.utils.models.components.ops.token_compressor import (
    reference_token_compressor_pool,
    resolve_token_compressor_pool,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TENNIS_LAB_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="CUDA operation tests require TENNIS_LAB_RUN_CUDA_TESTS=1 and CUDA",
)


def _inputs(
    n: int,
    sequence_length: int,
    dtype: torch.dtype,
    head_dim: int = 64,
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(753 + sequence_length + n)
    raw_kv = torch.randn(
        n,
        sequence_length,
        2,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    raw_gate = (
        torch.randn(
            n,
            sequence_length,
            2,
            head_dim,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 6
    )
    mask_storage = (
        torch.rand(
            n,
            sequence_length * 2,
            device="cuda",
            generator=generator,
        )
        > 0.25
    )
    state_valid = mask_storage[:, ::2]
    state_valid[-1] = False
    assert not state_valid.is_contiguous() or sequence_length == 1
    return raw_kv, raw_gate, state_valid


@pytest.mark.parametrize("head_dim", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_matches_reference_across_supported_widths_and_dtypes(
    head_dim: int,
    dtype: torch.dtype,
) -> None:
    raw_kv, raw_gate, state_valid = _inputs(3, 17, dtype, head_dim)
    reference_kv = raw_kv.detach().clone().requires_grad_(True)
    reference_gate = raw_gate.detach().clone().requires_grad_(True)
    cuda_kv = raw_kv.detach().clone().requires_grad_(True)
    cuda_gate = raw_gate.detach().clone().requires_grad_(True)
    expected, expected_valid = reference_token_compressor_pool(
        reference_kv,
        reference_gate,
        state_valid,
        compression_ratio=4,
    )
    pool = resolve_token_compressor_pool(
        "cuda", compression_ratio=4, head_dim=head_dim
    )
    actual, actual_valid = pool(cuda_kv, cuda_gate, state_valid)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)

    atol = 4.0e-3 if dtype == torch.float16 else 7.0e-2 if dtype == torch.bfloat16 else 3.0e-5
    rtol = 4.0e-3 if dtype == torch.float16 else 3.0e-2 if dtype == torch.bfloat16 else 3.0e-4
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_valid, expected_valid)
    assert cuda_kv.grad is not None and reference_kv.grad is not None
    assert cuda_gate.grad is not None and reference_gate.grad is not None
    torch.testing.assert_close(cuda_kv.grad, reference_kv.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        cuda_gate.grad, reference_gate.grad, atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sequence_length", [1, 3, 4, 5, 1023, 1024, 1025])
def test_cuda_matches_reference_forward_backward_and_validity(
    dtype: torch.dtype,
    sequence_length: int,
) -> None:
    n = 3 if sequence_length <= 5 else 4
    raw_kv, raw_gate, state_valid = _inputs(n, sequence_length, dtype)
    reference_kv = raw_kv.detach().clone().requires_grad_(True)
    reference_gate = raw_gate.detach().clone().requires_grad_(True)
    cuda_kv = raw_kv.detach().clone().requires_grad_(True)
    cuda_gate = raw_gate.detach().clone().requires_grad_(True)
    reference, reference_valid = reference_token_compressor_pool(
        reference_kv,
        reference_gate,
        state_valid,
        compression_ratio=4,
    )
    pool = resolve_token_compressor_pool("cuda", compression_ratio=4, head_dim=64)
    actual, actual_valid = pool(cuda_kv, cuda_gate, state_valid)
    upstream_storage = torch.randn(
        n,
        64,
        reference.shape[1],
        device="cuda",
        dtype=torch.float32,
    )
    upstream = upstream_storage.transpose(1, 2)
    assert not upstream.is_contiguous() or reference.shape[1] == 1
    reference.backward(upstream)
    actual.backward(upstream)

    atol = 3.0e-5 if dtype == torch.float32 else 7.0e-2
    rtol = 3.0e-4 if dtype == torch.float32 else 3.0e-2
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_valid, reference_valid)
    assert actual.dtype == torch.float32
    assert actual_valid.dtype == torch.bool and not actual_valid.requires_grad
    assert torch.count_nonzero(actual[-1]) == 0
    assert cuda_kv.grad is not None and reference_kv.grad is not None
    assert cuda_gate.grad is not None and reference_gate.grad is not None
    assert cuda_kv.grad.dtype == dtype
    assert cuda_gate.grad.dtype == torch.float32
    torch.testing.assert_close(cuda_kv.grad, reference_kv.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        cuda_gate.grad, reference_gate.grad, atol=atol, rtol=rtol
    )


def test_cuda_is_stable_for_large_finite_logits_and_exact_for_all_invalid() -> None:
    raw_kv = torch.randn(3, 5, 2, 64, device="cuda", requires_grad=True)
    raw_gate = torch.zeros(3, 5, 2, 64, device="cuda", requires_grad=True)
    with torch.no_grad():
        raw_gate[0, 0, 1] = 1.0e30
        raw_gate[0, 1, 1] = -1.0e30
    state_valid = torch.tensor(
        [
            [True, True, False, True, True],
            [True, False, False, False, True],
            [False, False, False, False, False],
        ],
        device="cuda",
    )
    with torch.no_grad():
        raw_kv[~state_valid] = torch.nan
        raw_gate[~state_valid] = torch.nan
    reference, _ = reference_token_compressor_pool(
        raw_kv, raw_gate, state_valid, compression_ratio=4
    )
    pool = resolve_token_compressor_pool("cuda", compression_ratio=4, head_dim=64)
    actual, actual_valid = pool(raw_kv, raw_gate, state_valid)

    torch.testing.assert_close(actual, reference, atol=3.0e-5, rtol=3.0e-4)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[2]) == 0
    assert not actual_valid[2].any()
    actual.square().sum().backward()
    assert raw_kv.grad is not None and torch.isfinite(raw_kv.grad).all()
    assert raw_gate.grad is not None and torch.isfinite(raw_gate.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [16, 64])
def test_full_compressor_tied_key_value_sums_both_gradient_paths(
    dtype: torch.dtype,
    head_dim: int,
) -> None:
    dim = 4 * head_dim
    config = TokenLevelCompressorConfig(
        dim=dim,
        n_heads=4,
        head_dim=head_dim,
        compression_ratio=4,
        overlap=True,
    )
    reference_module = TokenLevelKVCompressor(config, backend="reference").cuda()
    cuda_module = TokenLevelKVCompressor(config, backend="cuda").cuda()
    cuda_module.load_state_dict(reference_module.state_dict(), strict=True)
    reference_x = torch.randn(3, 5, dim, device="cuda", dtype=dtype).requires_grad_()
    cuda_x = reference_x.detach().clone().requires_grad_()
    state_valid = torch.tensor(
        [
            [True, False, True, True, True],
            [True, True, False, True, False],
            [False, False, False, False, False],
        ],
        device="cuda",
    )
    expected = reference_module(reference_x, state_valid)
    actual = cuda_module(cuda_x, state_valid)
    assert actual.key is actual.value
    key_upstream = torch.randn_like(actual.key)
    value_upstream = torch.randn_like(actual.value)
    expected.key.backward(key_upstream + value_upstream)
    actual.key.backward(key_upstream, retain_graph=True)
    actual.value.backward(value_upstream)

    atol = 6.0e-5 if dtype == torch.float32 else 9.0e-2
    rtol = 5.0e-4 if dtype == torch.float32 else 4.0e-2
    torch.testing.assert_close(actual.key, expected.key, atol=atol, rtol=rtol)
    assert actual.key.dtype == dtype
    assert actual.key.data_ptr() == actual.value.data_ptr()
    assert cuda_x.grad is not None and reference_x.grad is not None
    torch.testing.assert_close(cuda_x.grad, reference_x.grad, atol=atol, rtol=rtol)
    for actual_parameter, reference_parameter in zip(
        cuda_module.parameters(), reference_module.parameters(), strict=True
    ):
        assert actual_parameter.grad is not None
        assert reference_parameter.grad is not None
        torch.testing.assert_close(
            actual_parameter.grad,
            reference_parameter.grad,
            atol=atol,
            rtol=rtol,
        )


def test_cuda_explicitly_rejects_higher_order_gradients() -> None:
    raw_kv, raw_gate, state_valid = _inputs(3, 5, torch.float32)
    raw_kv.requires_grad_()
    raw_gate.requires_grad_()
    pool = resolve_token_compressor_pool("cuda", compression_ratio=4, head_dim=64)
    pooled, _ = pool(raw_kv, raw_gate, state_valid)

    with pytest.raises(RuntimeError, match="higher-order gradients"):
        torch.autograd.grad(pooled.sum(), raw_kv, create_graph=True)


def test_cuda_saves_only_raw_inputs_mask_and_unrounded_pool() -> None:
    raw_kv, raw_gate, state_valid = _inputs(3, 5, torch.bfloat16)
    raw_kv.requires_grad_()
    raw_gate.requires_grad_()
    saved: list[Tensor] = []

    def pack(tensor: Tensor) -> Tensor:
        saved.append(tensor)
        return tensor

    def unpack(tensor: Tensor) -> Tensor:
        return tensor

    pool = resolve_token_compressor_pool("cuda", compression_ratio=4, head_dim=64)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        pooled, _ = pool(raw_kv, raw_gate, state_valid)

    assert [tensor.shape for tensor in saved] == [
        torch.Size([3, 5, 2, 64]),
        torch.Size([3, 5, 2, 64]),
        torch.Size([3, 5]),
        torch.Size([3, 2, 64]),
    ]
    assert [tensor.dtype for tensor in saved] == [
        torch.bfloat16,
        torch.float32,
        torch.bool,
        torch.float32,
    ]
    assert all(tensor.is_contiguous() for tensor in saved)
    pooled.sum().backward()


def test_cuda_rejects_unsupported_dtype_shape_ratio_and_device() -> None:
    from src.utils.models.components.ops.token_compressor._triton import (
        cuda_token_compressor_pool,
    )

    state_valid = torch.ones(3, 5, dtype=torch.bool, device="cuda")
    valid_kv = torch.randn(3, 5, 2, 64, device="cuda")
    valid_gate = torch.randn_like(valid_kv)
    with pytest.raises(TypeError, match="float16, bfloat16, and float32"):
        cuda_token_compressor_pool(
            valid_kv.double(), valid_gate, state_valid, compression_ratio=4
        )
    with pytest.raises(TypeError, match="float32 raw_gate"):
        cuda_token_compressor_pool(
            valid_kv, valid_gate.bfloat16(), state_valid, compression_ratio=4
        )
    with pytest.raises(ValueError, match="Dh in"):
        cuda_token_compressor_pool(
            valid_kv[..., :8], valid_gate[..., :8], state_valid, compression_ratio=4
        )
    with pytest.raises(ValueError, match="compression_ratio=4"):
        cuda_token_compressor_pool(
            valid_kv, valid_gate, state_valid, compression_ratio=3
        )
    with pytest.raises(ValueError, match="CUDA tensor"):
        cuda_token_compressor_pool(
            valid_kv.cpu(), valid_gate.cpu(), state_valid.cpu(), compression_ratio=4
        )

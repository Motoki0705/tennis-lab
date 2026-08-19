"""Contract and dense-oracle tests for compressed sliding-window attention."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

from src.utils.models.components import cswa as cswa_module
from src.utils.models.components.cswa import (
    CompressedSlidingWindowSelfAttention,
    CSWAConfig,
)
from src.utils.models.components.rope import precompute_freqs_cis


def _config(
    *,
    dim: int = 8,
    n_heads: int = 2,
    head_dim: int = 4,
    rope_dim: int = 4,
    attn_dropout: float = 0.0,
    compression_ratio: int = 3,
    window_radius: int = 1,
    backend: str = "reference",
) -> CSWAConfig:
    return CSWAConfig(
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        attn_dropout=attn_dropout,
        compression_ratio=compression_ratio,
        window_radius=window_radius,
        backend=backend,  # type: ignore[arg-type]
    )


def _dense_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_valid: Tensor,
    key_valid: Tensor,
    compression_ratio: int,
    window_radius: int,
) -> Tensor:
    """Test-only explicit ``[T,Tc]`` attention oracle."""
    query_len = query.shape[2]
    key_len = key.shape[2]
    centers = torch.div(
        torch.arange(query_len, device=query.device),
        compression_ratio,
        rounding_mode="floor",
    )
    compressed_indices = torch.arange(key_len, device=query.device)
    local = (compressed_indices[None, :] - centers[:, None]).abs() <= window_radius
    keep = query_valid[:, :, None] & key_valid[:, None, :] & local[None]

    safe_query = torch.where(
        query_valid[:, None, :, None], query, torch.zeros_like(query)
    )
    safe_key = torch.where(key_valid[:, None, :, None], key, torch.zeros_like(key))
    safe_value = torch.where(
        key_valid[:, None, :, None], value, torch.zeros_like(value)
    )
    scores = torch.einsum("nhtd,nhcd->nhtc", safe_query, safe_key) / math.sqrt(
        query.shape[-1]
    )
    safe_keep = keep.clone()
    empty = ~safe_keep.any(dim=-1)
    safe_keep[..., 0] |= empty
    probabilities = torch.softmax(
        scores.masked_fill(~safe_keep[:, None], -torch.inf), dim=-1
    )
    output = torch.einsum("nhtc,nhcd->nhtd", probabilities, safe_value)
    return torch.where(query_valid[:, None, :, None], output, torch.zeros_like(output))


def _module_dense_oracle(
    module: CompressedSlidingWindowSelfAttention,
    x: Tensor,
    freqs_cis: Tensor,
    state_valid: Tensor,
) -> Tensor:
    n, query_len, _ = x.shape
    masked_x = torch.where(state_valid.unsqueeze(-1), x, torch.zeros_like(x))
    query = module._project(masked_x, module.wq).reshape(
        n, query_len, module.n_heads, module.head_dim
    )
    query = torch.cat(
        (
            module._apply_rope(query[..., : module.rope_dim], freqs_cis),
            query[..., module.rope_dim :],
        ),
        dim=-1,
    ).transpose(1, 2)

    compressed = module.compressor(masked_x, state_valid)
    key = compressed.key.transpose(1, 2)
    key_positions = compressed.positions.to(
        dtype=module.compressed_frequency_computer.inverse_frequencies.dtype
    ).unsqueeze(-1)
    key_freqs_cis = module.compressed_frequency_computer(key_positions)
    key = torch.cat(
        (
            module._apply_rope(key[..., : module.rope_dim], key_freqs_cis),
            key[..., module.rope_dim :],
        ),
        dim=-1,
    ).transpose(1, 2)
    output = _dense_attention(
        query,
        key,
        compressed.value,
        query_valid=state_valid,
        key_valid=compressed.state_valid,
        compression_ratio=module.compression_ratio,
        window_radius=module.window_radius,
    )
    output = output.transpose(1, 2).reshape(n, query_len, module.dim)
    output = module._project(output, module.wo)
    return torch.where(state_valid.unsqueeze(-1), output, torch.zeros_like(output))


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"dim": 0}, ValueError, "dim must be positive"),
        ({"n_heads": 0}, ValueError, "n_heads must be positive"),
        ({"head_dim": 0}, ValueError, "head_dim must be positive"),
        ({"dim": 7}, ValueError, r"dim must equal n_heads \* head_dim"),
        ({"rope_dim": 0}, ValueError, "positive and even"),
        ({"rope_dim": 3}, ValueError, "positive and even"),
        ({"rope_dim": 6}, ValueError, "cannot exceed head_dim"),
        ({"attn_dropout": -0.1}, ValueError, r"in \[0, 1\)"),
        ({"attn_dropout": 1.0}, ValueError, r"in \[0, 1\)"),
        ({"attn_dropout": True}, TypeError, "real number"),
        ({"compression_ratio": 1}, ValueError, "at least 2"),
        ({"window_radius": -1}, ValueError, "non-negative"),
        ({"backend": "automatic"}, ValueError, "Unsupported CSWA backend"),
        ({"dim": True}, TypeError, "dim must be an integer"),
    ],
)
def test_config_rejects_invalid_values(
    overrides: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "dim": 8,
        "n_heads": 2,
        "head_dim": 4,
        "rope_dim": 4,
        "attn_dropout": 0.0,
        "compression_ratio": 3,
        "window_radius": 1,
        "backend": "reference",
    }
    kwargs.update(overrides)
    with pytest.raises(error_type, match=message):
        CSWAConfig(**kwargs)  # type: ignore[arg-type]


def test_constructor_requires_config_and_resolves_cuda_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="config must be CSWAConfig"):
        CompressedSlidingWindowSelfAttention(object())  # type: ignore[arg-type]

    def unavailable(
        backend: str, *, compression_ratio: int, window_radius: int
    ) -> Callable[..., Tensor]:
        del backend, compression_ratio, window_radius
        raise RuntimeError("CUDA extension unavailable at construction")

    monkeypatch.setattr(
        cswa_module, "resolve_compressed_time_local_attention", unavailable
    )
    with pytest.raises(RuntimeError, match="at construction"):
        CompressedSlidingWindowSelfAttention(_config(backend="cuda"))


@pytest.mark.parametrize(
    ("make_x", "make_freqs", "make_mask", "error_type", "message"),
    [
        (
            lambda: torch.randn(2, 8),
            lambda: precompute_freqs_cis(dim=4, seqlen=2),
            lambda: torch.ones(2, 1, dtype=torch.bool),
            ValueError,
            "x must have shape",
        ),
        (
            lambda: torch.randn(2, 3, 7),
            lambda: precompute_freqs_cis(dim=4, seqlen=3),
            lambda: torch.ones(2, 3, dtype=torch.bool),
            ValueError,
            "feature dimension",
        ),
        (
            lambda: torch.randn(2, 3, 8),
            lambda: precompute_freqs_cis(dim=4, seqlen=3),
            lambda: torch.ones(2, 2, dtype=torch.bool),
            ValueError,
            "state_valid must have shape",
        ),
        (
            lambda: torch.randn(2, 3, 8),
            lambda: precompute_freqs_cis(dim=4, seqlen=3),
            lambda: torch.ones(2, 3),
            TypeError,
            "state_valid must have dtype bool",
        ),
        (
            lambda: torch.randn(2, 3, 8),
            lambda: torch.ones(3, 1, 2),
            lambda: torch.ones(2, 3, dtype=torch.bool),
            TypeError,
            "freqs_cis must be complex",
        ),
        (
            lambda: torch.randn(2, 3, 8),
            lambda: precompute_freqs_cis(dim=4, seqlen=2),
            lambda: torch.ones(2, 3, dtype=torch.bool),
            ValueError,
            "query length must be 3",
        ),
    ],
)
def test_forward_rejects_invalid_runtime_contracts(
    make_x: Callable[[], Tensor],
    make_freqs: Callable[[], Tensor],
    make_mask: Callable[[], Tensor],
    error_type: type[Exception],
    message: str,
) -> None:
    module = CompressedSlidingWindowSelfAttention(_config())
    with pytest.raises(error_type, match=message):
        module(make_x(), freqs_cis=make_freqs(), state_valid=make_mask())


def test_forward_matches_dense_oracle_and_parameter_backward() -> None:
    torch.manual_seed(41)
    actual_module = CompressedSlidingWindowSelfAttention(_config()).double()
    oracle_module = CompressedSlidingWindowSelfAttention(_config()).double()
    oracle_module.load_state_dict(actual_module.state_dict(), strict=True)
    actual_x = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
    oracle_x = actual_x.detach().clone().requires_grad_()
    state_valid = torch.tensor(
        [
            [True, True, False, True, True, False, True],
            [True, False, True, True, False, True, True],
        ]
    )
    freqs_cis = precompute_freqs_cis(dim=4, seqlen=7)
    upstream = torch.randn_like(actual_x)

    actual = actual_module(actual_x, freqs_cis=freqs_cis, state_valid=state_valid)
    expected = _module_dense_oracle(oracle_module, oracle_x, freqs_cis, state_valid)
    actual_parameters = tuple(actual_module.parameters())
    oracle_parameters = tuple(oracle_module.parameters())
    actual_gradients = torch.autograd.grad(
        (actual * upstream).sum(), (actual_x, *actual_parameters)
    )
    expected_gradients = torch.autograd.grad(
        (expected * upstream).sum(), (oracle_x, *oracle_parameters)
    )

    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-9)
    assert len(actual_gradients) == len(expected_gradients)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=1e-10, rtol=1e-8
        )


def test_query_and_compressed_key_use_separate_position_series() -> None:
    module = CompressedSlidingWindowSelfAttention(
        _config(compression_ratio=2, window_radius=1)
    )
    captured_positions: list[Tensor] = []

    def capture_positions(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        captured_positions.append(args[0].detach().clone())

    handle = module.compressed_frequency_computer.register_forward_pre_hook(
        capture_positions
    )
    query_positions = torch.arange(5, dtype=torch.long).unsqueeze(-1)
    query_freqs = module.compressed_frequency_computer(query_positions)
    module(
        torch.randn(1, 5, 8),
        freqs_cis=query_freqs,
        state_valid=torch.ones(1, 5, dtype=torch.bool),
    )
    handle.remove()

    assert len(captured_positions) == 2
    torch.testing.assert_close(
        captured_positions[0].squeeze(-1), torch.arange(5, dtype=torch.long)
    )
    torch.testing.assert_close(
        captured_positions[1].squeeze(-1), torch.tensor([0.5, 2.5, 4.0])
    )
    assert captured_positions[0].shape != captured_positions[1].shape


def test_masked_values_and_all_invalid_rows_are_zero_and_invariant() -> None:
    module = CompressedSlidingWindowSelfAttention(_config())
    x = torch.randn(2, 7, 8)
    state_valid = torch.tensor(
        [
            [True, False, True, True, False, True, True],
            [False, False, False, False, False, False, False],
        ]
    )
    freqs_cis = precompute_freqs_cis(dim=4, seqlen=7)
    expected = module(x, freqs_cis=freqs_cis, state_valid=state_valid)
    changed = x.clone()
    changed[~state_valid] = torch.nan
    actual = module(changed, freqs_cis=freqs_cis, state_valid=state_valid)

    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[~state_valid]) == 0
    assert torch.isfinite(actual).all()


def test_non_contiguous_input_and_mask_match_contiguous_versions() -> None:
    module = CompressedSlidingWindowSelfAttention(_config())
    x = torch.randn(2, 8, 7).transpose(1, 2)
    state_valid = torch.ones(2, 14, dtype=torch.bool)[:, ::2]
    state_valid[0, 3] = False
    freqs_cis = precompute_freqs_cis(dim=4, seqlen=7)
    assert not x.is_contiguous()
    assert not state_valid.is_contiguous()

    actual = module(x, freqs_cis=freqs_cis, state_valid=state_valid)
    expected = module(
        x.contiguous(),
        freqs_cis=freqs_cis,
        state_valid=state_valid.contiguous(),
    )

    torch.testing.assert_close(actual, expected)


def test_dropout_eval_determinism_and_training_seed_reproducibility() -> None:
    module = CompressedSlidingWindowSelfAttention(_config(attn_dropout=0.5))
    x = torch.randn(2, 7, 8)
    state_valid = torch.ones(2, 7, dtype=torch.bool)
    freqs_cis = precompute_freqs_cis(dim=4, seqlen=7)

    module.eval()
    torch.manual_seed(1)
    eval_first = module(x, freqs_cis=freqs_cis, state_valid=state_valid)
    torch.manual_seed(99)
    eval_second = module(x, freqs_cis=freqs_cis, state_valid=state_valid)
    module.train()
    torch.manual_seed(29)
    train_first = module(x, freqs_cis=freqs_cis, state_valid=state_valid)
    torch.manual_seed(29)
    train_second = module(x, freqs_cis=freqs_cis, state_valid=state_valid)

    torch.testing.assert_close(eval_first, eval_second, rtol=0, atol=0)
    torch.testing.assert_close(train_first, train_second, rtol=0, atol=0)
    assert torch.isfinite(train_first).all()


def test_bfloat16_input_is_finite_under_cpu_autocast() -> None:
    module = CompressedSlidingWindowSelfAttention(_config())
    x = torch.randn(2, 7, 8, dtype=torch.bfloat16, requires_grad=True)
    state_valid = torch.tensor(
        [
            [True, True, False, True, True, False, True],
            [True, False, True, True, True, False, True],
        ]
    )
    freqs_cis = precompute_freqs_cis(dim=4, seqlen=7)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = module(x, freqs_cis=freqs_cis, state_valid=state_valid)
        loss = output.float().square().mean()
    loss.backward()

    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_small_module_double_gradcheck() -> None:
    module = CompressedSlidingWindowSelfAttention(
        _config(dim=2, n_heads=1, head_dim=2, rope_dim=2, compression_ratio=2)
    ).double()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    x = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
    state_valid = torch.tensor([[True, False, True]])
    freqs_cis = precompute_freqs_cis(dim=2, seqlen=3)

    def attend(values: Tensor) -> Tensor:
        return module.forward(values, freqs_cis=freqs_cis, state_valid=state_valid)

    assert torch.autograd.gradcheck(
        attend,
        (x,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_state_dict_round_trip_preserves_output() -> None:
    source = CompressedSlidingWindowSelfAttention(_config())
    clone = CompressedSlidingWindowSelfAttention(_config())
    clone.load_state_dict(source.state_dict(), strict=True)
    x = torch.randn(2, 7, 8)
    state_valid = torch.ones(2, 7, dtype=torch.bool)
    freqs_cis = precompute_freqs_cis(dim=4, seqlen=7)

    expected = source(x, freqs_cis=freqs_cis, state_valid=state_valid)
    actual = clone(x, freqs_cis=freqs_cis, state_valid=state_valid)

    torch.testing.assert_close(actual, expected)

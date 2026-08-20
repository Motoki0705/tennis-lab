"""Contract and dense-oracle tests for compressed sliding-window attention."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import pytest
import torch
from torch import Tensor, nn

from src.utils.models.components import compressor as compressor_module
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


def _module_separate_projection_forward(
    module: CompressedSlidingWindowSelfAttention,
    x: Tensor,
    freqs_cis: Tensor,
    state_valid: Tensor,
) -> Tensor:
    """Evaluate the pre-packing CSWA projection sequence."""
    n, query_len, _ = x.shape
    masked_x = torch.where(state_valid.unsqueeze(-1), x, torch.zeros_like(x))
    query = module._project(masked_x, module.wq).reshape(
        n,
        query_len,
        module.n_heads,
        module.head_dim,
    )
    query = module._apply_configured_rope(query, freqs_cis).transpose(1, 2)

    compressed = module.compressor.forward_masked(masked_x, state_valid)
    key = compressed.key.transpose(1, 2)
    key_positions = compressed.positions.to(
        device=x.device,
        dtype=module.compressed_frequency_computer.inverse_frequencies.dtype,
    ).unsqueeze(-1)
    key_freqs_cis = module.compressed_frequency_computer(key_positions)
    key = module._apply_configured_rope(key, key_freqs_cis).transpose(1, 2)
    output = module.executor(
        query,
        key,
        compressed.value,
        query_valid=state_valid,
        key_valid=compressed.state_valid,
        dropout_p=module.attn_dropout,
        training=module.training,
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
    with pytest.raises(ValueError, match="compression_ratio=4"):
        CompressedSlidingWindowSelfAttention(_config(backend="cuda"))

    def unavailable(
        backend: str, *, compression_ratio: int, window_radius: int
    ) -> Callable[..., Tensor]:
        del backend, compression_ratio, window_radius
        raise RuntimeError("CUDA extension unavailable at construction")

    compressor_backends: list[str] = []
    reference_pool_resolver = compressor_module.resolve_token_compressor_pool

    def token_pool(
        backend: str, *, compression_ratio: int, head_dim: int
    ) -> Callable[..., tuple[Tensor, Tensor]]:
        compressor_backends.append(backend)
        return cast(
            Callable[..., tuple[Tensor, Tensor]],
            reference_pool_resolver(
                "reference",
                compression_ratio=compression_ratio,
                head_dim=head_dim,
            ),
        )

    monkeypatch.setattr(
        cswa_module, "resolve_compressed_time_local_attention", unavailable
    )
    monkeypatch.setattr(compressor_module, "resolve_token_compressor_pool", token_pool)
    with pytest.raises(RuntimeError, match="at construction"):
        CompressedSlidingWindowSelfAttention(
            _config(
                dim=128,
                n_heads=2,
                head_dim=64,
                compression_ratio=4,
                backend="cuda",
            )
        )
    assert compressor_backends == ["cuda"]


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


def test_full_head_rope_fast_path_matches_prior_concat_output_and_gradient() -> None:
    module = CompressedSlidingWindowSelfAttention(_config()).double()
    actual_input = torch.randn(2, 5, 2, 4, dtype=torch.float64, requires_grad=True)
    prior_input = actual_input.detach().clone().requires_grad_()
    freqs_cis = precompute_freqs_cis(dim=4, seqlen=5)
    upstream = torch.randn_like(actual_input)

    actual = module._apply_configured_rope(actual_input, freqs_cis)
    prior = torch.cat(
        (
            module._apply_rope(prior_input[..., : module.rope_dim], freqs_cis),
            prior_input[..., module.rope_dim :],
        ),
        dim=-1,
    )
    (actual_gradient,) = torch.autograd.grad((actual * upstream).sum(), actual_input)
    (prior_gradient,) = torch.autograd.grad((prior * upstream).sum(), prior_input)

    torch.testing.assert_close(actual, prior, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, prior_gradient, rtol=0, atol=0)


def test_cuda_full_head_rope_passes_holey_raw_qk_and_rank4_phasors_to_executor() -> (
    None
):
    torch.manual_seed(43)
    fused_module = CompressedSlidingWindowSelfAttention(_config())
    reference_module = CompressedSlidingWindowSelfAttention(_config())
    reference_module.load_state_dict(fused_module.state_dict(), strict=True)
    reference_executor = fused_module.executor
    observed: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []

    def emulate_fused_rope(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        query_valid: Tensor,
        key_valid: Tensor,
        dropout_p: float,
        training: bool,
        query_freqs_cis: Tensor,
        key_freqs_cis: Tensor,
    ) -> Tensor:
        observed.append((query, key, query_freqs_cis, key_freqs_cis))
        rotated_query = fused_module._apply_rope(
            query.transpose(1, 2), query_freqs_cis
        ).transpose(1, 2)
        rotated_key = fused_module._apply_rope(
            key.transpose(1, 2), key_freqs_cis
        ).transpose(1, 2)
        return reference_executor(
            rotated_query,
            rotated_key,
            value,
            query_valid=query_valid,
            key_valid=key_valid,
            dropout_p=dropout_p,
            training=training,
        )

    fused_module.backend = "cuda"
    fused_module.executor = emulate_fused_rope
    fused_x = torch.randn(2, 7, 8, requires_grad=True)
    reference_x = fused_x.detach().clone().requires_grad_()
    phase_storage = torch.randn(2, 2, 7, 2, dtype=torch.float32)
    query_freqs_cis = torch.polar(
        torch.ones_like(phase_storage), phase_storage
    ).transpose(1, 2)
    state_valid = torch.tensor(
        [
            [True, False, True, True, False, True, True],
            [False, True, True, True, False, True, True],
        ]
    )
    upstream = torch.randn_like(fused_x)

    actual = fused_module(
        fused_x,
        freqs_cis=query_freqs_cis,
        state_valid=state_valid,
    )
    expected = reference_module(
        reference_x,
        freqs_cis=query_freqs_cis,
        state_valid=state_valid,
    )
    (actual_gradient,) = torch.autograd.grad((actual * upstream).sum(), fused_x)
    (expected_gradient,) = torch.autograd.grad((expected * upstream).sum(), reference_x)

    assert len(observed) == 1
    raw_query, raw_key, observed_query_freqs, observed_key_freqs = observed[0]
    assert raw_query.stride() == (7 * 24, 4, 24, 1)
    assert not raw_query.is_contiguous()
    assert raw_key.shape == (2, 1, 3, 4)
    assert raw_query.dtype == raw_key.dtype == torch.float32
    assert observed_query_freqs is query_freqs_cis
    assert observed_query_freqs.shape == (2, 7, 2, 2)
    assert not observed_query_freqs.is_contiguous()
    assert observed_key_freqs.shape == (3, 1, 2)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize(
    ("rope_dim", "frequency_gradients"),
    [(2, False), (4, True)],
)
def test_cuda_partial_or_frequency_gradient_rope_stays_explicit(
    rope_dim: int,
    frequency_gradients: bool,
) -> None:
    module = CompressedSlidingWindowSelfAttention(_config(rope_dim=rope_dim))
    reference_executor = module.executor
    observed: list[dict[str, object]] = []

    def capture_explicit(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        **kwargs: object,
    ) -> Tensor:
        observed.append(kwargs)
        return reference_executor(query, key, value, **kwargs)

    module.backend = "cuda"
    module.executor = capture_explicit
    x = torch.randn(2, 7, 8, requires_grad=True)
    freqs_cis = precompute_freqs_cis(dim=rope_dim, seqlen=7)
    if frequency_gradients:
        freqs_cis = freqs_cis.detach().requires_grad_()
    output = module(
        x,
        freqs_cis=freqs_cis,
        state_valid=torch.ones(2, 7, dtype=torch.bool),
    )
    output.square().sum().backward()

    assert len(observed) == 1
    assert "query_freqs_cis" not in observed[0]
    assert "key_freqs_cis" not in observed[0]
    if frequency_gradients:
        assert freqs_cis.grad is not None
        assert torch.isfinite(freqs_cis.grad).all()


def test_partial_head_rope_preserves_unrotated_tail() -> None:
    module = CompressedSlidingWindowSelfAttention(_config(rope_dim=2)).double()
    values = torch.randn(2, 5, 2, 4, dtype=torch.float64)
    freqs_cis = precompute_freqs_cis(dim=2, seqlen=5)

    actual = module._apply_configured_rope(values, freqs_cis)
    expected_prefix = module._apply_rope(values[..., :2], freqs_cis)

    torch.testing.assert_close(actual[..., :2], expected_prefix, rtol=0, atol=0)
    torch.testing.assert_close(actual[..., 2:], values[..., 2:], rtol=0, atol=0)


def test_forward_passes_one_shared_kv_head_to_multi_head_executor() -> None:
    module = CompressedSlidingWindowSelfAttention(_config(n_heads=2, head_dim=4))
    resolved_executor = module.executor
    observed_shapes: list[tuple[torch.Size, torch.Size, torch.Size]] = []

    def capture_executor(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        query_valid: Tensor,
        key_valid: Tensor,
        dropout_p: float,
        training: bool,
    ) -> Tensor:
        observed_shapes.append((query.shape, key.shape, value.shape))
        return resolved_executor(
            query,
            key,
            value,
            query_valid=query_valid,
            key_valid=key_valid,
            dropout_p=dropout_p,
            training=training,
        )

    module.executor = capture_executor
    output = module(
        torch.randn(2, 7, 8),
        freqs_cis=precompute_freqs_cis(dim=4, seqlen=7),
        state_valid=torch.ones(2, 7, dtype=torch.bool),
    )

    assert observed_shapes == [
        (
            torch.Size([2, 2, 7, 4]),
            torch.Size([2, 1, 3, 4]),
            torch.Size([2, 1, 3, 4]),
        )
    ]
    assert output.shape == (2, 7, 8)


def test_forward_query_and_executor_output_enable_copy_free_projection_layout() -> None:
    module = CompressedSlidingWindowSelfAttention(_config(n_heads=2, head_dim=4))
    observed_layouts: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def layout_preserving_executor(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        query_valid: Tensor,
        key_valid: Tensor,
        dropout_p: float,
        training: bool,
    ) -> Tensor:
        del key, value, key_valid, dropout_p, training
        output = torch.empty_like(query)
        output.copy_(
            torch.where(query_valid[:, None, :, None], query, torch.zeros_like(query))
        )
        round_trip = output.transpose(1, 2)
        flattened = round_trip.reshape(2, 7, 8)
        observed_layouts.append((query.stride(), output.stride()))
        assert round_trip.is_contiguous()
        assert (
            flattened.untyped_storage().data_ptr()
            == output.untyped_storage().data_ptr()
        )
        return output

    module.executor = layout_preserving_executor
    output = module(
        torch.randn(2, 7, 8),
        freqs_cis=precompute_freqs_cis(dim=4, seqlen=7),
        state_valid=torch.ones(2, 7, dtype=torch.bool),
    )

    assert observed_layouts == [((56, 4, 8, 1), (56, 4, 8, 1))]
    assert output.shape == (2, 7, 8)


def test_forward_uses_validated_projected_compressor_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = CompressedSlidingWindowSelfAttention(_config())
    state_valid = torch.tensor(
        [
            [True, False, True, True, False, True, True],
            [False, True, True, False, True, True, False],
        ]
    )
    x = torch.randn(2, 7, 8)
    expected_projection = module._project_query_kv_gate
    expected_pool = module.compressor.forward_projected
    observed_masked: list[Tensor] = []
    observed_projected: list[tuple[Tensor, Tensor, Tensor]] = []

    def capture_projection(masked_x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        observed_masked.append(masked_x)
        return expected_projection(masked_x)

    def capture_projected(
        raw_kv: Tensor,
        raw_gate: Tensor,
        mask: Tensor,
    ) -> object:
        observed_projected.append((raw_kv, raw_gate, mask))
        return expected_pool(raw_kv, raw_gate, mask)

    def reject_standalone(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("CSWA must not run standalone compressor projections")

    monkeypatch.setattr(module, "_project_query_kv_gate", capture_projection)
    monkeypatch.setattr(module.compressor, "forward_projected", capture_projected)
    monkeypatch.setattr(module.compressor, "forward_masked", reject_standalone)
    monkeypatch.setattr(module.compressor, "forward", reject_standalone)
    module(
        x,
        freqs_cis=precompute_freqs_cis(dim=4, seqlen=7),
        state_valid=state_valid,
    )

    assert len(observed_masked) == 1
    assert torch.count_nonzero(observed_masked[0][~state_valid]) == 0
    assert len(observed_projected) == 1
    raw_kv, raw_gate, observed_mask = observed_projected[0]
    assert raw_kv.shape == (2, 7, 2, 4)
    assert raw_gate.shape == raw_kv.shape
    assert raw_gate.dtype == raw_kv.dtype == x.dtype
    assert observed_mask is state_valid


def test_forward_uses_one_packed_input_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = CompressedSlidingWindowSelfAttention(_config())
    original_linear = cswa_module.F.linear
    observed: list[tuple[torch.Size, torch.Size, bool]] = []

    def capture_linear(input: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        observed.append((input.shape, weight.shape, bias is None))
        return original_linear(input, weight, bias)

    monkeypatch.setattr(cswa_module.F, "linear", capture_linear)
    module(
        torch.randn(2, 7, 8),
        freqs_cis=precompute_freqs_cis(dim=4, seqlen=7),
        state_valid=torch.ones(2, 7, dtype=torch.bool),
    )

    assert observed == [
        (torch.Size([2, 7, 8]), torch.Size([24, 8]), True),
        (torch.Size([2, 7, 8]), torch.Size([8, 8]), True),
    ]


def test_packed_projection_matches_separate_projection_forward_and_backward() -> None:
    torch.manual_seed(37)
    packed_module = CompressedSlidingWindowSelfAttention(_config()).double()
    with torch.no_grad():
        packed_module.compressor.w_kv.bias.copy_(
            torch.linspace(-0.8, 0.7, 8, dtype=torch.float64)
        )
        packed_module.compressor.w_gate.bias.copy_(
            torch.linspace(0.6, -0.9, 8, dtype=torch.float64)
        )
        packed_module.compressor.w_gate.weight.normal_(mean=0.0, std=0.2)
        packed_module.compressor.ape.normal_(mean=0.0, std=0.3)
    separate_module = CompressedSlidingWindowSelfAttention(_config()).double()
    separate_module.load_state_dict(packed_module.state_dict(), strict=True)
    packed_x = torch.randn(3, 7, 8, dtype=torch.float64, requires_grad=True)
    separate_x = packed_x.detach().clone().requires_grad_()
    state_valid = torch.tensor(
        [
            [True, False, True, True, False, True, True],
            [False, False, False, False, False, False, False],
            [True, True, True, False, True, False, True],
        ]
    )
    freqs_cis = precompute_freqs_cis(dim=4, seqlen=7)
    upstream = torch.randn_like(packed_x)

    packed = packed_module(
        packed_x,
        freqs_cis=freqs_cis,
        state_valid=state_valid,
    )
    separate = _module_separate_projection_forward(
        separate_module,
        separate_x,
        freqs_cis,
        state_valid,
    )
    packed_named_parameters = dict(packed_module.named_parameters())
    separate_named_parameters = dict(separate_module.named_parameters())
    assert packed_named_parameters.keys() == separate_named_parameters.keys()
    packed_gradients = torch.autograd.grad(
        (packed * upstream).sum(),
        (packed_x, *packed_named_parameters.values()),
    )
    separate_gradients = torch.autograd.grad(
        (separate * upstream).sum(),
        (separate_x, *separate_named_parameters.values()),
    )

    assert torch.count_nonzero(packed_module.compressor.w_kv.bias) == 8
    assert torch.count_nonzero(packed_module.compressor.w_gate.bias) == 8
    assert torch.count_nonzero(packed[~state_valid]) == 0
    torch.testing.assert_close(packed, separate, atol=1e-11, rtol=1e-9)
    for name, packed_gradient, separate_gradient in zip(
        ("input", *packed_named_parameters.keys()),
        packed_gradients,
        separate_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            packed_gradient,
            separate_gradient,
            atol=1e-10,
            rtol=1e-8,
            msg=f"gradient mismatch for {name}",
        )
    assert torch.count_nonzero(packed_gradients[0][~state_valid]) == 0


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


def test_invalid_output_loss_has_zero_input_and_parameter_contributions() -> None:
    module = CompressedSlidingWindowSelfAttention(_config())
    assert module.wo.bias is None
    x = torch.randn(2, 7, 8, requires_grad=True)
    state_valid = torch.tensor(
        [
            [True, False, True, True, False, True, True],
            [False, True, True, False, True, False, True],
        ]
    )
    output = module(
        x,
        freqs_cis=precompute_freqs_cis(dim=4, seqlen=7),
        state_valid=state_valid,
    )
    upstream = torch.randn_like(output).masked_fill(state_valid.unsqueeze(-1), 0)
    output.backward(upstream)

    assert torch.count_nonzero(output[~state_valid]) == 0
    assert x.grad is not None
    assert torch.count_nonzero(x.grad) == 0
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


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
    assert list(source.state_dict()) == [
        "wq.weight",
        "compressor.ape",
        "compressor.w_kv.weight",
        "compressor.w_kv.bias",
        "compressor.w_gate.weight",
        "compressor.w_gate.bias",
        "wo.weight",
    ]
    clone.load_state_dict(source.state_dict(), strict=True)
    x = torch.randn(2, 7, 8)
    state_valid = torch.ones(2, 7, dtype=torch.bool)
    freqs_cis = precompute_freqs_cis(dim=4, seqlen=7)

    expected = source(x, freqs_cis=freqs_cis, state_valid=state_valid)
    actual = clone(x, freqs_cis=freqs_cis, state_valid=state_valid)

    torch.testing.assert_close(actual, expected)

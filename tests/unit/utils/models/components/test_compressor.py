"""Contract tests for fixed-ratio token-level KV compression."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from src.utils.models.components.compressor import (
    CompressedKV,
    TokenLevelCompressorConfig,
    TokenLevelKVCompressor,
)
from src.utils.models.components.ops.token_compressor import (
    build_token_compressor_layout,
)


def _config(
    *,
    dim: int = 8,
    n_heads: int = 2,
    head_dim: int = 4,
    compression_ratio: int = 2,
    overlap: bool = True,
) -> TokenLevelCompressorConfig:
    return TokenLevelCompressorConfig(
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        compression_ratio=compression_ratio,
        overlap=overlap,
    )


def _loop_oracle(
    module: TokenLevelKVCompressor,
    x: Tensor,
    state_valid: Tensor,
) -> CompressedKV:
    """Small direct implementation of the documented channel-wise reduction."""
    n, sequence_length, _ = x.shape
    ratio = module.compression_ratio
    compressed_length = (sequence_length + ratio - 1) // ratio
    masked_x = torch.where(state_valid.unsqueeze(-1), x, torch.zeros_like(x))
    raw_kv = F.linear(masked_x, module.w_kv.weight, module.w_kv.bias).reshape(
        n,
        sequence_length,
        2,
        module.kv_dim,
    )
    raw_gate = F.linear(
        masked_x,
        module.w_gate.weight,
        module.w_gate.bias,
    ).reshape(n, sequence_length, 2, module.kv_dim)
    ape = module.ape.reshape(ratio, 2, module.kv_dim)
    raw_gate = raw_gate + ape[
        torch.arange(sequence_length, device=x.device) % ratio
    ].unsqueeze(0)

    rows: list[Tensor] = []
    valid_rows: list[Tensor] = []
    for compressed_index in range(compressed_length):
        sample_rows: list[Tensor] = []
        sample_valid: list[Tensor] = []
        for batch_index in range(n):
            source_values: list[Tensor] = []
            source_logits: list[Tensor] = []
            for branch, block_index in (
                (0, compressed_index - 1),
                (1, compressed_index),
            ):
                start = block_index * ratio
                for token_index in range(start, start + ratio):
                    if (
                        0 <= token_index < sequence_length
                        and state_valid[batch_index, token_index]
                    ):
                        source_values.append(raw_kv[batch_index, token_index, branch])
                        source_logits.append(raw_gate[batch_index, token_index, branch])
            if source_values:
                values = torch.stack(source_values).to(
                    torch.float64 if x.dtype == torch.float64 else torch.float32
                )
                logits = torch.stack(source_logits).to(values.dtype)
                weights = torch.softmax(logits, dim=0)
                sample_rows.append((weights * values).sum(dim=0).to(x.dtype))
                sample_valid.append(torch.tensor(True, device=x.device))
            else:
                sample_rows.append(
                    torch.zeros(module.kv_dim, dtype=x.dtype, device=x.device)
                )
                sample_valid.append(torch.tensor(False, device=x.device))
        rows.append(torch.stack(sample_rows))
        valid_rows.append(torch.stack(sample_valid))

    compressed = torch.stack(rows, dim=1)
    compressed_valid = torch.stack(valid_rows, dim=1)
    shared_kv = compressed.unsqueeze(1)
    positions = (
        torch.arange(compressed_length, device=x.device, dtype=torch.float32) * ratio
        + (ratio - 1) / 2
    ).clamp_max(float(sequence_length - 1))
    return CompressedKV(
        key=shared_kv,
        value=shared_kv,
        state_valid=compressed_valid,
        positions=positions,
    )


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"dim": 0}, ValueError, "dim must be positive"),
        ({"n_heads": 0}, ValueError, "n_heads must be positive"),
        ({"head_dim": 0}, ValueError, "head_dim must be positive"),
        (
            {"compression_ratio": 1},
            ValueError,
            "compression_ratio must be at least 2",
        ),
        ({"overlap": False}, ValueError, "overlap must be True"),
        ({"dim": 7}, ValueError, r"dim must equal n_heads \* head_dim"),
        ({"dim": True}, TypeError, "dim must be an integer"),
        ({"overlap": 1}, TypeError, "overlap must be a bool"),
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
        "compression_ratio": 2,
        "overlap": True,
    }
    kwargs.update(overrides)
    with pytest.raises(error_type, match=message):
        TokenLevelCompressorConfig(**kwargs)  # type: ignore[arg-type]


def test_constructor_requires_typed_config_and_uses_head_dim_projection_width() -> None:
    with pytest.raises(TypeError, match="TokenLevelCompressorConfig"):
        TokenLevelKVCompressor(object())  # type: ignore[arg-type]

    module = TokenLevelKVCompressor(
        _config(dim=12, n_heads=3, head_dim=4, compression_ratio=3)
    )

    assert module.kv_dim == 4
    assert module.w_kv.weight.shape == (8, 12)
    assert module.w_kv.bias.shape == (8,)
    assert module.w_gate.weight.shape == (8, 12)
    assert module.w_gate.bias.shape == (8,)
    assert module.ape.shape == (3, 8)
    assert torch.count_nonzero(module.w_gate.weight) == 0
    assert torch.count_nonzero(module.w_gate.bias) == 0
    assert torch.count_nonzero(module.ape) == 0


def test_constructor_rejects_unknown_backend_without_automatic_dispatch() -> None:
    with pytest.raises(ValueError, match="Unsupported token-compressor backend"):
        TokenLevelKVCompressor(
            _config(),
            backend="automatic",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="compression_ratio=4"):
        TokenLevelKVCompressor(_config(compression_ratio=3), backend="cuda")
    with pytest.raises(ValueError, match="head_dim in"):
        TokenLevelKVCompressor(_config(compression_ratio=4), backend="cuda")


@pytest.mark.parametrize(
    ("make_x", "make_mask", "error_type", "message"),
    [
        (
            lambda: torch.randn(2, 8),
            lambda: torch.ones(2, 1, dtype=torch.bool),
            ValueError,
            "x must have shape",
        ),
        (
            lambda: torch.randn(2, 3, 7),
            lambda: torch.ones(2, 3, dtype=torch.bool),
            ValueError,
            "feature dimension",
        ),
        (
            lambda: torch.randn(2, 0, 8),
            lambda: torch.ones(2, 0, dtype=torch.bool),
            ValueError,
            "sequence length T must be positive",
        ),
        (
            lambda: torch.ones(2, 3, 8, dtype=torch.int64),
            lambda: torch.ones(2, 3, dtype=torch.bool),
            TypeError,
            "x must use",
        ),
        (
            lambda: torch.randn(2, 3, 8),
            lambda: torch.ones(2, 2, dtype=torch.bool),
            ValueError,
            "state_valid shape",
        ),
        (
            lambda: torch.randn(2, 3, 8),
            lambda: torch.ones(2, 3),
            TypeError,
            "state_valid must have dtype bool",
        ),
    ],
)
def test_forward_rejects_invalid_runtime_contracts(
    make_x: Callable[[], Tensor],
    make_mask: Callable[[], Tensor],
    error_type: type[Exception],
    message: str,
) -> None:
    module = TokenLevelKVCompressor(_config())
    with pytest.raises(error_type, match=message):
        module(make_x(), make_mask())


@pytest.mark.parametrize(
    ("sequence_length", "ratio", "compressed_length", "positions"),
    [
        (8, 4, 2, [1.5, 5.5]),
        (9, 4, 3, [1.5, 5.5, 8.0]),
        (2, 4, 1, [1.0]),
    ],
)
def test_ceil_tail_shapes_and_static_positions(
    sequence_length: int,
    ratio: int,
    compressed_length: int,
    positions: list[float],
) -> None:
    module = TokenLevelKVCompressor(_config(compression_ratio=ratio))
    x = torch.randn(3, sequence_length, 8)
    output = module(x, torch.ones(3, sequence_length, dtype=torch.bool))

    assert output.key.shape == (3, 1, compressed_length, 4)
    assert output.value.shape == (3, 1, compressed_length, 4)
    assert output.key is output.value
    assert output.state_valid.shape == (3, compressed_length)
    assert output.positions.dtype == torch.float32
    torch.testing.assert_close(output.positions, torch.tensor(positions))


def test_source_layout_covers_first_middle_and_partial_last_blocks() -> None:
    layout = build_token_compressor_layout(5, 2, torch.device("cpu"))

    torch.testing.assert_close(
        layout.source_indices,
        torch.tensor(
            [
                [0, 0, 0, 1],
                [0, 1, 2, 3],
                [2, 3, 4, 4],
            ]
        ),
    )
    torch.testing.assert_close(
        layout.source_branches,
        torch.tensor(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ]
        ),
    )
    torch.testing.assert_close(
        layout.boundary_valid,
        torch.tensor(
            [
                [False, False, True, True],
                [True, True, True, True],
                [True, True, True, False],
            ]
        ),
    )


def test_compressed_validity_covers_sparse_single_and_all_invalid_rows() -> None:
    module = TokenLevelKVCompressor(_config(compression_ratio=2))
    x = torch.randn(4, 5, 8)
    state_valid = torch.tensor(
        [
            [True, True, True, True, True],
            [True, False, False, False, True],
            [False, True, False, False, False],
            [False, False, False, False, False],
        ]
    )

    output = module(x, state_valid)

    torch.testing.assert_close(
        output.state_valid,
        torch.tensor(
            [
                [True, True, True],
                [True, True, True],
                [True, True, False],
                [False, False, False],
            ]
        ),
    )
    assert torch.count_nonzero(output.key[3]) == 0
    assert torch.count_nonzero(output.value[3]) == 0
    assert torch.isfinite(output.key).all()
    assert torch.isfinite(output.value).all()


def test_padding_values_do_not_affect_compressed_outputs() -> None:
    torch.manual_seed(11)
    module = TokenLevelKVCompressor(_config(compression_ratio=3))
    state_valid = torch.tensor([[True, False, True, False, False, True, False]])
    base = torch.randn(1, 7, 8)
    changed = base.clone()
    changed[~state_valid] = torch.randn_like(changed[~state_valid]) * 10_000

    expected = module(base, state_valid)
    actual = module(changed, state_valid)

    torch.testing.assert_close(actual.key, expected.key)
    torch.testing.assert_close(actual.value, expected.value)
    torch.testing.assert_close(actual.state_valid, expected.state_valid)


def test_validated_masked_and_projected_seams_match_standalone_masking() -> None:
    module = TokenLevelKVCompressor(_config(compression_ratio=3))
    x = torch.randn(2, 7, 8)
    state_valid = torch.tensor(
        [
            [True, False, True, True, False, True, True],
            [False, True, True, False, True, False, True],
        ]
    )
    masked_x = torch.where(state_valid.unsqueeze(-1), x, torch.zeros_like(x))
    raw_kv = module._project(masked_x, module.w_kv).reshape(2, 7, 2, 4)
    raw_gate = module._project(masked_x, module.w_gate).reshape(2, 7, 2, 4)

    expected = module(x, state_valid)
    masked_actual = module.forward_masked(masked_x, state_valid)
    projected_actual = module.forward_projected(raw_kv, raw_gate, state_valid)

    for actual in (masked_actual, projected_actual):
        torch.testing.assert_close(actual.key, expected.key)
        torch.testing.assert_close(actual.value, expected.value)
        torch.testing.assert_close(actual.state_valid, expected.state_valid)
        assert actual.key is actual.value
    with pytest.raises(ValueError, match="state_valid shape"):
        module.forward_masked(masked_x, state_valid[:, :-1])


def test_projected_seam_rejects_invalid_shapes_and_dtypes() -> None:
    module = TokenLevelKVCompressor(_config(compression_ratio=3))
    raw_kv = torch.randn(2, 7, 2, 4)
    raw_gate = torch.randn_like(raw_kv)
    state_valid = torch.ones(2, 7, dtype=torch.bool)

    with pytest.raises(ValueError, match="raw_kv must have shape"):
        module.forward_projected(raw_kv.flatten(2), raw_gate, state_valid)
    with pytest.raises(ValueError, match="raw_gate shape must equal"):
        module.forward_projected(raw_kv, raw_gate[:, :-1], state_valid)
    with pytest.raises(TypeError, match="raw_gate dtype must equal"):
        module.forward_projected(raw_kv, raw_gate.double(), state_valid)
    with pytest.raises(TypeError, match="raw_kv must use"):
        module.forward_projected(raw_kv.long(), raw_gate.long(), state_valid)
    with pytest.raises(ValueError, match="state_valid shape"):
        module.forward_projected(raw_kv, raw_gate, state_valid[:, :-1])
    with pytest.raises(TypeError, match="state_valid must have dtype bool"):
        module.forward_projected(raw_kv, raw_gate, state_valid.float())


def test_forward_matches_loop_based_channel_wise_oracle() -> None:
    torch.manual_seed(17)
    module = TokenLevelKVCompressor(_config(compression_ratio=3))
    with torch.no_grad():
        module.w_gate.weight.normal_(mean=0.0, std=0.2)
        module.w_gate.bias.normal_(mean=0.0, std=0.1)
        module.ape.normal_(mean=0.0, std=0.3)
    x = torch.randn(2, 7, 8)
    state_valid = torch.tensor(
        [
            [True, False, True, True, False, True, True],
            [False, False, False, True, False, False, False],
        ]
    )

    expected = _loop_oracle(module, x, state_valid)
    actual = module(x, state_valid)

    torch.testing.assert_close(actual.key, expected.key, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.value, expected.value, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.state_valid, expected.state_valid)
    torch.testing.assert_close(actual.positions, expected.positions)


def test_two_branch_projection_produces_one_tied_key_value_latent() -> None:
    module = TokenLevelKVCompressor(_config(dim=6, n_heads=2, head_dim=3))
    with torch.no_grad():
        module.w_kv.weight.zero_()
        module.w_kv.bias.zero_()
        current_branch = module.w_kv.bias.reshape(2, module.kv_dim)[1]
        current_branch.copy_(torch.arange(module.kv_dim, dtype=torch.float32))
    output = module(torch.zeros(1, 1, 6), torch.ones(1, 1, dtype=torch.bool))

    torch.testing.assert_close(
        output.key,
        torch.tensor([[[[0.0, 1.0, 2.0]]]]),
    )
    assert output.key is output.value
    assert output.key.data_ptr() == output.value.data_ptr()


def test_non_contiguous_inputs_match_contiguous_inputs() -> None:
    module = TokenLevelKVCompressor(_config())
    x = torch.randn(2, 8, 5).transpose(1, 2)
    state_valid = torch.ones(5, 2, dtype=torch.bool).transpose(0, 1)
    state_valid[0, 2] = False
    assert not x.is_contiguous()
    assert not state_valid.is_contiguous()

    strided = module(x, state_valid)
    contiguous = module(x.contiguous(), state_valid.contiguous())

    torch.testing.assert_close(strided.key, contiguous.key)
    torch.testing.assert_close(strided.value, contiguous.value)
    torch.testing.assert_close(strided.state_valid, contiguous.state_valid)


def test_float32_backward_reaches_input_and_all_parameters() -> None:
    torch.manual_seed(23)
    module = TokenLevelKVCompressor(_config(compression_ratio=3))
    x = torch.randn(2, 7, 8, requires_grad=True)
    state_valid = torch.tensor(
        [
            [True, True, False, True, False, True, True],
            [True, False, True, True, True, False, True],
        ]
    )

    output = module(x, state_valid)
    loss = output.key.square().sum() + output.value.square().sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[state_valid]) > 0
    assert torch.count_nonzero(x.grad[~state_valid]) == 0
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_small_double_gradcheck() -> None:
    torch.manual_seed(29)
    module = TokenLevelKVCompressor(
        _config(dim=2, n_heads=1, head_dim=2, compression_ratio=2)
    )
    module.double()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    x = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
    state_valid = torch.tensor([[True, False, True]])

    def compress(values: Tensor) -> Tensor:
        output = module.forward(values, state_valid)
        return output.key

    assert torch.autograd.gradcheck(
        compress,
        (x,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_cpu_autocast_keeps_outputs_finite_and_gradients_finite() -> None:
    module = TokenLevelKVCompressor(_config(compression_ratio=2))
    x = torch.randn(2, 5, 8, requires_grad=True)
    state_valid = torch.tensor(
        [
            [True, True, False, True, True],
            [False, True, True, False, True],
        ]
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = module(x, state_valid)
        loss = output.key.square().mean() + output.value.square().mean()
    loss.backward()

    assert output.key.dtype == x.dtype
    assert output.value.dtype == x.dtype
    assert torch.isfinite(output.key).all()
    assert torch.isfinite(output.value).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_state_dict_round_trip_is_exact() -> None:
    torch.manual_seed(31)
    source = TokenLevelKVCompressor(_config(compression_ratio=4))
    clone = TokenLevelKVCompressor(_config(compression_ratio=4))
    clone.load_state_dict(source.state_dict(), strict=True)
    x = torch.randn(2, 6, 8)
    state_valid = torch.tensor(
        [
            [True, True, False, True, False, True],
            [False, True, True, True, True, False],
        ]
    )

    expected = source(x, state_valid)
    actual = clone(x, state_valid)

    torch.testing.assert_close(actual.key, expected.key)
    torch.testing.assert_close(actual.value, expected.value)
    torch.testing.assert_close(actual.state_valid, expected.state_valid)
    torch.testing.assert_close(actual.positions, expected.positions)


def test_parameter_count_does_not_depend_on_runtime_source_count() -> None:
    module = TokenLevelKVCompressor(_config(compression_ratio=4))
    count = sum(parameter.numel() for parameter in module.parameters())

    for sequence_length in (1, 3, 4, 5, 19):
        x = torch.randn(2, sequence_length, 8)
        state_valid = torch.ones(2, sequence_length, dtype=torch.bool)
        output = module(x, state_valid)
        assert output.key.shape[2] == (sequence_length + 3) // 4
        assert sum(parameter.numel() for parameter in module.parameters()) == count

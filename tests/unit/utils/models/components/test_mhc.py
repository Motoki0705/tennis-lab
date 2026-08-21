"""Unit tests for fixed-width manifold-constrained hyper-connections."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor

from src.utils.models.components.mhc import (
    ManifoldConstrainedHyperConnection,
    MHCConfig,
)


def _config(**overrides: object) -> MHCConfig:
    values: dict[str, object] = {
        "dim": 6,
        "num_streams": 4,
        "coefficient_dim": 8,
        "sinkhorn_iters": 20,
        "eps": 1e-6,
        "residual_identity_bias": 4.0,
        "update_scale_init": 0.0,
    }
    values.update(overrides)
    return MHCConfig(**values)  # type: ignore[arg-type]


def _activate_dynamic_coefficients(
    mhc: ManifoldConstrainedHyperConnection,
) -> None:
    """Open the residual gate and move zero-initialized heads off zero."""
    with torch.no_grad():
        mhc.residual_mix_gate.fill_(0.5)
        mhc.pre_head.weight.normal_(std=0.02)
        mhc.pre_head.bias.normal_(std=0.02)
        mhc.post_head.weight.normal_(std=0.02)
        mhc.post_head.bias.normal_(std=0.02)
        mhc.residual_key.weight.normal_(std=0.02)
        mhc.pair_out.weight.normal_(std=0.02)
        mhc.pair_out.bias.normal_(std=0.02)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dim", 0),
        ("num_streams", -1),
        ("coefficient_dim", True),
        ("sinkhorn_iters", 0),
        ("eps", 0.0),
        ("eps", float("inf")),
        ("residual_identity_bias", -0.1),
        ("residual_identity_bias", float("nan")),
        ("update_scale_init", float("inf")),
    ],
)
def test_config_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


def test_constructor_requires_typed_config() -> None:
    with pytest.raises(TypeError, match="MHCConfig"):
        ManifoldConstrainedHyperConnection(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("streams", "mask", "error"),
    [
        (torch.randn(6), torch.ones(4, dtype=torch.bool), "at least 2"),
        (torch.randn(3, 6), torch.ones(3, dtype=torch.bool), "trailing shape"),
        (torch.randn(4, 5), torch.ones(4, dtype=torch.bool), "trailing shape"),
        (torch.randn(2, 4, 6), torch.ones(4, dtype=torch.bool), "valid_mask shape"),
        (torch.randn(4, 6), torch.ones(4), "dtype bool"),
        (
            torch.ones(4, 6, dtype=torch.int64),
            torch.ones(4, dtype=torch.bool),
            "streams must use",
        ),
    ],
)
def test_pre_rejects_invalid_runtime_contracts(
    streams: Tensor,
    mask: Tensor,
    error: str,
) -> None:
    mhc = ManifoldConstrainedHyperConnection(_config())

    with pytest.raises((TypeError, ValueError), match=error):
        mhc.pre(streams, mask)


def test_pre_rejects_input_on_a_different_device_from_module() -> None:
    mhc = ManifoldConstrainedHyperConnection(_config())
    streams = torch.empty(4, 6, device="meta")
    mask = torch.ones(4, dtype=torch.bool, device="meta")

    with pytest.raises(ValueError, match="module device"):
        mhc.pre(streams, mask)


def test_pre_rejects_mask_on_a_different_device_from_streams() -> None:
    mhc = ManifoldConstrainedHyperConnection(_config())
    streams = torch.empty(4, 6)
    mask = torch.ones(4, dtype=torch.bool, device="meta")

    with pytest.raises(ValueError, match="streams and valid_mask"):
        mhc.pre(streams, mask)


def test_pre_and_post_preserve_multiple_leading_dimensions_and_noncontiguous_input() -> (
    None
):
    torch.manual_seed(10)
    mhc = ManifoldConstrainedHyperConnection(_config(update_scale_init=0.25))
    contiguous = torch.randn(2, 3, 4, 6)
    streams = contiguous.transpose(0, 1)
    mask = torch.ones(3, 2, 4, dtype=torch.bool).transpose(0, 1).transpose(0, 1)
    assert not streams.is_contiguous()

    projected, state = mhc.pre(streams, mask)
    output = mhc.post(torch.randn(3, 2, 1, 6), streams, state)

    assert projected.shape == (3, 2, 1, 6)
    assert state.residual_mix.shape == (3, 2, 4, 4)
    assert state.post_weights.shape == (3, 2, 4, 1)
    assert output.shape == streams.shape
    assert torch.isfinite(projected).all()
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([[True, True, True, True]]),
        torch.tensor([[True, False, True, False]]),
        torch.tensor([[False, True, False, False]]),
        torch.tensor([[False, False, False, False]]),
    ],
)
def test_masked_sinkhorn_has_valid_row_and_column_sums(mask: Tensor) -> None:
    torch.manual_seed(11)
    mhc = ManifoldConstrainedHyperConnection(_config())
    _activate_dynamic_coefficients(mhc)
    streams = torch.randn(1, 4, 6)

    projected, state = mhc.pre(streams, mask)
    mix = state.residual_mix
    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)

    assert torch.all(mix >= 0.0)
    assert torch.equal(mix.masked_select(~pair_mask), torch.zeros_like(mix[~pair_mask]))
    if mask.any():
        torch.testing.assert_close(
            mix.sum(dim=-1)[mask],
            torch.ones_like(mix.sum(dim=-1)[mask]),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            mix.sum(dim=-2)[mask],
            torch.ones_like(mix.sum(dim=-2)[mask]),
            atol=1e-4,
            rtol=1e-4,
        )
    else:
        assert torch.equal(projected, torch.zeros_like(projected))
        output = mhc.post(torch.randn(1, 1, 6), streams, state)
        assert torch.equal(output, torch.zeros_like(output))


def test_padding_values_do_not_affect_valid_results_or_receive_gradients() -> None:
    torch.manual_seed(12)
    mhc = ManifoldConstrainedHyperConnection(_config(update_scale_init=0.3))
    _activate_dynamic_coefficients(mhc)
    mask = torch.tensor([[True, False, True, False]])
    streams = torch.randn(1, 4, 6)
    changed = streams.clone()
    changed[:, ~mask[0]] = torch.randn(1, 2, 6) * 10_000.0
    update = torch.randn(1, 1, 6)

    projected, state = mhc.pre(streams, mask)
    projected_changed, state_changed = mhc.pre(changed, mask)
    output = mhc.post(update, streams, state)
    output_changed = mhc.post(update, changed, state_changed)

    torch.testing.assert_close(projected, projected_changed)
    torch.testing.assert_close(state.residual_mix, state_changed.residual_mix)
    torch.testing.assert_close(state.post_weights, state_changed.post_weights)
    torch.testing.assert_close(output[mask], output_changed[mask])
    assert torch.equal(output[~mask], torch.zeros_like(output[~mask]))

    grad_streams = changed.detach().requires_grad_(True)
    grad_update = update.detach().requires_grad_(True)
    grad_projected, grad_state = mhc.pre(grad_streams, mask)
    grad_output = mhc.post(grad_update, grad_streams, grad_state)
    (grad_projected.square().sum() + grad_output.square().sum()).backward()

    assert grad_streams.grad is not None
    assert grad_update.grad is not None
    assert torch.equal(
        grad_streams.grad[~mask], torch.zeros_like(grad_streams.grad[~mask])
    )
    assert torch.isfinite(grad_streams.grad).all()
    assert torch.isfinite(grad_update.grad).all()


def test_stream_permutation_is_equivariant() -> None:
    torch.manual_seed(13)
    mhc = ManifoldConstrainedHyperConnection(_config(update_scale_init=0.2))
    _activate_dynamic_coefficients(mhc)
    streams = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, False, True, True], [False, True, True, False]])
    update = torch.randn(2, 1, 6)
    permutation = torch.tensor([2, 0, 3, 1])

    projected, state = mhc.pre(streams, mask)
    output = mhc.post(update, streams, state)
    permuted_projected, permuted_state = mhc.pre(
        streams[:, permutation], mask[:, permutation]
    )
    permuted_output = mhc.post(
        update,
        streams[:, permutation],
        permuted_state,
    )

    torch.testing.assert_close(permuted_projected, projected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        permuted_state.residual_mix,
        state.residual_mix[:, permutation][:, :, permutation],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        permuted_state.post_weights,
        state.post_weights[:, permutation],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        permuted_output,
        output[:, permutation],
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([[True, True, True, True]]),
        torch.tensor([[True, False, True, False]]),
        torch.tensor([[False, True, False, False]]),
        torch.tensor([[False, False, False, False]]),
    ],
    ids=("all-valid", "partial-noncontiguous", "single-valid", "all-invalid"),
)
def test_initialization_is_exact_masked_identity_and_update_preserving(
    mask: Tensor,
) -> None:
    mhc = ManifoldConstrainedHyperConnection(_config())
    streams = torch.randn(1, 4, 6)
    arbitrary_update = torch.randn(1, 1, 6) * 1_000.0

    _, state = mhc.pre(streams, mask)
    output = mhc.post(arbitrary_update, streams, state)
    expected_mix = (
        torch.eye(4).unsqueeze(0) * mask.unsqueeze(-1).to(dtype=streams.dtype)
    )
    expected_output = streams * mask.unsqueeze(-1)

    assert torch.equal(state.residual_mix, expected_mix)
    assert mhc.update_scale.item() == 0.0
    assert mhc.residual_mix_gate.item() == 0.0
    assert torch.equal(output, expected_output)


def test_zero_residual_mix_gate_is_not_dead() -> None:
    mhc = ManifoldConstrainedHyperConnection(_config())
    streams = torch.randn(1, 4, 6)
    mask = torch.ones(1, 4, dtype=torch.bool)

    _, state = mhc.pre(streams, mask)
    cost = torch.zeros(1, 4, 4, dtype=streams.dtype)
    cost[0, 0, 1] = 1.0
    (state.residual_mix * cost).sum().backward()

    assert mhc.residual_mix_gate.grad is not None
    assert torch.isfinite(mhc.residual_mix_gate.grad)
    assert mhc.residual_mix_gate.grad.abs().item() > 0.0


def test_residual_mix_gate_opens_from_zero_and_clamps_to_convex_endpoints() -> None:
    """Challenge both trainability at zero and bounded loaded-state behavior."""
    mhc = ManifoldConstrainedHyperConnection(_config())
    streams = torch.randn(1, 4, 6)
    mask = torch.ones(1, 4, dtype=torch.bool)
    identity = torch.eye(4).unsqueeze(0)

    _, initial_state = mhc.pre(streams, mask)
    opening_loss = -initial_state.residual_mix[0, 0, 1]
    opening_loss.backward()

    gradient = mhc.residual_mix_gate.grad
    assert gradient is not None
    assert gradient.item() < 0.0
    with torch.no_grad():
        mhc.residual_mix_gate.add_(-0.5 * gradient)
    assert 0.0 < mhc.residual_mix_gate.item() < 1.0

    _, opened_state = mhc.pre(streams, mask)
    assert opened_state.residual_mix[0, 0, 1].item() > 0.0
    assert not torch.equal(opened_state.residual_mix, identity)

    with torch.no_grad():
        mhc.residual_mix_gate.fill_(1.0)
    _, unit_gate_state = mhc.pre(streams, mask)
    with torch.no_grad():
        mhc.residual_mix_gate.fill_(10.0)
    _, upper_clamped_state = mhc.pre(streams, mask)
    torch.testing.assert_close(
        upper_clamped_state.residual_mix,
        unit_gate_state.residual_mix,
        rtol=0,
        atol=0,
    )

    with torch.no_grad():
        mhc.residual_mix_gate.fill_(-10.0)
    _, lower_clamped_state = mhc.pre(streams, mask)
    assert torch.equal(lower_clamped_state.residual_mix, identity)


def test_forward_and_backward_are_finite() -> None:
    torch.manual_seed(14)
    mhc = ManifoldConstrainedHyperConnection(_config(update_scale_init=0.5))
    _activate_dynamic_coefficients(mhc)
    streams = torch.randn(2, 4, 6, requires_grad=True)
    update = torch.randn(2, 1, 6, requires_grad=True)
    mask = torch.tensor([[True, True, False, True], [False, True, True, False]])

    projected, state = mhc.pre(streams, mask)
    output = mhc.post(update, streams, state)
    loss = projected.square().mean() + output.square().mean()
    loss.backward()

    assert torch.isfinite(projected).all()
    assert torch.isfinite(output).all()
    assert streams.grad is not None and torch.isfinite(streams.grad).all()
    assert update.grad is not None and torch.isfinite(update.grad).all()
    for parameter in mhc.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    for parameter in (
        mhc.residual_mix_gate,
        mhc.residual_query.weight,
        mhc.residual_key.weight,
        mhc.pair_in.weight,
        mhc.pair_out.weight,
    ):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad).item() > 0


def test_double_precision_gradcheck() -> None:
    torch.manual_seed(15)
    mhc = ManifoldConstrainedHyperConnection(
        _config(
            dim=2,
            num_streams=2,
            coefficient_dim=3,
            sinkhorn_iters=6,
            residual_identity_bias=2.0,
            update_scale_init=0.2,
        )
    ).double()
    streams = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
    update = torch.randn(1, 1, 2, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True]])

    def apply_mhc(residual: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
        projected, state = mhc.pre(residual, mask)
        return projected, mhc.post(delta, residual, state)

    assert torch.autograd.gradcheck(
        apply_mhc,
        (streams, update),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_post_rejects_state_residual_and_update_mismatches() -> None:
    mhc = ManifoldConstrainedHyperConnection(_config())
    streams = torch.randn(2, 4, 6)
    mask = torch.ones(2, 4, dtype=torch.bool)
    update = torch.randn(2, 1, 6)
    _, state = mhc.pre(streams, mask)

    with pytest.raises(ValueError, match="update shape"):
        mhc.post(torch.randn(2, 2, 6), streams, state)
    with pytest.raises(ValueError, match="trailing shape"):
        mhc.post(update, torch.randn(2, 3, 6), state)
    with pytest.raises(TypeError, match="update dtype"):
        mhc.post(update.double(), streams, state)
    with pytest.raises(TypeError, match="state.residual_mix dtype"):
        mhc.post(update.double(), streams.double(), state)

    bad_mix = replace(state, residual_mix=state.residual_mix[..., :-1])
    with pytest.raises(ValueError, match="residual_mix shape"):
        mhc.post(update, streams, bad_mix)
    bad_post = replace(state, post_weights=state.post_weights[..., :-1, :])
    with pytest.raises(ValueError, match="post_weights shape"):
        mhc.post(update, streams, bad_post)
    bad_mask = replace(state, valid_mask=state.valid_mask[:, :-1])
    with pytest.raises(ValueError, match="valid_mask shape"):
        mhc.post(update, streams, bad_mask)
    with pytest.raises(TypeError, match="MHCState"):
        mhc.post(update, streams, object())  # type: ignore[arg-type]


def test_state_dict_round_trip_and_parameter_count_is_stream_width_independent() -> (
    None
):
    torch.manual_seed(16)
    config = _config(update_scale_init=0.4)
    source = ManifoldConstrainedHyperConnection(config)
    clone = ManifoldConstrainedHyperConnection(config)
    clone.load_state_dict(source.state_dict(), strict=True)
    streams = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, True, False, True], [True, False, True, False]])
    update = torch.randn(2, 1, 6)

    source_projected, source_state = source.pre(streams, mask)
    clone_projected, clone_state = clone.pre(streams, mask)
    source_output = source.post(update, streams, source_state)
    clone_output = clone.post(update, streams, clone_state)

    torch.testing.assert_close(clone_projected, source_projected)
    torch.testing.assert_close(clone_output, source_output)

    wider = ManifoldConstrainedHyperConnection(_config(num_streams=7))
    source_parameter_count = sum(parameter.numel() for parameter in source.parameters())
    wider_parameter_count = sum(parameter.numel() for parameter in wider.parameters())
    assert wider_parameter_count == source_parameter_count

    with pytest.raises(ValueError, match="trailing shape"):
        source.pre(torch.randn(2, 7, 6), torch.ones(2, 7, dtype=torch.bool))


def test_low_precision_inputs_use_float32_coefficients_and_restore_dtype() -> None:
    mhc = ManifoldConstrainedHyperConnection(_config(update_scale_init=0.2))
    streams = torch.randn(2, 4, 6, dtype=torch.bfloat16)
    mask = torch.tensor([[True, True, False, True], [False, False, False, False]])
    update = torch.randn(2, 1, 6, dtype=torch.bfloat16)

    projected, state = mhc.pre(streams, mask)
    output = mhc.post(update, streams, state)

    assert projected.dtype == torch.bfloat16
    assert state.residual_mix.dtype == torch.bfloat16
    assert state.post_weights.dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(projected).all()
    assert torch.isfinite(output).all()
    assert torch.equal(output[1], torch.zeros_like(output[1]))


def test_autocast_does_not_lower_coefficient_or_output_dtype() -> None:
    mhc = ManifoldConstrainedHyperConnection(_config(update_scale_init=0.2))
    streams = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, True, False, True], [False, True, True, False]])
    update = torch.randn(2, 1, 6)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        projected, state = mhc.pre(streams, mask)
        output = mhc.post(update, streams, state)

    assert projected.dtype == torch.float32
    assert state.residual_mix.dtype == torch.float32
    assert state.post_weights.dtype == torch.float32
    assert output.dtype == torch.float32
    assert torch.isfinite(projected).all()
    assert torch.isfinite(output).all()

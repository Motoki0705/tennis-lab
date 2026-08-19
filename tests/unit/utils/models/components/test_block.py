"""Transformer block residual and attention-dispatch contract tests."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import replace
from typing import Literal, cast

import pytest
import torch
from torch import Tensor

from src.utils.models import (
    CompressedSlidingWindowSelfAttention as RootCSWA,
)
from src.utils.models import (
    CSWAConfig as RootCSWAConfig,
)
from src.utils.models.components import (
    CompressedSlidingWindowSelfAttention as ComponentCSWA,
)
from src.utils.models.components import (
    CSWAConfig as ComponentCSWAConfig,
)
from src.utils.models.components.attention import (
    GroupedQuerySelfAttention,
    MultiHeadSelfAttention,
)
from src.utils.models.components.block import TransformerBlock, TransformerBlockConfig
from src.utils.models.components.cswa import (
    CompressedSlidingWindowSelfAttention,
    CSWAConfig,
)
from src.utils.models.components.rope import precompute_freqs_cis

AttentionType = Literal["mha", "gqa", "cswa"]
DenseAttentionType = Literal["mha", "gqa"]
DenseInvocation = Literal["module", "direct", "update"]
InvalidDenseMask = Literal[
    "float",
    "rank_two",
    "query_broadcast",
    "batch_broadcast",
    "wrong_device",
]


def _cswa_config(*, attn_dropout: float = 0.0) -> CSWAConfig:
    return CSWAConfig(
        dim=8,
        n_heads=2,
        head_dim=4,
        rope_dim=4,
        attn_dropout=attn_dropout,
        compression_ratio=2,
        window_radius=1,
        backend="reference",
    )


def _block_config(
    attention_type: AttentionType,
    *,
    n_kv_heads: int | None = None,
    attn_dropout: float = 0.0,
    cswa: CSWAConfig | None = None,
) -> TransformerBlockConfig:
    return TransformerBlockConfig(
        dim=8,
        n_heads=2,
        ffn_dim=16,
        head_dim=4,
        rope_dim=4,
        attn_dropout=attn_dropout,
        attention_type=attention_type,
        n_kv_heads=n_kv_heads,
        rope_base=10_000.0,
        ffn_type="mlp",
        cswa=cswa,
    )


def _dense_inputs() -> tuple[Tensor, Tensor, Tensor]:
    x = torch.randn(2, 5, 8)
    freqs_cis = precompute_freqs_cis(dim=4, seqlen=5)
    attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)
    attn_mask[0, :, -1] = False
    return x, freqs_cis, attn_mask


def _dense_attention_output(
    block: TransformerBlock,
    x: Tensor,
    freqs_cis: Tensor,
    attn_mask: Tensor,
) -> Tensor:
    assert isinstance(block.attn, (MultiHeadSelfAttention, GroupedQuerySelfAttention))
    return cast(
        Tensor,
        block.attn(
            block.attn_norm(x),
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
        ),
    )


def _invoke_dense_block(
    block: TransformerBlock,
    invocation: DenseInvocation,
    x: Tensor,
    freqs_cis: Tensor,
    attn_mask: Tensor,
) -> Tensor:
    if invocation == "module":
        return cast(Tensor, block(x, freqs_cis=freqs_cis, attn_mask=attn_mask))
    if invocation == "direct":
        return block.forward(x, freqs_cis=freqs_cis, attn_mask=attn_mask)
    return block.forward_update(x, freqs_cis=freqs_cis, attn_mask=attn_mask)


def _invalid_dense_mask(case: InvalidDenseMask, x: Tensor) -> Tensor:
    batch_size, sequence_length, _ = x.shape
    if case == "float":
        return torch.ones(batch_size, sequence_length, sequence_length)
    if case == "rank_two":
        return torch.ones(sequence_length, sequence_length, dtype=torch.bool)
    if case == "query_broadcast":
        return torch.ones(batch_size, 1, sequence_length, dtype=torch.bool)
    if case == "batch_broadcast":
        return torch.ones(1, sequence_length, sequence_length, dtype=torch.bool)
    return torch.ones(
        batch_size,
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device="meta",
    )


@pytest.mark.parametrize(("attention_type", "n_kv_heads"), [("mha", None), ("gqa", 1)])
def test_dense_forward_matches_pre_change_oracle(
    attention_type: AttentionType,
    n_kv_heads: int | None,
) -> None:
    torch.manual_seed(753)
    block = TransformerBlock(
        _block_config(attention_type, n_kv_heads=n_kv_heads)
    ).eval()
    x, freqs_cis, attn_mask = _dense_inputs()

    attn_output = _dense_attention_output(block, x, freqs_cis, attn_mask)
    x_attn = x + attn_output
    expected = x_attn + block.ffn(block.ffn_norm(x_attn))
    actual = block(x, freqs_cis=freqs_cis, attn_mask=attn_mask)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(("attention_type", "n_kv_heads"), [("mha", None), ("gqa", 1)])
def test_forward_is_input_plus_residual_free_update(
    attention_type: AttentionType,
    n_kv_heads: int | None,
) -> None:
    block = TransformerBlock(
        _block_config(attention_type, n_kv_heads=n_kv_heads)
    ).eval()
    x, freqs_cis, attn_mask = _dense_inputs()

    update = block.forward_update(
        x,
        freqs_cis=freqs_cis,
        attn_mask=attn_mask,
    )
    output = block(x, freqs_cis=freqs_cis, attn_mask=attn_mask)

    torch.testing.assert_close(output, x + update, atol=0, rtol=0)


@pytest.mark.parametrize(("attention_type", "n_kv_heads"), [("mha", None), ("gqa", 1)])
def test_forward_update_is_attention_plus_ffn_with_attention_residual_input(
    attention_type: AttentionType,
    n_kv_heads: int | None,
) -> None:
    block = TransformerBlock(
        _block_config(attention_type, n_kv_heads=n_kv_heads)
    ).eval()
    x, freqs_cis, attn_mask = _dense_inputs()

    attn_output = _dense_attention_output(block, x, freqs_cis, attn_mask)
    x_attn = x + attn_output
    ffn_output = block.ffn(block.ffn_norm(x_attn))
    update = block.forward_update(
        x,
        freqs_cis=freqs_cis,
        attn_mask=attn_mask,
    )

    torch.testing.assert_close(update, attn_output + ffn_output, atol=0, rtol=0)
    assert not torch.allclose(update, x + attn_output + ffn_output)


def test_added_config_field_preserves_existing_positional_construction() -> None:
    cfg = TransformerBlockConfig(
        8,
        2,
        16,
        4,
        4,
        0.0,
        "mha",
        None,
        10_000.0,
        "mlp",
    )

    assert cfg.cswa is None
    assert isinstance(TransformerBlock(cfg).attn, MultiHeadSelfAttention)


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (_block_config("mha", n_kv_heads=1), "n_kv_heads must be None"),
        (_block_config("mha", cswa=_cswa_config()), "cswa must be None"),
        (_block_config("gqa"), "n_kv_heads must be set"),
        (
            _block_config("gqa", n_kv_heads=1, cswa=_cswa_config()),
            "cswa must be None",
        ),
        (_block_config("cswa"), "cswa must be set"),
        (
            _block_config("cswa", n_kv_heads=1, cswa=_cswa_config()),
            "n_kv_heads must be None",
        ),
    ],
)
def test_attention_config_matrix_rejects_conflicting_fields(
    cfg: TransformerBlockConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TransformerBlock(cfg)


@pytest.mark.parametrize(
    ("cfg", "field"),
    [
        (replace(_block_config("cswa", cswa=_cswa_config()), dim=9), "dim"),
        (replace(_block_config("cswa", cswa=_cswa_config()), n_heads=4), "n_heads"),
        (
            replace(_block_config("cswa", cswa=_cswa_config()), head_dim=8),
            "head_dim",
        ),
        (replace(_block_config("cswa", cswa=_cswa_config()), rope_dim=2), "rope_dim"),
        (
            replace(_block_config("cswa", cswa=_cswa_config()), attn_dropout=0.25),
            "attn_dropout",
        ),
    ],
)
def test_cswa_duplicate_config_values_must_match_parent(
    cfg: TransformerBlockConfig,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=rf"cswa\.{field} must match"):
        TransformerBlock(cfg)


def test_unknown_attention_type_is_rejected() -> None:
    cfg = _block_config(cast(AttentionType, "mla"))
    with pytest.raises(ValueError, match="Unsupported attention_type=mla"):
        TransformerBlock(cfg)


@pytest.mark.parametrize("method_name", ["forward", "forward_update"])
def test_dense_runtime_requires_attn_mask_and_prohibits_state_valid(
    method_name: str,
) -> None:
    block = TransformerBlock(_block_config("mha"))
    x, freqs_cis, attn_mask = _dense_inputs()
    method = block if method_name == "forward" else block.forward_update

    with pytest.raises(ValueError, match="attn_mask is required"):
        method(x, freqs_cis=freqs_cis)
    with pytest.raises(ValueError, match="state_valid is prohibited"):
        method(
            x,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            state_valid=torch.ones(2, 5, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    ("attention_type", "n_kv_heads"), [("mha", None), ("gqa", 1)]
)
@pytest.mark.parametrize("invocation", ["module", "direct", "update"])
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("float", "dtype torch.bool"),
        ("rank_two", "exact shape"),
        ("query_broadcast", "exact shape"),
        ("batch_broadcast", "exact shape"),
        ("wrong_device", "same device as x"),
    ],
)
def test_dense_boundary_rejects_noncanonical_full_masks(
    attention_type: DenseAttentionType,
    n_kv_heads: int | None,
    invocation: DenseInvocation,
    case: InvalidDenseMask,
    message: str,
) -> None:
    block = TransformerBlock(
        _block_config(attention_type, n_kv_heads=n_kv_heads)
    ).eval()
    x, freqs_cis, _ = _dense_inputs()
    attn_mask = _invalid_dense_mask(case, x)

    with pytest.raises(ValueError, match=message):
        _invoke_dense_block(block, invocation, x, freqs_cis, attn_mask)


@pytest.mark.parametrize(
    ("attention_type", "n_kv_heads"), [("mha", None), ("gqa", 1)]
)
@pytest.mark.parametrize("invocation", ["module", "direct", "update"])
def test_dense_boundary_accepts_canonical_boolean_full_masks(
    attention_type: DenseAttentionType,
    n_kv_heads: int | None,
    invocation: DenseInvocation,
) -> None:
    block = TransformerBlock(
        _block_config(attention_type, n_kv_heads=n_kv_heads)
    ).eval()
    x, freqs_cis, attn_mask = _dense_inputs()

    output = _invoke_dense_block(block, invocation, x, freqs_cis, attn_mask)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("method_name", ["forward", "forward_update"])
def test_cswa_runtime_requires_state_valid_and_prohibits_attn_mask(
    method_name: str,
) -> None:
    block = TransformerBlock(_block_config("cswa", cswa=_cswa_config()))
    x, freqs_cis, attn_mask = _dense_inputs()
    method = block if method_name == "forward" else block.forward_update

    with pytest.raises(ValueError, match="state_valid is required"):
        method(x, freqs_cis=freqs_cis)
    with pytest.raises(ValueError, match="attn_mask is prohibited"):
        method(
            x,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            state_valid=torch.ones(2, 5, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    (
        "attention_type",
        "n_kv_heads",
        "case",
        "include_attn_mask",
        "include_state_valid",
        "message",
    ),
    [
        ("mha", None, "missing", False, False, "attn_mask is required"),
        ("mha", None, "prohibited", False, True, "state_valid is prohibited"),
        ("mha", None, "conflicting", True, True, "state_valid is prohibited"),
        ("gqa", 1, "missing", False, False, "attn_mask is required"),
        ("gqa", 1, "prohibited", False, True, "state_valid is prohibited"),
        ("gqa", 1, "conflicting", True, True, "state_valid is prohibited"),
        ("cswa", None, "missing", False, False, "state_valid is required"),
        ("cswa", None, "prohibited", True, False, "attn_mask is prohibited"),
        ("cswa", None, "conflicting", True, True, "attn_mask is prohibited"),
    ],
)
def test_direct_forward_matches_module_call_for_every_invalid_mask_combination(
    attention_type: AttentionType,
    n_kv_heads: int | None,
    case: str,
    include_attn_mask: bool,
    include_state_valid: bool,
    message: str,
) -> None:
    del case
    cswa = _cswa_config() if attention_type == "cswa" else None
    block = TransformerBlock(
        _block_config(attention_type, n_kv_heads=n_kv_heads, cswa=cswa)
    )
    x, freqs_cis, attn_mask = _dense_inputs()
    state_valid = torch.ones(2, 5, dtype=torch.bool)
    kwargs: dict[str, Tensor] = {"freqs_cis": freqs_cis}
    if include_attn_mask:
        kwargs["attn_mask"] = attn_mask
    if include_state_valid:
        kwargs["state_valid"] = state_valid

    with pytest.raises(ValueError, match=message) as module_error:
        block(x, **kwargs)
    with pytest.raises(ValueError, match=message) as direct_error:
        block.forward(x, **kwargs)
    assert str(direct_error.value) == str(module_error.value)


@pytest.mark.parametrize(
    ("attention_type", "n_kv_heads"),
    [("mha", None), ("gqa", 1), ("cswa", None)],
)
def test_direct_forward_matches_module_call_for_valid_arguments(
    attention_type: AttentionType,
    n_kv_heads: int | None,
) -> None:
    cswa = _cswa_config() if attention_type == "cswa" else None
    block = TransformerBlock(
        _block_config(attention_type, n_kv_heads=n_kv_heads, cswa=cswa)
    ).eval()
    x, freqs_cis, attn_mask = _dense_inputs()

    if attention_type == "cswa":
        state_valid = torch.ones(2, 5, dtype=torch.bool)
        expected = block(x, freqs_cis=freqs_cis, state_valid=state_valid)
        actual = block.forward(x, freqs_cis=freqs_cis, state_valid=state_valid)
    else:
        expected = block(x, freqs_cis=freqs_cis, attn_mask=attn_mask)
        actual = block.forward(x, freqs_cis=freqs_cis, attn_mask=attn_mask)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_forward_decorator_preserves_explicit_public_signatures() -> None:
    wrapped_forward = TransformerBlock.forward
    unwrapped_forward = inspect.unwrap(wrapped_forward)
    assert wrapped_forward is not unwrapped_forward

    expected_unbound = inspect.signature(unwrapped_forward, follow_wrapped=False)
    assert inspect.signature(wrapped_forward) == expected_unbound
    assert (
        inspect.signature(wrapped_forward, follow_wrapped=False) == expected_unbound
    )

    block = TransformerBlock(_block_config("mha"))
    bound_unwrapped = unwrapped_forward.__get__(block, TransformerBlock)
    expected_bound = inspect.signature(bound_unwrapped, follow_wrapped=False)
    assert inspect.signature(block.forward) == expected_bound
    assert inspect.signature(block.forward, follow_wrapped=False) == expected_bound
    for parameter_name in ("freqs_cis", "attn_mask", "state_valid"):
        assert (
            expected_bound.parameters[parameter_name].kind
            is inspect.Parameter.KEYWORD_ONLY
        )


@pytest.mark.parametrize("method_name", ["forward", "forward_update"])
def test_public_block_methods_reject_keyword_only_arguments_passed_positionally(
    method_name: str,
) -> None:
    block = TransformerBlock(_block_config("mha"))
    x, freqs_cis, attn_mask = _dense_inputs()
    method = cast(Callable[..., Tensor], getattr(block, method_name))

    with pytest.raises(TypeError, match="positional arguments"):
        method(x, freqs_cis, attn_mask)


@pytest.mark.parametrize("method_name", ["forward", "forward_update"])
def test_public_block_attention_arguments_remain_keyword_only(method_name: str) -> None:
    block = TransformerBlock(_block_config("mha"))
    signature = inspect.signature(getattr(block, method_name))

    assert signature.parameters["x"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for parameter_name in ("freqs_cis", "attn_mask", "state_valid"):
        assert (
            signature.parameters[parameter_name].kind
            is inspect.Parameter.KEYWORD_ONLY
        )


@pytest.mark.parametrize(
    ("attention_type", "n_kv_heads"),
    [("mha", None), ("gqa", 1), ("cswa", None)],
)
def test_direct_forward_matches_module_call_values_gradients_and_state(
    attention_type: AttentionType,
    n_kv_heads: int | None,
) -> None:
    torch.manual_seed(753)
    cswa = _cswa_config() if attention_type == "cswa" else None
    cfg = _block_config(attention_type, n_kv_heads=n_kv_heads, cswa=cswa)
    module_block = TransformerBlock(cfg).eval()
    direct_block = TransformerBlock(cfg).eval()
    direct_block.load_state_dict(module_block.state_dict(), strict=True)
    initial_state = {
        name: tensor.detach().clone() for name, tensor in module_block.state_dict().items()
    }
    module_x, freqs_cis, attn_mask = _dense_inputs()
    module_x.requires_grad_()
    direct_x = module_x.detach().clone().requires_grad_()
    kwargs: dict[str, Tensor] = {"freqs_cis": freqs_cis}
    if attention_type == "cswa":
        kwargs["state_valid"] = torch.tensor(
            [[True, True, False, True, True], [True, False, True, True, True]]
        )
    else:
        kwargs["attn_mask"] = attn_mask

    module_output = module_block(module_x, **kwargs)
    direct_output = direct_block.forward(direct_x, **kwargs)
    upstream = torch.randn_like(module_output)
    module_gradients = torch.autograd.grad(
        (module_output * upstream).sum(),
        (module_x, *module_block.parameters()),
    )
    direct_gradients = torch.autograd.grad(
        (direct_output * upstream).sum(),
        (direct_x, *direct_block.parameters()),
    )

    torch.testing.assert_close(module_output, direct_output, atol=0, rtol=0)
    assert len(module_gradients) == len(direct_gradients)
    for module_gradient, direct_gradient in zip(
        module_gradients, direct_gradients, strict=True
    ):
        torch.testing.assert_close(module_gradient, direct_gradient, atol=0, rtol=0)
    assert set(module_block.state_dict()) == set(direct_block.state_dict()) == set(
        initial_state
    )
    for name, initial_tensor in initial_state.items():
        torch.testing.assert_close(module_block.state_dict()[name], initial_tensor)
        torch.testing.assert_close(direct_block.state_dict()[name], initial_tensor)


@pytest.mark.parametrize(
    ("attention_type", "n_kv_heads", "expected_keys"),
    [
        (
            "mha",
            None,
            {
                "attn_norm.weight",
                "attn.wqkv.weight",
                "attn.wo.weight",
                "ffn_norm.weight",
                "ffn.fc1.weight",
                "ffn.fc2.weight",
            },
        ),
        (
            "gqa",
            1,
            {
                "attn_norm.weight",
                "attn.wq.weight",
                "attn.wk.weight",
                "attn.wv.weight",
                "attn.wo.weight",
                "ffn_norm.weight",
                "ffn.fc1.weight",
                "ffn.fc2.weight",
            },
        ),
    ],
)
def test_dense_state_dict_keys_and_gradients_are_preserved(
    attention_type: AttentionType,
    n_kv_heads: int | None,
    expected_keys: set[str],
) -> None:
    source = TransformerBlock(_block_config(attention_type, n_kv_heads=n_kv_heads))
    clone = TransformerBlock(_block_config(attention_type, n_kv_heads=n_kv_heads))
    clone.load_state_dict(source.state_dict(), strict=True)
    assert set(source.state_dict()) == expected_keys
    x, freqs_cis, attn_mask = _dense_inputs()
    x.requires_grad_()
    oracle_x = x.detach().clone().requires_grad_()

    output = source(x, freqs_cis=freqs_cis, attn_mask=attn_mask)
    oracle_attn = _dense_attention_output(clone, oracle_x, freqs_cis, attn_mask)
    oracle_x_attn = oracle_x + oracle_attn
    oracle_output = oracle_x_attn + clone.ffn(clone.ffn_norm(oracle_x_attn))
    upstream = torch.randn_like(output)
    gradients = torch.autograd.grad(
        (output * upstream).sum(),
        (x, *source.parameters()),
    )
    oracle_gradients = torch.autograd.grad(
        (oracle_output * upstream).sum(),
        (oracle_x, *clone.parameters()),
    )

    torch.testing.assert_close(output, oracle_output, atol=1e-6, rtol=1e-6)
    assert len(gradients) == len(oracle_gradients)
    for gradient, oracle_gradient in zip(gradients, oracle_gradients, strict=True):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, oracle_gradient, atol=1e-6, rtol=1e-5)


def test_cswa_forward_update_shape_finite_gradient_and_round_trip() -> None:
    cfg = _block_config("cswa", cswa=_cswa_config())
    source = TransformerBlock(cfg)
    clone = TransformerBlock(cfg)
    clone.load_state_dict(source.state_dict(), strict=True)
    x = torch.randn(2, 5, 8, requires_grad=True)
    freqs_cis = precompute_freqs_cis(dim=4, seqlen=5)
    state_valid = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, True]]
    )

    update = source.forward_update(
        x,
        freqs_cis=freqs_cis,
        state_valid=state_valid,
    )
    output = source(x, freqs_cis=freqs_cis, state_valid=state_valid)
    clone_output = clone(x.detach(), freqs_cis=freqs_cis, state_valid=state_valid)
    output.square().mean().backward()

    assert update.shape == x.shape
    assert output.shape == x.shape
    assert torch.isfinite(update).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in source.parameters()
    )
    torch.testing.assert_close(output.detach(), clone_output)


def test_cswa_block_preserves_dropout_train_eval_semantics() -> None:
    cfg = _block_config(
        "cswa",
        attn_dropout=0.5,
        cswa=_cswa_config(attn_dropout=0.5),
    )
    block = TransformerBlock(cfg)
    x = torch.randn(2, 8, 8)
    freqs_cis = precompute_freqs_cis(dim=4, seqlen=8)
    state_valid = torch.ones(2, 8, dtype=torch.bool)

    block.eval()
    torch.manual_seed(1)
    eval_first = block(x, freqs_cis=freqs_cis, state_valid=state_valid)
    torch.manual_seed(2)
    eval_second = block(x, freqs_cis=freqs_cis, state_valid=state_valid)
    block.train()
    torch.manual_seed(3)
    train_first = block(x, freqs_cis=freqs_cis, state_valid=state_valid)
    torch.manual_seed(3)
    train_second = block(x, freqs_cis=freqs_cis, state_valid=state_valid)

    torch.testing.assert_close(eval_first, eval_second, atol=0, rtol=0)
    torch.testing.assert_close(train_first, train_second, atol=0, rtol=0)
    assert not torch.allclose(train_first, eval_first)


def test_cswa_public_exports_are_minimal_high_level_symbols() -> None:
    assert ComponentCSWAConfig is RootCSWAConfig is CSWAConfig
    assert ComponentCSWA is RootCSWA is CompressedSlidingWindowSelfAttention

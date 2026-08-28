"""Formula, stability, and block-dispatch tests for dense FFN variants."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.utils.models.components.block import (
    CrossAttnBlock,
    CrossAttnBlockConfig,
    TransformerBlock,
    TransformerBlockConfig,
)
from src.utils.models.components.ffn_layers import (
    MLP,
    DeepSeekV4SwiGLU,
    FFNType,
    GPTOSSSwiGLU,
    KimiK3SiTUGLU,
    SwiGLU,
    build_ffn,
)

GatedFFN = SwiGLU | KimiK3SiTUGLU | DeepSeekV4SwiGLU | GPTOSSSwiGLU
ModuleFactory = Callable[..., nn.Module]


def _set_scalar_identity(module: GatedFFN) -> None:
    with torch.no_grad():
        module.w1.weight.fill_(1.0)
        module.w2.weight.fill_(1.0)
        module.w3.weight.fill_(1.0)


def _transformer_config(ffn_type: FFNType) -> TransformerBlockConfig:
    return TransformerBlockConfig(
        dim=8,
        n_heads=2,
        ffn_dim=16,
        head_dim=4,
        rope_dim=4,
        attn_dropout=0.0,
        attention_type="mha",
        n_kv_heads=None,
        rope_base=10_000.0,
        ffn_type=ffn_type,
    )


def _cross_attention_config(ffn_type: FFNType) -> CrossAttnBlockConfig:
    return CrossAttnBlockConfig(
        dim=8,
        n_heads=2,
        ffn_dim=16,
        head_dim=4,
        rope_dim=4,
        attn_dropout=0.0,
        ffn_type=ffn_type,
    )


def test_kimi_k3_situglu_matches_published_formula() -> None:
    module = KimiK3SiTUGLU(dim=1, ffn_dim=1)
    _set_scalar_identity(module)
    x = torch.tensor([[-100.0], [-2.0], [0.0], [2.0], [100.0]])

    expected = (
        4.0
        * torch.tanh(x / 4.0)
        * torch.sigmoid(x)
        * 25.0
        * torch.tanh(x / 25.0)
    )

    torch.testing.assert_close(module(x), expected)


def test_deepseek_v4_swiglu_matches_asymmetric_clamping() -> None:
    module = DeepSeekV4SwiGLU(dim=1, ffn_dim=1)
    _set_scalar_identity(module)
    x = torch.tensor([[-100.0], [-2.0], [0.0], [2.0], [100.0]])

    expected = F.silu(x.clamp(max=10.0)) * x.clamp(min=-10.0, max=10.0)

    torch.testing.assert_close(module(x), expected)


def test_gpt_oss_swiglu_matches_clipped_shifted_formula() -> None:
    module = GPTOSSSwiGLU(dim=1, ffn_dim=1)
    _set_scalar_identity(module)
    x = torch.tensor([[-100.0], [-2.0], [0.0], [2.0], [100.0]])
    bounded_gate = x.clamp(max=7.0)
    bounded_up = x.clamp(min=-7.0, max=7.0)

    expected = (
        bounded_gate
        * torch.sigmoid(1.702 * bounded_gate)
        * (bounded_up + 1.0)
    )

    torch.testing.assert_close(module(x), expected)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (KimiK3SiTUGLU, {"beta_gate": 0.0}),
        (KimiK3SiTUGLU, {"beta_up": float("inf")}),
        (DeepSeekV4SwiGLU, {"limit": -1.0}),
        (GPTOSSSwiGLU, {"alpha": float("nan")}),
        (GPTOSSSwiGLU, {"limit": 0.0}),
    ],
)
def test_variant_hyperparameters_must_be_positive_and_finite(
    factory: ModuleFactory,
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="finite and greater than zero"):
        factory(4, 8, **kwargs)


@pytest.mark.parametrize(
    ("ffn_type", "expected_type"),
    [
        ("swiglu", SwiGLU),
        ("mlp", MLP),
        ("kimi_k3_situglu", KimiK3SiTUGLU),
        ("deepseek_v4_swiglu", DeepSeekV4SwiGLU),
        ("gpt_oss_swiglu", GPTOSSSwiGLU),
    ],
)
def test_build_ffn_selects_exact_variant(
    ffn_type: FFNType,
    expected_type: type[nn.Module],
) -> None:
    module = build_ffn(ffn_type=ffn_type, dim=8, ffn_dim=16)

    assert type(module) is expected_type


@pytest.mark.parametrize(
    "ffn_type",
    [
        "swiglu",
        "mlp",
        "kimi_k3_situglu",
        "deepseek_v4_swiglu",
        "gpt_oss_swiglu",
    ],
)
def test_all_variants_preserve_shape_and_finite_gradients(ffn_type: FFNType) -> None:
    torch.manual_seed(818)
    module = build_ffn(ffn_type=ffn_type, dim=8, ffn_dim=16)
    x = torch.randn(2, 5, 8, requires_grad=True)

    output = module(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    ("ffn_type", "expected_type"),
    [
        ("swiglu", SwiGLU),
        ("mlp", MLP),
        ("kimi_k3_situglu", KimiK3SiTUGLU),
        ("deepseek_v4_swiglu", DeepSeekV4SwiGLU),
        ("gpt_oss_swiglu", GPTOSSSwiGLU),
    ],
)
def test_transformer_and_cross_attention_blocks_dispatch_same_variant(
    ffn_type: FFNType,
    expected_type: type[nn.Module],
) -> None:
    transformer = TransformerBlock(_transformer_config(ffn_type))
    cross_attention = CrossAttnBlock(_cross_attention_config(ffn_type))

    assert type(transformer.ffn) is expected_type
    assert type(cross_attention.ffn) is expected_type


@pytest.mark.parametrize(
    "factory",
    [SwiGLU, KimiK3SiTUGLU, DeepSeekV4SwiGLU, GPTOSSSwiGLU],
)
def test_gated_variants_preserve_legacy_projection_state_keys(
    factory: ModuleFactory,
) -> None:
    module = factory(8, 16)

    assert set(module.state_dict()) == {"w1.weight", "w2.weight", "w3.weight"}


def test_unknown_ffn_type_is_rejected_by_factory_and_blocks() -> None:
    unknown = cast(FFNType, "unknown")

    with pytest.raises(ValueError, match="Unsupported ffn_type=unknown"):
        build_ffn(ffn_type=unknown, dim=8, ffn_dim=16)
    with pytest.raises(ValueError, match="Unsupported ffn_type=unknown"):
        TransformerBlock(_transformer_config(unknown))
    with pytest.raises(ValueError, match="Unsupported ffn_type=unknown"):
        CrossAttnBlock(_cross_attention_config(unknown))

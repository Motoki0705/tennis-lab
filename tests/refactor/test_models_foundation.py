"""Unit tests for the shared models-foundation modules.

These tests are intentionally dependency-light: they import only the shared
``src.utils.models`` modules and avoid pulling in heavy task packages.
"""

from __future__ import annotations

import torch

from src.utils.models.architectures import (
    TransformerSequenceDiscriminator,
    build_trajectory_discriminator,
)
from src.utils.models.heads import MLPHead
from src.utils.models.transformer_utils import resolve_rope_bases


def test_mlp_head_output_shape() -> None:
    head = MLPHead(input_dim=16, hidden_dim=8, output_dim=3, num_layers=2, dropout=0.0)
    head.eval()
    x = torch.randn(4, 16)
    out = head(x)
    assert out.shape == (4, 3)


def test_mlp_head_state_dict_keys() -> None:
    head = MLPHead(input_dim=16, hidden_dim=8, output_dim=3, num_layers=2, dropout=0.0)
    keys = set(head.state_dict().keys())
    # 2 hidden blocks => Linear at index 0 and 4, LayerNorm at 1 and 5,
    # final Linear at index 8. Keys must be under the ``mlp`` attribute.
    assert "mlp.0.weight" in keys
    assert "mlp.0.bias" in keys
    assert "mlp.1.weight" in keys  # LayerNorm
    assert "mlp.8.weight" in keys  # final Linear
    assert all(key.startswith("mlp.") for key in keys)


def test_resolve_rope_bases_fallback_two_tuple() -> None:
    # No type axis => 2-tuple; None falls back to rope_theta.
    bases = resolve_rope_bases(10000.0, None, None)
    assert bases == (10000.0, 10000.0)
    assert all(isinstance(b, float) for b in bases)


def test_resolve_rope_bases_two_tuple_overrides() -> None:
    bases = resolve_rope_bases(10000.0, 5000.0, None)
    assert bases == (5000.0, 10000.0)


def test_resolve_rope_bases_three_tuple() -> None:
    # type axis provided => 3-tuple.
    bases = resolve_rope_bases(10000.0, None, None, 100.0)
    assert bases == (10000.0, 10000.0, 100.0)
    bases2 = resolve_rope_bases(10000.0, 1.0, 2.0, 3.0)
    assert bases2 == (1.0, 2.0, 3.0)


def test_build_trajectory_discriminator_returns_instance_and_forward() -> None:
    disc_cfg = {
        "hidden_dim": 16,
        "num_layers": 2,
        "num_heads": 2,
        "dropout": 0.0,
        "max_seq_len": 12,
    }
    disc = build_trajectory_discriminator(
        input_dim=3,
        disc_cfg=disc_cfg,
        default_max_seq_len=120,
    )
    assert isinstance(disc, TransformerSequenceDiscriminator)
    assert disc.input_dim == 3

    disc.eval()
    x = torch.randn(5, 7, 3)  # (B, T, input_dim)
    score = disc(x)
    assert score.shape == (5,)


def test_build_trajectory_discriminator_default_max_seq_len() -> None:
    # When disc_cfg lacks max_seq_len, the default is used.
    disc = build_trajectory_discriminator(
        input_dim=2,
        disc_cfg={"hidden_dim": 16, "num_heads": 2},
        default_max_seq_len=8,
    )
    assert disc.max_seq_len == 8
    assert disc.input_dim == 2

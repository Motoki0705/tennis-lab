"""Tests for reference MoE compute and construction-time backend selection."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.utils.models.components.moe import MoEConfig, MoELayer
from src.utils.models.components.ops.moe import api as moe_api


def _config(*, use_cuda_ops: bool = False) -> MoEConfig:
    return MoEConfig(
        dim=4,
        num_experts=2,
        top_k=1,
        ffn_dim=8,
        ffn_type="mlp",
        router_bias=False,
        router_jitter_noise=0.0,
        normalize_router_weights=True,
        capacity_factor=None,
        drop_policy="none",
        use_cuda_ops=use_cuda_ops,
    )


def test_reference_moe_valid_compute_preserves_shape_and_gradients() -> None:
    layer = MoELayer(_config()).eval()
    hidden_states = torch.randn(2, 3, 4, requires_grad=True)

    output = layer(hidden_states)
    output.square().mean().backward()

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
    assert hidden_states.grad is not None


def test_cuda_backend_is_resolved_during_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> None:
        raise RuntimeError("extension unavailable at composition")

    monkeypatch.setattr(moe_api, "require_moe_cuda_extension", unavailable)

    with pytest.raises(RuntimeError, match="at composition"):
        MoELayer(_config(use_cuda_ops=True))


def test_capacity_policy_is_validated_during_construction() -> None:
    config = _config()
    invalid = replace(config, drop_policy="capacity", capacity_factor=None)

    with pytest.raises(ValueError, match="capacity_factor is required"):
        MoELayer(invalid)

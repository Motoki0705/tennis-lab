"""Tests for :mod:`src.utils.models.attention_extraction`.

The extractor reconstructs the softmax attention probabilities that SDPA hides.
These tests check that reconstruction against an explicit SDPA forward pass on a
tiny fake block (no DINOv3 weights needed), plus the rollout/flow helpers that
consume the captured maps.
"""

from __future__ import annotations

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.utils.models.attention_extraction import (
    AttentionExtractor,
    find_attention_modules,
    is_sdpa_self_attention,
    iter_attention_maps,
)


class _FakeSelfAttention(nn.Module):
    """Minimal SDPA self-attention mirroring the DINOv3 block contract."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, rope: object = None) -> Tensor:  # noqa: ARG002
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        )
        query, key, value = (t.transpose(1, 2) for t in torch.unbind(qkv, 2))
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(attended)


class _FakeViT(nn.Module):
    def __init__(self, dim: int = 16, num_heads: int = 4, depth: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.ModuleDict({"attn": _FakeSelfAttention(dim, num_heads)})
            for _ in range(depth)
        )

    def forward_features(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = x + block["attn"](x)
        return x


def _manual_probs(attn: _FakeSelfAttention, x: Tensor) -> Tensor:
    batch, tokens, channels = x.shape
    qkv = attn.qkv(x).reshape(
        batch, tokens, 3, attn.num_heads, channels // attn.num_heads
    )
    query, key, _ = (t.transpose(1, 2) for t in torch.unbind(qkv, 2))
    scores = (query.float() @ key.float().transpose(-2, -1)) * attn.scale
    return scores.softmax(dim=-1)


class TestDiscovery:
    def test_predicate_matches_fake_attention(self) -> None:
        attn = _FakeSelfAttention(16, 4)
        assert is_sdpa_self_attention(attn)
        assert not is_sdpa_self_attention(nn.Linear(4, 4))

    def test_find_modules_in_forward_order(self) -> None:
        model = _FakeViT(depth=3)
        found = find_attention_modules(model)
        assert [name for name, _ in found] == [
            "blocks.0.attn",
            "blocks.1.attn",
            "blocks.2.attn",
        ]


class TestExtractor:
    def test_captures_one_map_per_layer(self) -> None:
        torch.manual_seed(0)
        model = _FakeViT(dim=16, num_heads=4, depth=3).eval()
        x = torch.randn(2, 7, 16)
        with AttentionExtractor(model) as extractor:
            model.forward_features(x)
        maps = extractor.attentions
        assert len(maps) == 3
        assert maps[0].shape == (2, 4, 7, 7)

    def test_probs_are_row_stochastic(self) -> None:
        model = _FakeViT().eval()
        x = torch.randn(1, 5, 16)
        with AttentionExtractor(model) as extractor:
            model.forward_features(x)
        sums = extractor.attentions[0].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_reconstructed_probs_match_explicit_sdpa(self) -> None:
        torch.manual_seed(1)
        model = _FakeViT(dim=16, num_heads=4, depth=2).eval()
        x = torch.randn(1, 6, 16)
        with AttentionExtractor(model) as extractor:
            model.forward_features(x)
        expected = _manual_probs(model.blocks[0]["attn"], x)
        assert torch.allclose(extractor.attentions[0], expected, atol=1e-6)

    def test_fuse_heads_averages_over_heads(self) -> None:
        model = _FakeViT(dim=16, num_heads=4).eval()
        x = torch.randn(1, 5, 16)
        with AttentionExtractor(model, fuse_heads=True) as extractor:
            model.forward_features(x)
        assert extractor.attentions[0].shape == (1, 5, 5)

    def test_hooks_removed_on_exit(self) -> None:
        model = _FakeViT().eval()
        attn = model.blocks[0]["attn"]
        with AttentionExtractor(model):
            assert attn._forward_pre_hooks
        assert not attn._forward_pre_hooks

    def test_raises_without_attention_modules(self) -> None:
        with pytest.raises(ValueError, match="No SDPA self-attention"):
            AttentionExtractor(nn.Sequential(nn.Linear(4, 4)))

    def test_iter_attention_maps_helper(self) -> None:
        model = _FakeViT(depth=3).eval()
        x = torch.randn(1, 5, 16)
        maps = list(iter_attention_maps(model, x))
        assert len(maps) == 3


# --------------------------------------------------------------------------- #
# Rollout / flow propagation (re-implemented inline to keep the util test
# self-contained; mirrors scripts/analysis/models/attention_maps.py).
# --------------------------------------------------------------------------- #
def _residual(a: Tensor) -> Tensor:
    aug = a + torch.eye(a.size(-1))
    return aug / aug.sum(dim=-1, keepdim=True)


def _rollout(attentions: list[Tensor], n_special: int) -> Tensor:
    result: Tensor | None = None
    for a in attentions:
        layer = _residual(a)
        result = layer if result is None else layer @ result
    assert result is not None
    return result[0, n_special:]


def _flow(attentions: list[Tensor], n_special: int) -> Tensor:
    layers = [_residual(a).numpy() for a in attentions]
    depth, n = len(layers), layers[0].shape[0]
    graph = nx.DiGraph()
    for li, w in enumerate(layers):
        for dst in range(n):
            for src in range(n):
                graph.add_edge(f"{li}_{src}", f"{li + 1}_{dst}", capacity=float(w[dst, src]))
    sink = f"{depth}_0"
    flows = np.array(
        [nx.maximum_flow_value(graph, f"0_{t}", sink) for t in range(n)],
        dtype=np.float32,
    )
    return torch.from_numpy(flows[n_special:])


class TestPropagation:
    def _attentions(self, n: int = 6, depth: int = 3) -> list[Tensor]:
        torch.manual_seed(2)
        return [torch.softmax(torch.randn(n, n), dim=-1) for _ in range(depth)]

    def test_rollout_shape_and_residual_identity(self) -> None:
        attentions = self._attentions(n=6)
        cls = _rollout(attentions, n_special=1)
        assert cls.shape == (5,)
        # An identity attention stack rolls out to a uniform-ish residual map.
        identity = [torch.eye(6) for _ in range(3)]
        cls_id = _rollout(identity, n_special=1)
        assert torch.allclose(cls_id, torch.zeros_like(cls_id), atol=1e-6)

    def test_flow_is_nonnegative_and_matches_patch_count(self) -> None:
        attentions = self._attentions(n=6, depth=2)
        flow = _flow(attentions, n_special=1)
        assert flow.shape == (5,)
        assert (flow >= 0).all()

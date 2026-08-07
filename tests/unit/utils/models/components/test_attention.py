"""Valid-compute tests for boundary-normalized shared attention inputs."""

from __future__ import annotations

import torch

from src.utils.models.components.attention import (
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)
from src.utils.models.components.block import CrossAttnBlock, CrossAttnBlockConfig
from src.utils.models.components.rope import (
    RotaryFrequencyComputer,
    precompute_freqs_cis,
    precompute_freqs_cis_nd,
)


def test_rotary_frequency_computer_prepares_canonical_broadcast_shape() -> None:
    computer = RotaryFrequencyComputer(
        dim=4,
        base=(10_000.0, 1_000.0),
        n_axes=2,
    )
    positions = torch.zeros(2, 3, 2, dtype=torch.long)

    frequencies = computer(positions)

    assert frequencies.shape == (2, 3, 1, 2)
    assert frequencies.is_complex()


def test_self_attention_accepts_canonical_batch_keep_mask() -> None:
    attention = MultiHeadSelfAttention(
        dim=8,
        n_heads=2,
        head_dim=4,
        rope_dim=4,
        attn_dropout=0.0,
        bias=False,
    ).eval()
    tokens = torch.randn(2, 3, 8)
    frequencies = precompute_freqs_cis(dim=4, seqlen=3)
    keep_mask = torch.tensor(
        [
            [[True, True, False], [True, True, False], [True, True, True]],
            [[True, False, False], [True, True, False], [True, True, True]],
        ]
    )

    with torch.no_grad():
        output = attention(tokens, freqs_cis=frequencies, attn_mask=keep_mask)

    assert output.shape == tokens.shape
    assert torch.isfinite(output).all()


def test_cross_attention_accepts_batched_rotary_frequencies() -> None:
    attention = MultiHeadCrossAttention(
        dim=8,
        n_heads=2,
        head_dim=4,
        rope_dim=4,
        attn_dropout=0.0,
        bias=False,
    ).eval()
    queries = torch.randn(2, 3, 8)
    context = torch.randn(2, 4, 8)
    query_positions = torch.arange(3).view(1, 3, 1).expand(2, -1, -1)
    context_positions = torch.arange(4).view(1, 4, 1).expand(2, -1, -1)
    query_frequencies = precompute_freqs_cis_nd(dim=4, pos=query_positions)
    context_frequencies = precompute_freqs_cis_nd(dim=4, pos=context_positions)
    keep_mask = torch.ones(2, 3, 4, dtype=torch.bool)

    with torch.no_grad():
        output = attention(
            queries,
            context,
            freqs_q_cis=query_frequencies,
            freqs_k_cis=context_frequencies,
            attn_mask=keep_mask,
        )

    assert output.shape == queries.shape
    assert torch.isfinite(output).all()


def test_cross_attention_block_consumes_prepared_mask() -> None:
    block = CrossAttnBlock(
        CrossAttnBlockConfig(
            dim=8,
            n_heads=2,
            ffn_dim=16,
            head_dim=4,
            rope_dim=4,
            attn_dropout=0.0,
            ffn_type="mlp",
        )
    ).eval()
    queries = torch.randn(2, 3, 8)
    context = torch.randn(2, 4, 8)
    query_frequencies = precompute_freqs_cis(dim=4, seqlen=3)
    context_frequencies = precompute_freqs_cis(dim=4, seqlen=4)
    keep_mask = torch.ones(2, 3, 4, dtype=torch.bool)

    with torch.no_grad():
        output = block(
            queries,
            context,
            attn_mask=keep_mask,
            freqs_q_cis=query_frequencies,
            freqs_k_cis=context_frequencies,
        )

    assert output.shape == queries.shape
    assert torch.isfinite(output).all()

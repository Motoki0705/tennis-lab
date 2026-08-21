"""Common Transformer components (pure PyTorch, DeepSeek-style).

This package provides reusable building blocks used across tasks:

- Attention: `MultiHeadSelfAttention`, `MultiHeadCrossAttention`
- Norm: `RMSNorm`, `LayerNorm`
- RoPE: 1D (`precompute_freqs_cis`) and interleaved N-D (`precompute_freqs_cis_nd`, `apply_rotary_emb`)
- FFN/MoE: `SwiGLU`, `MLP`, `default_ffn_dim`, `TopKRouter`, `MoELayer`
- Blocks: `TransformerBlock`, `TransformerBlockConfig`, `CrossAttnBlockConfig`, `CrossAttnBlock`
- Compressed temporal attention: `CSWAConfig`, `CompressedSlidingWindowSelfAttention`

Note:
This repository previously had a separate "unified MHA/GQA/MLA" implementation.
Strategy A treats the DeepSeek-style implementation as canonical.
"""

from src.utils.models.components.attention import (
    GroupedQuerySelfAttention,
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)
from src.utils.models.components.block import (
    CrossAttnBlock,
    CrossAttnBlockConfig,
    TransformerBlock,
    TransformerBlockConfig,
)
from src.utils.models.components.cswa import (
    CompressedSlidingWindowSelfAttention,
    CSWAConfig,
)
from src.utils.models.components.ffn_layers import MLP, SwiGLU, default_ffn_dim
from src.utils.models.components.fixed_query_track_stage import FixedQueryTrackStage
from src.utils.models.components.moe import MoEConfig, MoELayer, MoERouting, TopKRouter
from src.utils.models.components.norm import LayerNorm, RMSNorm
from src.utils.models.components.rope import (
    RotaryFrequencyComputer,
    apply_rotary_emb,
    precompute_freqs_cis,
    precompute_freqs_cis_nd,
)

__all__ = [
    # Attention
    "GroupedQuerySelfAttention",
    "MultiHeadCrossAttention",
    "MultiHeadSelfAttention",
    "CSWAConfig",
    "CompressedSlidingWindowSelfAttention",
    # Norm
    "RMSNorm",
    "LayerNorm",
    # RoPE
    "RotaryFrequencyComputer",
    "precompute_freqs_cis",
    "precompute_freqs_cis_nd",
    "apply_rotary_emb",
    # FFN
    "MLP",
    "SwiGLU",
    "default_ffn_dim",
    "MoEConfig",
    "MoELayer",
    "MoERouting",
    "TopKRouter",
    # Blocks
    "TransformerBlockConfig",
    "TransformerBlock",
    "CrossAttnBlockConfig",
    "CrossAttnBlock",
    "FixedQueryTrackStage",
]

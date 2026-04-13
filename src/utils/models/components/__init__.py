"""Common Transformer components (pure PyTorch, DeepSeek-style).

This package provides reusable building blocks used across tasks:

- Attention: `MultiHeadSelfAttention`, `MultiHeadCrossAttention`
- Norm: `RMSNorm`, `LayerNorm`
- RoPE: 1D (`precompute_freqs_cis`, `YaRNConfig`) and 2D axial (`precompute_freqs_cis_2d`, `apply_rotary_emb_2d`, `YaRNConfig2D`)
        plus compatibility helpers (`RotaryPositionEmbedding2D`, `PositionGetter`)
- FFN: `SwiGLU`, `MLP`, `default_ffn_dim`
- Blocks: `TransformerBlock`, `TransformerBlockConfig`, `CrossAttnBlockConfig`, `CrossAttnBlock`

Note:
This repository previously had a separate "unified MHA/GQA/MLA" implementation.
Strategy A treats the DeepSeek-style implementation as canonical.
"""

from src.utils.models.components.attention import (
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)
from src.utils.models.components.block import (
    CrossAttnBlock,
    CrossAttnBlockConfig,
    TransformerBlock,
    TransformerBlockConfig,
)
from src.utils.models.components.ffn_layers import MLP, SwiGLU, default_ffn_dim
from src.utils.models.components.norm import LayerNorm, RMSNorm
from src.utils.models.components.rope import (
    PositionGetter,
    RotaryPositionEmbedding2D,
    YaRNConfig,
    YaRNConfig2D,
    apply_rotary_emb_2d,
    precompute_freqs_cis,
    precompute_freqs_cis_2d,
)

__all__ = [
    # Attention
    "MultiHeadCrossAttention",
    "MultiHeadSelfAttention",
    # Norm
    "RMSNorm",
    "LayerNorm",
    # RoPE
    "YaRNConfig",
    "YaRNConfig2D",
    "precompute_freqs_cis",
    "precompute_freqs_cis_2d",
    "apply_rotary_emb_2d",
    "RotaryPositionEmbedding2D",
    "PositionGetter",
    # FFN
    "MLP",
    "SwiGLU",
    "default_ffn_dim",
    # Blocks
    "TransformerBlockConfig",
    "TransformerBlock",
    "CrossAttnBlockConfig",
    "CrossAttnBlock",
]

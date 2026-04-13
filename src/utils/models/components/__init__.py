"""Common Transformer components (pure PyTorch, DeepSeek-style).

This package provides reusable building blocks used across tasks:

- Attention: `MultiHeadSelfAttention`, `MultiHeadCrossAttention`
- Norm: `RMSNorm`, `LayerNorm`
- RoPE: 1D (`precompute_freqs_cis`, `YaRNConfig`) and 2D axial (`precompute_freqs_cis_2d`, `apply_rotary_emb_2d`, `YaRNConfig2D`)
        plus compatibility helpers (`RotaryPositionEmbedding2D`, `PositionGetter`)
- MLP / MoE: `SwiGLU`, `MoE`, `MoEConfig`
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
from src.utils.models.components.moe import MoE, MoEConfig, SwiGLU
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
    # MLP / MoE
    "SwiGLU",
    "MoEConfig",
    "MoE",
    # Blocks
    "TransformerBlockConfig",
    "TransformerBlock",
    "CrossAttnBlockConfig",
    "CrossAttnBlock",
]

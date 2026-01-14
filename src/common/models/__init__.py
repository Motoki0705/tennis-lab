"""Common model components (DeepSeek-style).

Strategy A: treat the DeepSeek-style, pure PyTorch implementation in
`src.common.models.components` as canonical and re-export it here.
"""

from src.common.models.components import (
    KVCache,
    LayerNorm,
    MoE,
    MoEConfig,
    MultiHeadSelfAttention,
    PositionGetter,
    RMSNorm,
    RotaryPositionEmbedding2D,
    SwiGLU,
    TransformerBlock,
    TransformerBlockConfig,
    ViTBlock,
    ViTBlockConfig,
    YaRNConfig,
    YaRNConfig2D,
    apply_rotary_emb_2d,
    precompute_freqs_cis,
    precompute_freqs_cis_2d,
)
from src.common.models.vit import ViTConfig, ViTEncoder

__all__ = [
    # Attention
    "KVCache",
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
    "ViTBlockConfig",
    "ViTBlock",
    # ViT
    "ViTConfig",
    "ViTEncoder",
]

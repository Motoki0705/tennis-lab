"""Common model components (DeepSeek-style).

Strategy A: treat the DeepSeek-style, pure PyTorch implementation in
`src.utils.models.components` as canonical and re-export it here.
"""

from src.utils.models.components import (
    CrossAttnBlock,
    CrossAttnBlockConfig,
    MLP,
    LayerNorm,
    MSDeformAttnConfig,
    MSDeformCrossAttnBlock,
    MSDeformCrossAttnBlockConfig,
    MultiHeadCrossAttention,
    MultiScaleDeformableAttention,
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
    default_ffn_dim,
    precompute_freqs_cis,
    precompute_freqs_cis_2d,
)
from src.utils.models.embeddings import (
    Ball3DEmbedding,
    BallUVEmbedding,
    CourtKPUVEmbedding,
    InvisibleTokenEmbedding,
    PlayerKPUVEmbedding,
)

__all__ = [
    # Attention
    "MultiHeadCrossAttention",
    "MultiHeadSelfAttention",
    "MSDeformAttnConfig",
    "MultiScaleDeformableAttention",
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
    "MSDeformCrossAttnBlockConfig",
    "MSDeformCrossAttnBlock",
    "ViTBlockConfig",
    "ViTBlock",
    # Token embeddings
    "InvisibleTokenEmbedding",
    "CourtKPUVEmbedding",
    "PlayerKPUVEmbedding",
    "BallUVEmbedding",
    "Ball3DEmbedding",
]

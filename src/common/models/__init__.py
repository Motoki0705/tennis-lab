"""Common model components (DeepSeek-style).

Strategy A: treat the DeepSeek-style, pure PyTorch implementation in
`src.common.models.components` as canonical and re-export it here.
"""

from src.common.models.components import (
    CrossAttnBlock,
    CrossAttnBlockConfig,
    KVCache,
    LayerNorm,
    MoE,
    MoEConfig,
    MultiHeadCrossAttention,
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
from src.common.models.embeddings import (
    Ball3DEmbedding,
    BallUVEmbedding,
    CourtKPUVEmbedding,
    InvisibleTokenEmbedding,
    PlayerKPUVEmbedding,
)
from src.common.models.token_embeddings import (
    CourtKPUVTokenEmbedding,
    UVObsTokenEmbedding,
)
from src.common.models.vit import ViTConfig, ViTEncoder

__all__ = [
    # Attention
    "KVCache",
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
    "ViTBlockConfig",
    "ViTBlock",
    # ViT
    "ViTConfig",
    "ViTEncoder",
    # Token embeddings
    "CourtKPUVTokenEmbedding",
    "UVObsTokenEmbedding",
    "InvisibleTokenEmbedding",
    "CourtKPUVEmbedding",
    "PlayerKPUVEmbedding",
    "BallUVEmbedding",
    "Ball3DEmbedding",
]

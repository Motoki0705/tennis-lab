"""Common model components (DeepSeek-style).

Strategy A: treat the DeepSeek-style, pure PyTorch implementation in
`src.utils.models.components` as canonical and re-export it here.
"""

from src.utils.models.components import (
    MLP,
    CrossAttnBlock,
    CrossAttnBlockConfig,
    LayerNorm,
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    TransformerBlockConfig,
    apply_rotary_emb,
    default_ffn_dim,
    precompute_freqs_cis,
    precompute_freqs_cis_nd,
)
from src.utils.models.embeddings import (
    Ball3DEmbedding,
    BallUVEmbedding,
    CourtKPUVEmbedding,
    InvisibleTokenEmbedding,
    PlayerKPUVEmbedding,
)
from src.utils.models.architectures import (
    TransformerSequenceDiscriminator,
)

__all__ = [
    # Attention
    "MultiHeadCrossAttention",
    "MultiHeadSelfAttention",
    # Norm
    "RMSNorm",
    "LayerNorm",
    # RoPE
    "precompute_freqs_cis",
    "precompute_freqs_cis_nd",
    "apply_rotary_emb",
    # FFN
    "MLP",
    "SwiGLU",
    "default_ffn_dim",
    # Blocks
    "TransformerBlockConfig",
    "TransformerBlock",
    "CrossAttnBlockConfig",
    "CrossAttnBlock",
    # Token embeddings
    "InvisibleTokenEmbedding",
    "CourtKPUVEmbedding",
    "PlayerKPUVEmbedding",
    "BallUVEmbedding",
    "Ball3DEmbedding",
    "TransformerSequenceDiscriminator",
]

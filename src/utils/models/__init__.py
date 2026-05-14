"""Common model components (DeepSeek-style).

Strategy A: treat the DeepSeek-style, pure PyTorch implementation in
`src.utils.models.components` as canonical and re-export it here.
"""

from src.utils.models.architectures import (
    TransformerSequenceDiscriminator,
)
from src.utils.models.components import (
    MLP,
    CrossAttnBlock,
    CrossAttnBlockConfig,
    GroupedQuerySelfAttention,
    LayerNorm,
    MoEConfig,
    MoELayer,
    MoERouting,
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
    RMSNorm,
    SwiGLU,
    TopKRouter,
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

__all__ = [
    # Attention
    "GroupedQuerySelfAttention",
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
    "MoEConfig",
    "MoELayer",
    "MoERouting",
    "TopKRouter",
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

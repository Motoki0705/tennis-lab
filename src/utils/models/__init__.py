"""Common model components (DeepSeek-style).

Strategy A: treat the DeepSeek-style, pure PyTorch implementation in
`src.utils.models.components` as canonical and re-export it here.
"""

from src.utils.models.architectures import (
    TransformerSequenceDiscriminator,
)
from src.utils.models.attention_extraction import (
    AttentionExtractor,
    AttentionPredicate,
    find_attention_modules,
    is_sdpa_self_attention,
    iter_attention_maps,
)
from src.utils.models.blocks import (
    Conv2dWiseWiseBlock,
    DepthwiseSeparableConv2d,
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
from src.utils.models.kimi_delta_attention import kimi_delta_attention
from src.utils.models.lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    iter_lora_parameters,
    mark_only_lora_as_trainable,
)
from src.utils.models.transformer_utils import (
    build_self_attn_mask,
    resolve_rope_bases,
    validate_rope_dim,
)

__all__ = [
    # Attention
    "GroupedQuerySelfAttention",
    "MultiHeadCrossAttention",
    "MultiHeadSelfAttention",
    "kimi_delta_attention",
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
    # Conv blocks
    "DepthwiseSeparableConv2d",
    "Conv2dWiseWiseBlock",
    # Token embeddings
    "InvisibleTokenEmbedding",
    "CourtKPUVEmbedding",
    "PlayerKPUVEmbedding",
    "BallUVEmbedding",
    "Ball3DEmbedding",
    "TransformerSequenceDiscriminator",
    # Attention extraction / analysis
    "AttentionExtractor",
    "AttentionPredicate",
    "find_attention_modules",
    "is_sdpa_self_attention",
    "iter_attention_maps",
    # LoRA adaptation
    "LoRAConfig",
    "LoRALinear",
    "apply_lora",
    "iter_lora_parameters",
    "mark_only_lora_as_trainable",
    # Shared transformer utilities
    "build_self_attn_mask",
    "resolve_rope_bases",
    "validate_rope_dim",
]

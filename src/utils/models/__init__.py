"""Canonical public API for repository-wide model primitives.

Implementation modules remain responsibility-focused below this package. The
symbols listed in ``__all__`` use this package root as their public import path;
specialized APIs not listed here remain owned by their defining subpackage.
"""

from src.utils.models.components import (
    CompressedSlidingWindowSelfAttention,
    CrossAttnBlock,
    CrossAttnBlockConfig,
    CSWAConfig,
    FixedQueryTrackStage,
    RMSNorm,
    RotaryFrequencyComputer,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis_nd,
)
from src.utils.models.kimi_delta_attention import kimi_delta_attention
from src.utils.models.multiview_padding import (
    FixedQueryPaddingMasks,
    build_fixed_query_padding_masks,
)
from src.utils.models.transformer_utils import (
    build_self_attn_mask,
    resolve_axial_rope_bases,
    resolve_rope_bases,
    validate_rope_dim,
)

__all__ = [
    "CSWAConfig",
    "CompressedSlidingWindowSelfAttention",
    "CrossAttnBlock",
    "CrossAttnBlockConfig",
    "FixedQueryPaddingMasks",
    "FixedQueryTrackStage",
    "RMSNorm",
    "RotaryFrequencyComputer",
    "TransformerBlock",
    "TransformerBlockConfig",
    "build_self_attn_mask",
    "build_fixed_query_padding_masks",
    "kimi_delta_attention",
    "precompute_freqs_cis_nd",
    "resolve_axial_rope_bases",
    "resolve_rope_bases",
    "validate_rope_dim",
]

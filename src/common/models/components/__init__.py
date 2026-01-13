"""Common Transformer components for model architectures.

This module provides reusable components for building modern Transformer models:

Normalization:
    - RMSNorm: Root Mean Square Layer Normalization

Positional Encoding:
    - RoPE / RoPEConfig: Rotary Position Embedding (1D)
    - RoPE2D / RoPE2DConfig: 2D Rotary Position Embedding for ViT

Attention:
    - MHA / MHAConfig: Multi-Head Attention (standard)
    - GQA / GQAConfig: Grouped-Query Attention
    - MLA / MLAConfig: Multi-head Latent Attention (DeepSeek-V2/V3)
    - AttentionType: Enum for attention type selection
    - build_attention: Factory function to build attention modules

MLP / MoE:
    - SwiGLUMLP: SwiGLU Feed-Forward Network
    - MoELayer / MoEConfig: Mixture of Experts layer

Blocks:
    - TransformerBlock / BlockConfig: Configurable Transformer block
    - ViTBlock / ViTBlockConfig: Vision Transformer block with 2D RoPE
"""

from src.common.models.components.attention import (
    GQA,
    GQAConfig,
    MHA,
    MHAConfig,
    MLA,
    MLAConfig,
    AttentionType,
    build_attention,
)
from src.common.models.components.blocks import (
    BlockConfig,
    TransformerBlock,
    ViTBlock,
    ViTBlockConfig,
)
from src.common.models.components.mlp import MoEConfig, MoEGate, MoELayer, SwiGLUMLP
from src.common.models.components.norm import RMSNorm
from src.common.models.components.rope import RoPE, RoPE2D, RoPE2DConfig, RoPEConfig

__all__ = [
    # Normalization
    "RMSNorm",
    # Positional Encoding
    "RoPE",
    "RoPEConfig",
    "RoPE2D",
    "RoPE2DConfig",
    # Attention
    "MHA",
    "MHAConfig",
    "GQA",
    "GQAConfig",
    "MLA",
    "MLAConfig",
    "AttentionType",
    "build_attention",
    # MLP / MoE
    "SwiGLUMLP",
    "MoEConfig",
    "MoEGate",
    "MoELayer",
    # Blocks
    "BlockConfig",
    "TransformerBlock",
    "ViTBlock",
    "ViTBlockConfig",
]

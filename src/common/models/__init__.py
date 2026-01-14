"""Common model components for Transformer architectures.

This module provides reusable components:

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
    - MoELayer / MoEConfig / MoEGate: Mixture of Experts layer

Blocks:
    - TransformerBlock / BlockConfig: Configurable Transformer block
    - ViTBlock / ViTBlockConfig: Vision Transformer block with 2D RoPE

ViT:
    - ViTEncoder / ViTConfig: Modern Vision Transformer encoder
"""

from src.common.models.components import (
    AttentionType,
    BlockConfig,
    GQA,
    GQAConfig,
    MHA,
    MHAConfig,
    MLA,
    MLAConfig,
    MoEConfig,
    MoEGate,
    MoELayer,
    RMSNorm,
    RoPE,
    RoPE2D,
    RoPE2DConfig,
    RoPEConfig,
    SwiGLUMLP,
    TransformerBlock,
    ViTBlock,
    ViTBlockConfig,
    build_attention,
)
from src.common.models.vit import ViTConfig, ViTEncoder

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
    # ViT
    "ViTConfig",
    "ViTEncoder",
]

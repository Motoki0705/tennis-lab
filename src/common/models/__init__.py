"""Common model components for Llama-style Transformer architecture.

This module provides reusable components:
- RMSNorm: Root Mean Square Layer Normalization
- RoPE / RoPEConfig: Rotary Position Embedding (1D)
- RoPE2D / RoPE2DConfig: 2D Rotary Position Embedding for ViT
- GQASelfAttention: Grouped-Query Attention with SDPA
- MLA / MLAConfig: Multi-head Latent Attention (DeepSeek-V2/V3)
- SwiGLUMLP: SwiGLU Feed-Forward Network
- MoELayer / MoEConfig: Mixture of Experts layer
- TransformerBlock: Pre-norm Transformer block with GQA + SwiGLU
- ViTBlock: Vision Transformer block with 2D RoPE and optional MoE
- ViTEncoder / ViTConfig: Modern Vision Transformer encoder
"""

from src.common.models.components import (
    MLA,
    GQASelfAttention,
    MLAConfig,
    MoEConfig,
    MoELayer,
    RMSNorm,
    RoPE,
    RoPE2D,
    RoPE2DConfig,
    RoPEConfig,
    SwiGLUMLP,
    TransformerBlock,
)
from src.common.models.components.blocks import ViTBlock
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
    "GQASelfAttention",
    "MLA",
    "MLAConfig",
    # MLP
    "SwiGLUMLP",
    # MoE
    "MoEConfig",
    "MoELayer",
    # Blocks
    "TransformerBlock",
    "ViTBlock",
    # ViT
    "ViTConfig",
    "ViTEncoder",
]

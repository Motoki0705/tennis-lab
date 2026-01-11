"""Common model components for Llama-style Transformer architecture.

This module provides reusable components:
- RMSNorm: Root Mean Square Layer Normalization
- RoPE / RoPEConfig: Rotary Position Embedding
- GQASelfAttention: Grouped-Query Attention with SDPA
- SwiGLUMLP: SwiGLU Feed-Forward Network
- TransformerBlock: Pre-norm Transformer block with GQA + SwiGLU
"""

from src.common.models.attention import GQASelfAttention, RoPE, RoPEConfig
from src.common.models.blocks import TransformerBlock
from src.common.models.mlp import SwiGLUMLP
from src.common.models.norm import RMSNorm

__all__ = [
    "RMSNorm",
    "RoPE",
    "RoPEConfig",
    "GQASelfAttention",
    "SwiGLUMLP",
    "TransformerBlock",
]

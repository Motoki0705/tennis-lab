# ==============================================================================
# NOTE ON ORIGIN / LICENSE
#
# This file is derived from (and/or inspired by) DeepSeek's inference reference:
#   https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py
#
# MIT License
#
# Copyright (c) 2025 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
block.py (pure PyTorch)

Includes:
- TransformerBlock: DeepSeek-style block structure (RMSNorm + residual accumulator)
- ViTBlock: standard Vision Transformer block (LayerNorm + residual inside the block)

This module assumes single-GPU / non-distributed execution and uses the pure PyTorch
attention / norm / MoE implementations from sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.utils.models.components.attention import (
    KVCache,
    MSDeformAttnConfig,
    MultiHeadCrossAttention,
    MultiScaleDeformableAttention,
    MultiHeadSelfAttention,
)
from src.utils.models.components.moe import MoE, MoEConfig, SwiGLU
from src.utils.models.components.norm import RMSNorm
from src.utils.models.components.rope import (
    PositionGetter,
    YaRNConfig,
    precompute_freqs_cis_2d,
)


@dataclass
class TransformerBlockConfig:
    """Configuration for TransformerBlock.

    Args:
        dim: Token embedding dimension.
        n_heads: Number of attention heads.
        mlp_inter_dim: Hidden dimension for the MLP/FFN.
        head_dim: Per-head dimension (defaults to dim // n_heads).
        rope_dim: Rotary dimension per head for 1D RoPE.
        attn_dropout: Dropout probability for attention.
        rope_base: Base theta for 1D RoPE.
        yarn: Optional YaRN correction config.
        use_moe: Whether to use MoE FFN.
        moe_config: MoE configuration when `use_moe=True`.
    """

    dim: int
    n_heads: int
    mlp_inter_dim: int
    # attention
    head_dim: int | None = None
    rope_dim: int | None = None
    attn_dropout: float = 0.0
    # RoPE
    rope_base: float = 10000.0
    yarn: YaRNConfig | None = None
    # MoE (optional)
    use_moe: bool = False
    moe_config: MoEConfig | None = None


class TransformerBlock(nn.Module):
    """
    DeepSeek-style Transformer block with a residual accumulator.

    forward returns (x, residual):
      - x is the "current" stream
      - residual is the running residual stream

    Typical usage (prefill):
        x, residual = block(x, residual=None, start_pos=0, freqs_cis=freqs, is_causal=True)
    """

    def __init__(self, cfg: TransformerBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = MultiHeadSelfAttention(
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            rope_dim=cfg.rope_dim,
            attn_dropout=cfg.attn_dropout,
        )

        self.ffn_norm = RMSNorm(cfg.dim)
        if cfg.use_moe:
            if cfg.moe_config is None:
                raise ValueError("use_moe=True requires moe_config.")
            self.ffn: nn.Module = MoE(cfg.moe_config)
        else:
            self.ffn = SwiGLU(cfg.dim, cfg.mlp_inter_dim)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        *,
        start_pos: int,
        freqs_cis: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool | None = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            x_norm = self.attn_norm(x)
            residual = x
        else:
            x_norm, residual = self.attn_norm(x, residual)

        x = self.attn(
            x_norm,
            start_pos=start_pos,
            freqs_cis=freqs_cis,
            kv_cache=kv_cache,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )

        x_norm, residual = self.ffn_norm(x, residual)
        x = self.ffn(x_norm)
        return x, residual


@dataclass
class CrossAttnBlockConfig:
    """Configuration for CrossAttnBlock."""

    dim: int
    n_heads: int
    mlp_inter_dim: int
    # attention
    head_dim: int | None = None
    rope_dim: int | None = None
    attn_dropout: float = 0.0
    # MoE (optional)
    use_moe: bool = False
    moe_config: MoEConfig | None = None


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention block with optional RoPE on query/key."""

    def __init__(self, cfg: CrossAttnBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.q_norm = RMSNorm(cfg.dim)
        self.kv_norm = RMSNorm(cfg.dim)
        self.attn = MultiHeadCrossAttention(
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            rope_dim=cfg.rope_dim,
            attn_dropout=cfg.attn_dropout,
        )
        self.ffn_norm = RMSNorm(cfg.dim)
        if cfg.use_moe:
            if cfg.moe_config is None:
                raise ValueError("use_moe=True requires moe_config.")
            self.ffn: nn.Module = MoE(cfg.moe_config)
        else:
            self.ffn = SwiGLU(cfg.dim, cfg.mlp_inter_dim)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        *,
        key_valid: torch.Tensor | None = None,
        freqs_q_cis: torch.Tensor | None = None,
        freqs_k_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = q.shape
        _, k_len, _ = kv.shape

        q_norm = self.q_norm(q)
        kv_norm = self.kv_norm(kv)

        attn_mask: torch.Tensor | None = None
        if key_valid is not None:
            if key_valid.shape != (bsz, k_len):
                raise ValueError(
                    f"key_valid must have shape {(bsz, k_len)}, got {tuple(key_valid.shape)}"
                )
            key_keep = key_valid > 0
            fully_masked = ~key_keep.any(dim=1)
            if fully_masked.any():
                key_keep = key_keep.clone()
                key_keep[fully_masked, 0] = True
                kv_norm = kv_norm.clone()
                kv_norm[fully_masked] = 0.0
            attn_mask = key_keep[:, None, :].expand(bsz, q_len, k_len)

        q = q + self.attn(
            q_norm,
            kv_norm,
            freqs_q_cis=freqs_q_cis,
            freqs_k_cis=freqs_k_cis,
            attn_mask=attn_mask,
        )
        q = q + self.ffn(self.ffn_norm(q))
        return q


@dataclass
class MSDeformCrossAttnBlockConfig:
    """Configuration for MSDeformCrossAttnBlock."""

    dim: int
    n_heads: int
    n_levels: int
    n_points: int
    mlp_inter_dim: int
    attn_dropout: float = 0.0
    use_cuda_kernel: bool = True
    allow_fallback: bool = True
    offset_scale: float = 0.5


class MSDeformCrossAttnBlock(nn.Module):
    """Pre-norm multi-scale deformable cross-attention block."""

    def __init__(self, cfg: MSDeformCrossAttnBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.dim % cfg.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads.")
        if cfg.n_levels <= 0:
            raise ValueError("n_levels must be positive.")
        if cfg.n_points <= 0:
            raise ValueError("n_points must be positive.")

        self.q_norm = RMSNorm(cfg.dim)
        self.kv_norm = RMSNorm(cfg.dim)
        self.attn = MultiScaleDeformableAttention(
            MSDeformAttnConfig(
                dim=cfg.dim,
                num_heads=cfg.n_heads,
                num_levels=cfg.n_levels,
                num_points=cfg.n_points,
                use_cuda_kernel=cfg.use_cuda_kernel,
                allow_fallback=cfg.allow_fallback,
            )
        )
        self.ref_point_proj = nn.Linear(cfg.dim, cfg.n_levels * 2)
        self.offset_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.n_levels * cfg.n_points * 2)
        self.attn_weight_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.n_levels * cfg.n_points)

        self.offset_scale = float(cfg.offset_scale)
        self.dropout = nn.Dropout(float(cfg.attn_dropout))
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg.dim, cfg.mlp_inter_dim)

    def _prepare_memory(self, memory_levels: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(memory_levels) != self.cfg.n_levels:
            raise ValueError(
                f"Expected {self.cfg.n_levels} memory levels, got {len(memory_levels)}."
            )

        value_levels: list[torch.Tensor] = []
        spatial_shapes: list[tuple[int, int]] = []
        bsz_ref = int(memory_levels[0].shape[0])
        for lvl in memory_levels:
            if lvl.dim() != 4:
                raise ValueError(f"memory level must be (B,D,H,W), got {tuple(lvl.shape)}")
            bsz, dim, h, w = lvl.shape
            if bsz != bsz_ref:
                raise ValueError("All memory levels must have the same batch size.")
            if dim != self.cfg.dim:
                raise ValueError(f"Expected memory dim {self.cfg.dim}, got {dim}.")
            tok = lvl.flatten(2).transpose(1, 2).contiguous()
            tok = self.kv_norm(tok)
            value_levels.append(tok)
            spatial_shapes.append((int(h), int(w)))

        value = torch.cat(value_levels, dim=1)
        shape_t = torch.tensor(spatial_shapes, device=value.device, dtype=torch.long)
        return value, shape_t

    def _build_sampling(self, q_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = q_norm.shape
        n_heads = self.cfg.n_heads
        n_levels = self.cfg.n_levels
        n_points = self.cfg.n_points

        ref = torch.sigmoid(self.ref_point_proj(q_norm)).view(bsz, q_len, n_levels, 2)
        offsets = self.offset_proj(q_norm).view(bsz, q_len, n_heads, n_levels, n_points, 2)
        offsets = torch.tanh(offsets) * self.offset_scale
        sampling_locations = torch.clamp(ref[:, :, None, :, None, :] + offsets, min=0.0, max=1.0)

        weights = self.attn_weight_proj(q_norm).view(bsz, q_len, n_heads, n_levels * n_points)
        weights = torch.softmax(weights, dim=-1).view(bsz, q_len, n_heads, n_levels, n_points)
        return sampling_locations, weights

    def forward(self, q: torch.Tensor, memory_levels: list[torch.Tensor]) -> torch.Tensor:
        if q.dim() != 3:
            raise ValueError(f"q must be (B,Q,D), got {tuple(q.shape)}")
        if q.shape[-1] != self.cfg.dim:
            raise ValueError(f"q dim mismatch: expected {self.cfg.dim}, got {q.shape[-1]}")

        q_norm = self.q_norm(q)
        value, spatial_shapes = self._prepare_memory(memory_levels)
        sampling_locations, attention_weights = self._build_sampling(q_norm)

        q = q + self.dropout(
            self.attn(
                query=q_norm,
                value=value,
                spatial_shapes=spatial_shapes,
                sampling_locations=sampling_locations,
                attention_weights=attention_weights,
            )
        )
        q = q + self.ffn(self.ffn_norm(q))
        return q


@dataclass
class ViTBlockConfig:
    """Configuration for ViTBlock.

    Args:
        dim: Token embedding dimension.
        n_heads: Number of attention heads.
        mlp_inter_dim: Hidden dimension for the MLP/FFN.
        attn_dropout: Dropout probability for attention.
        mlp_dropout: Dropout probability for the MLP/FFN.
        use_2d_rope: Whether to apply 2D axial RoPE.
        rope2d_frequency: Base theta for 2D RoPE.
        rope_dim: Rotary dimension per head for 2D RoPE.
        use_moe: Whether to use MoE FFN.
        moe_config: MoE configuration when `use_moe=True`.
    """

    dim: int
    n_heads: int
    mlp_inter_dim: int
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    # optional 2D RoPE
    use_2d_rope: bool = False
    rope2d_frequency: float = 100.0
    rope_dim: int | None = None  # by default full head_dim
    # MoE (optional)
    use_moe: bool = False
    moe_config: MoEConfig | None = None


class ViTBlock(nn.Module):
    """
    Vision Transformer block (pre-norm).

    This block is meant for non-causal attention (is_causal=False).
    Optionally applies 2D RoPE to q/k using positions_2d.

    Expected input:
        x: (B, N, C)
        positions_2d: (B, T_rope, 2) if use_2d_rope is enabled.
                     If T_rope < N, 2D RoPE is applied to the last T_rope tokens and
                     the prefix (e.g., [CLS]/[REG...]) is left unchanged.
    """

    def __init__(self, cfg: ViTBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = MultiHeadSelfAttention(
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            rope_dim=cfg.rope_dim,
        )
        self.norm2 = nn.LayerNorm(cfg.dim)

        if cfg.use_moe:
            if cfg.moe_config is None:
                raise ValueError("use_moe=True requires moe_config.")
            self.ffn: nn.Module = MoE(cfg.moe_config)
        else:
            self.ffn = SwiGLU(cfg.dim, cfg.mlp_inter_dim)

        self.use_2d_rope = bool(cfg.use_2d_rope)
        self.rope2d_base = float(cfg.rope2d_frequency)
        self._rope2d_cache: dict[tuple[int, int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

        self.pos_getter: PositionGetter | None
        if cfg.use_2d_rope:
            self.pos_getter = PositionGetter()
        else:
            self.pos_getter = None

    def _get_rope2d_freqs(
        self,
        *,
        dim: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (dim, height, width, device)
        if key not in self._rope2d_cache:
            freqs_y, freqs_x = precompute_freqs_cis_2d(
                dim=dim,
                height=height,
                width=width,
                base=self.rope2d_base,
                device=device,
            )
            self._rope2d_cache[key] = (freqs_y, freqs_x)
        return self._rope2d_cache[key]

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions_2d: torch.Tensor | None = None,
        grid_hw: tuple[int, int] | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
            positions_2d: (B, T_rope, 2) integer y/x coordinates for tokens that receive 2D RoPE.
            grid_hw: convenience for patch tokens: if provided and positions_2d is None,
                     positions are generated for a HxW grid (patch tokens only).
            attn_mask: optional SDPA mask (usually None for ViT)

        Returns:
            (B, N, C)
        """
        bsz, n, _ = x.shape

        rope2d: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.use_2d_rope:
            if positions_2d is None:
                if grid_hw is None or self.pos_getter is None:
                    raise ValueError("use_2d_rope=True requires positions_2d or grid_hw.")
                h, w = grid_hw
                positions_2d = self.pos_getter(bsz, h, w, x.device)

            if positions_2d.size(1) > n:
                raise ValueError(f"positions_2d has T={positions_2d.size(1)} but x has N={n}")

            if grid_hw is not None:
                h, w = grid_hw
            else:
                h = int(positions_2d[..., 0].max().item()) + 1
                w = int(positions_2d[..., 1].max().item()) + 1

            rope2d = self._get_rope2d_freqs(dim=self.attn.rope_dim, height=h, width=w, device=x.device)

        x = x + self.attn(
            self.norm1(x),
            start_pos=0,
            rope2d=rope2d,
            positions_2d=positions_2d,
            attn_mask=attn_mask,
            is_causal=False,
        )
        x = x + self.ffn(self.norm2(x))
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo_cfg = ViTBlockConfig(
        dim=32,
        n_heads=4,
        mlp_inter_dim=64,
        attn_dropout=0.0,
        mlp_dropout=0.0,
        use_2d_rope=False,
        use_moe=False,
    )
    demo_block = ViTBlock(demo_cfg).eval().to(demo_device)
    demo_input = torch.randn(1, 6, 32, device=demo_device)

    with torch.no_grad():
        demo_output = demo_block(demo_input)

    print(demo_output)

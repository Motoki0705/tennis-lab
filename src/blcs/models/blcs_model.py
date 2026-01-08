"""Main BLCS model implementation.

Ball Localization in Court System: Decoder-Only Transformer with GQA + RoPE + SDPA.
Estimates ball 3D trajectory in tennis court coordinates from 2D ball observations
and court keypoints.

Architecture:
    - Court keypoints and ball trajectory are tokenized and concatenated
    - Decoder-only Transformer with RoPE positional encoding
    - Grouped-Query Attention (GQA) using F.scaled_dot_product_attention (SDPA)
    - SwiGLU MLP and RMSNorm for efficiency
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.blcs.models.components.heads import Trajectory3DHead, VelocityHead
from src.utils.geometry import NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


# -------------------------
# Utils / Norm
# -------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


# -------------------------
# RoPE (Rotary Position Embedding)
# -------------------------
@dataclass
class RoPEConfig:
    """Configuration for Rotary Position Embedding."""

    rope_dim: int
    rope_theta: float = 10000.0


class RoPE(nn.Module):
    """Rotary Position Embedding.

    Applies rotary position embedding to last-dim features in pairs
    on the first rope_dim dims. Works on (B, H, S, D_head).
    """

    def __init__(self, cfg: RoPEConfig) -> None:
        super().__init__()
        assert cfg.rope_dim % 2 == 0
        self.cfg = cfg

    def _build_inv_freq(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        half = self.cfg.rope_dim // 2
        i = torch.arange(half, device=device, dtype=dtype)
        return self.cfg.rope_theta ** (-i / half)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """Apply rotary position embedding.

        Args:
            x: Input tensor, shape (B, H, S, Dh).
            pos: Position indices, shape (S,) or (B, S).

        Returns:
            Tensor with rotary position embedding applied.

        """
        B, H, S, Dh = x.shape
        rope_dim = min(self.cfg.rope_dim, Dh)
        if rope_dim <= 0:
            return x

        x_rope = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]

        device, dtype = x.device, x.dtype
        inv_freq = self._build_inv_freq(device, dtype)

        if pos.dim() == 2:
            pos_ = pos[0]
        else:
            pos_ = pos
        pos_ = pos_.to(device=device, dtype=dtype)

        angles = torch.outer(pos_, inv_freq)
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]

        x1 = x_rope[..., 0::2]
        x2 = x_rope[..., 1::2]

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        y = torch.empty_like(x_rope)
        y[..., 0::2] = y1
        y[..., 1::2] = y2

        return torch.cat([y, x_pass], dim=-1)


# -------------------------
# GQA Attention with SDPA
# -------------------------
class GQASelfAttention(nn.Module):
    """Self-attention with Grouped-Query Attention using SDPA.

    Uses F.scaled_dot_product_attention for efficiency:
      - Q has num_heads
      - K/V have num_kv_heads
      - K/V are expanded (not repeated) to match num_heads via reshape
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float,
        rope: Optional[RoPE] = None,
        causal: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = dropout
        self.rope = rope
        self.causal = causal

        self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: Tensor,
        pos: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with SDPA.

        Args:
            x: Input tensor, shape (B, S, D).
            pos: Position indices for RoPE, shape (S,).
            key_padding_mask: Mask where True = keep, False = mask out, shape (B, S).

        Returns:
            Output tensor, shape (B, S, D).

        """
        B, S, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, hd)
        k = self.wk(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, Hkv, S, hd)
        v = self.wv(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, Hkv, S, hd)

        # Apply RoPE
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # GQA: expand K/V to match num_heads using view/expand (no memory copy)
        # (B, Hkv, S, hd) -> (B, Hkv, 1, S, hd) -> (B, Hkv, G, S, hd) -> (B, H, S, hd)
        k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.num_groups, S, self.head_dim)
        k = k.reshape(B, self.num_heads, S, self.head_dim)
        v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.num_groups, S, self.head_dim)
        v = v.reshape(B, self.num_heads, S, self.head_dim)

        # Build attention mask for SDPA
        attn_mask: Optional[Tensor] = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, S) True=keep, False=mask
            # SDPA expects: True=mask out (opposite convention)
            # Convert to (B, 1, 1, S) additive mask
            mask = ~key_padding_mask  # True = masked position
            attn_mask = torch.zeros(B, 1, 1, S, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # Use SDPA for efficient attention computation
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=self.causal and attn_mask is None,
        )  # (B, H, S, hd)

        out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return self.wo(out)


# -------------------------
# SwiGLU MLP
# -------------------------
class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.wu = nn.Linear(dim, ffn_dim, bias=False)
        self.wg = nn.Linear(dim, ffn_dim, bias=False)
        self.wd = nn.Linear(ffn_dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        h = self.wu(x) * F.silu(self.wg(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.wd(h)


# -------------------------
# Transformer block
# -------------------------
class TransformerBlock(nn.Module):
    """Transformer block with GQA attention and SwiGLU MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim: int,
        dropout: float,
        rope: Optional[RoPE],
        causal: bool,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = GQASelfAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            rope=rope,
            causal=causal,
        )
        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim=dim, ffn_dim=ffn_dim, dropout=dropout)

    def forward(
        self, x: Tensor, pos: Tensor, key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        x = x + self.attn(self.attn_norm(x), pos=pos, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# -------------------------
# Token embeddings
# -------------------------
class CourtTokenEmbedding(nn.Module):
    """Embed court keypoints as tokens.

    Args:
        court_kp: Court keypoints, shape (B, 40) or (B, 20, 2).
        court_vis: Court visibility, shape (B, 20). Optional.

    Returns:
        Tokens of shape (B, NUM_COURT_KP, D).

    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        in_dim = 2 + 1  # (u, v) + visibility
        self.proj = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, court_kp: Tensor, court_vis: Optional[Tensor]) -> Tensor:
        B = court_kp.shape[0]
        if court_kp.dim() == 2:
            court_kp = court_kp.view(B, NUM_COURT_KP, 2)
        vis = (
            torch.ones(B, NUM_COURT_KP, device=court_kp.device, dtype=court_kp.dtype)
            if court_vis is None
            else court_vis.to(court_kp.dtype)
        )
        x = torch.cat([court_kp, vis.unsqueeze(-1)], dim=-1)
        return self.proj(x)


class BallTokenEmbedding(nn.Module):
    """Embed ball trajectory as tokens.

    Args:
        ball_uv: Ball 2D positions, shape (B, T, 2).
        ball_mask: Ball visibility mask, shape (B, T). Optional.

    Returns:
        Tokens of shape (B, T, D).

    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        in_dim = 2 + 1  # (u, v) + visibility
        self.proj = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, ball_uv: Tensor, ball_mask: Optional[Tensor]) -> Tensor:
        B, T, _ = ball_uv.shape
        vis = (
            torch.ones(B, T, device=ball_uv.device, dtype=ball_uv.dtype)
            if ball_mask is None
            else ball_mask.to(ball_uv.dtype)
        )
        x = torch.cat([ball_uv, vis.unsqueeze(-1)], dim=-1)
        return self.proj(x)


# -------------------------
# Main model (legacy alias preserved)
# -------------------------
BLCSTransformerModel = "BLCSModel"  # Forward reference for compatibility


class BLCSModel(nn.Module):
    """BLCS: Ball Localization in Court System.

    Decoder-Only Transformer architecture with:
    - Grouped-Query Attention (GQA) with SDPA for efficiency
    - Rotary Position Embedding (RoPE)
    - SwiGLU MLP and RMSNorm

    Tokens = [court_tokens(NUM_COURT_KP), ball_tokens(T)]
    Predicts 3D positions from ball tokens only.

    Input:
        - ball_uv: Ball 2D positions (u, v), shape (B, T, 2)
        - court_kp: Court 2D keypoints (20 landmarks), shape (B, 40) or (B, 20, 2)
        - ball_mask: Ball visibility mask, shape (B, T). Optional.
        - court_vis: Court keypoint visibility, shape (B, 20). Optional.

    Output:
        - position: Normalized (x, y, z) trajectory, shape (B, T, 3)
        - velocity: Normalized velocities (optional), shape (B, T, 3)

    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        rope_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        causal: bool = False,
        predict_velocity: bool = False,
        max_seq_len: int = 120,
    ) -> None:
        """Initialize the BLCS model.

        Args:
            hidden_dim: Hidden dimension for all components.
            num_layers: Number of Transformer blocks.
            num_heads: Number of query attention heads.
            num_kv_heads: Number of key/value heads (for GQA).
            ffn_dim: FFN intermediate dimension. Defaults to 8/3 * hidden_dim.
            dropout: Dropout probability.
            rope_dim: RoPE dimension. Defaults to head_dim.
            rope_theta: RoPE theta parameter.
            causal: Use causal attention mask.
            predict_velocity: Also predict velocities (for auxiliary loss).
            max_seq_len: Maximum sequence length.

        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.predict_velocity = predict_velocity
        self.causal = causal

        head_dim = hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else rope_dim
        self.rope = RoPE(RoPEConfig(rope_dim=rope_dim, rope_theta=rope_theta))

        if ffn_dim is None:
            ffn_dim = int((8 * hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64  # Round to multiple of 64

        self.court_embed = CourtTokenEmbedding(dim=hidden_dim, dropout=dropout)
        self.ball_embed = BallTokenEmbedding(dim=hidden_dim, dropout=dropout)

        # Type embedding: 0 = court, 1 = ball
        self.type_embed = nn.Embedding(2, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    rope=self.rope,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_dim)

        self.position_head = Trajectory3DHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )
        self.velocity_head = None
        if predict_velocity:
            self.velocity_head = VelocityHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=3,
                num_layers=2,
                dropout=dropout,
            )

    @classmethod
    def from_config(cls, config: DictConfig) -> BLCSModel:
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            BLCSModel: Initialized model.

        """
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        return cls(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 6),
            num_heads=model_cfg.get("num_heads", 8),
            num_kv_heads=model_cfg.get("num_kv_heads", 2),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=model_cfg.get("dropout", 0.1),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=model_cfg.get("rope_theta", 10000.0),
            causal=model_cfg.get("causal", False),
            predict_velocity=model_cfg.get("predict_velocity", False),
            max_seq_len=model_cfg.get(
                "max_seq_len", data_cfg.get("max_seq_len", 120)
            ),
        )

    def _build_positions(self, T: int, device: torch.device) -> Tensor:
        """Build position indices for court + ball tokens."""
        S = NUM_COURT_KP + T
        return torch.arange(S, device=device, dtype=torch.long)

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_mask: Optional[Tensor] = None,
        court_vis: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            ball_uv: Ball 2D positions, shape (B, T, 2).
            court_kp: Court keypoints, shape (B, 40) or (B, 20, 2).
            ball_mask: Ball visibility mask, shape (B, T). Optional.
            court_vis: Court visibility mask, shape (B, 20). Optional.

        Returns:
            dict: Dictionary with 'position' (B, T, 3) and optionally 'velocity'.

        """
        B, T, _ = ball_uv.shape

        # Tokenize court and ball
        court_tok = self.court_embed(court_kp, court_vis)  # (B, 20, D)
        ball_tok = self.ball_embed(ball_uv, ball_mask)  # (B, T, D)

        # Add type embeddings
        court_type = self.type_embed(
            torch.zeros(NUM_COURT_KP, device=ball_uv.device, dtype=torch.long)
        )[None, :, :]  # (1, 20, D)
        ball_type = self.type_embed(
            torch.ones(T, device=ball_uv.device, dtype=torch.long)
        )[None, :, :]  # (1, T, D)

        x = torch.cat(
            [court_tok + court_type, ball_tok + ball_type], dim=1
        )  # (B, S, D)

        pos = self._build_positions(T, device=x.device)  # (S,)

        # Build key_padding_mask if ball_mask provided
        key_padding_mask: Optional[Tensor] = None
        if ball_mask is not None:
            # Court tokens are always valid, ball tokens use ball_mask
            court_mask = torch.ones(B, NUM_COURT_KP, device=x.device, dtype=torch.bool)
            key_padding_mask = torch.cat([court_mask, ball_mask > 0], dim=1)  # (B, S)

        for blk in self.blocks:
            x = blk(x, pos=pos, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)
        ball_out = x[:, NUM_COURT_KP:, :]  # (B, T, D)

        out: dict[str, Tensor] = {"position": self.position_head(ball_out)}  # (B, T, 3)

        if self.predict_velocity and self.velocity_head is not None:
            out["velocity"] = self.velocity_head(ball_out)  # (B, T, 3)

        return out

    def predict(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_mask: Optional[Tensor] = None,
        court_vis: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Inference mode prediction.

        Same as forward but ensures eval mode and no gradients.

        Args:
            ball_uv: Ball 2D positions.
            court_kp: Court keypoints.
            ball_mask: Ball visibility mask.
            court_vis: Court visibility mask.

        Returns:
            dict: Predictions with position (and optionally velocity).

        """
        self.eval()
        with torch.no_grad():
            return self.forward(ball_uv, court_kp, ball_mask, court_vis)

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Legacy alias for backward compatibility
BLCSTransformerModel = BLCSModel

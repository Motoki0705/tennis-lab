"""Main BLCS model implementation.

Ball Localization in Court System: Decoder-Only Transformer with GQA + RoPE + SDPA.
Estimates ball 3D trajectory in tennis court coordinates from 2D ball observations
and court keypoints.

Architecture:
    - Court keypoints and ball trajectory are tokenized and concatenated
    - Decoder-only Transformer with RoPE positional encoding
    - Grouped-Query Attention (GQA) using F.scaled_dot_product_attention (SDPA)
    - SwiGLU MLP and RMSNorm for efficiency

Uses shared components from src.common.models.components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.blcs.models.components.heads import Trajectory3DHead, VelocityHead
from src.common.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis,
)
from src.utils.geometry import NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


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

    def forward(self, court_kp: Tensor, court_vis: Tensor | None) -> Tensor:
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

    def forward(self, ball_uv: Tensor, ball_mask: Tensor | None) -> Tensor:
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
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
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
        self.max_tokens = int(NUM_COURT_KP + self.max_seq_len)

        head_dim = hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else rope_dim
        self.rope_dim = int(rope_dim)
        self.rope_theta = float(rope_theta)

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
                    TransformerBlockConfig(
                        dim=hidden_dim,
                        n_heads=num_heads,
                        mlp_inter_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        use_moe=False,
                    )
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

        freqs_cis = precompute_freqs_cis(
            dim=self.rope_dim,
            seqlen=self.max_tokens,
            base=self.rope_theta,
            device=None,  # initialized on CPU; moved by `model.to(device)`
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

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

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
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
        S = x.shape[1]

        # Build key_padding_mask if ball_mask provided
        key_padding_mask: Tensor | None = None
        if ball_mask is not None:
            # Court tokens are always valid, ball tokens use ball_mask
            court_mask = torch.ones(B, NUM_COURT_KP, device=x.device, dtype=torch.bool)
            key_padding_mask = torch.cat([court_mask, ball_mask > 0], dim=1)  # (B, S)

        if S > self.freqs_cis.shape[0]:
            raise ValueError(
                f"Sequence length S={S} exceeds cached freqs_cis length {self.freqs_cis.shape[0]}. "
                "Increase max_seq_len."
            )
        freqs_cis = self.freqs_cis[:S]
        if freqs_cis.device != x.device:
            freqs_cis = freqs_cis.to(x.device)

        attn_mask: Tensor | None = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, :].expand(B, S, S)

        residual = None
        for blk in self.blocks:
            x, residual = blk(
                x,
                residual,
                start_pos=0,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
                is_causal=self.causal,
            )

        if residual is None:
            x = self.final_norm(x)
        else:
            x, _ = self.final_norm(x, residual)
        ball_out = x[:, NUM_COURT_KP:, :]  # (B, T, D)

        out: dict[str, Tensor] = {"position": self.position_head(ball_out)}  # (B, T, 3)

        if self.predict_velocity and self.velocity_head is not None:
            out["velocity"] = self.velocity_head(ball_out)  # (B, T, 3)

        return out

    def predict(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
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

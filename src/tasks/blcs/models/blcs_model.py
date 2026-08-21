"""Main BLCS model implementation.
Ball Localization in Court System: Decoder-Only Transformer with MHA + RoPE + SDPA.
Estimates ball 3D trajectory in tennis court coordinates from 2D ball observations
and court keypoints.

Architecture:
    - Court keypoints and ball trajectory are tokenized and concatenated
    - Decoder-only Transformer with RoPE positional encoding
    - Multi-Head Self-Attention (MHA) using F.scaled_dot_product_attention (SDPA)
    - SwiGLU MLP and RMSNorm for efficiency

Uses shared components from src.utils.models.components.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.blcs.configuration import SingleModelConfig
from src.tasks.blcs.models.components.heads import build_trajectory_output
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis_nd,
    resolve_rope_bases,
)
from src.utils.models.embeddings import (
    BallUVEmbedding,
    CourtKPUVEmbedding,
    InvisibleTokenEmbedding,
)


class BLCSModel(nn.Module):
    """BLCS: Ball Localization in Court System.

    Decoder-Only Transformer architecture with:
    - Multi-Head Self-Attention (MHA) with SDPA for efficiency
    - Rotary Position Embedding (RoPE)
    - SwiGLU MLP and RMSNorm

    Tokens = [court_tokens(NUM_COURT_KP), ball_tokens(T)]
    Predicts 3D positions from ball tokens only.

    Input:
        - ball_uv: Ball 2D positions (u, v), shape (B, T, 2)
        - court_kp: Court 2D keypoints (20 landmarks), shape (B, 40) or (B, 20, 2)
        - ball_vis: Ball visibility flags, shape (B, T). Optional.
        - attention_mask: Adapter-prepared token attention mask.
        - court_vis: Court keypoint visibility, shape (B, 20). Optional.

    Output:
        - position: Normalized (x, y, z) trajectory, shape (B, T, 3)
        - velocity: Normalized velocities (optional), shape (B, T, 3)
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        rope_dim: int,
        rope_theta: float,
        rope_theta_time: float,
        rope_theta_camera: float,
        rope_theta_type: float,
        ffn_type: Literal["swiglu", "mlp"],
        predict_velocity: bool,
        max_seq_len: int,
        invisible_init_std: float,
        num_court_tokens: int,
    ) -> None:
        """Initialize the BLCS model.

        Args:
            hidden_dim: Hidden dimension for all components.
            num_layers: Number of Transformer blocks.
            num_heads: Number of query attention heads.
            ffn_dim: FFN intermediate dimension. Defaults to 8/3 * hidden_dim.
            dropout: Dropout probability.
            rope_dim: RoPE dimension. Defaults to head_dim.
            rope_theta: RoPE theta parameter.
            predict_velocity: Also predict velocities (for auxiliary loss).
            max_seq_len: Maximum sequence length.
            invisible_init_std: Initialization std for invisible tokens.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.num_court_tokens = int(num_court_tokens)
        self.max_tokens = int(self.num_court_tokens + self.max_seq_len)

        self._validate_init_args(hidden_dim=hidden_dim, num_heads=num_heads)
        head_dim = hidden_dim // num_heads

        self.rope_dim = int(rope_dim)
        self.rope_theta = float(rope_theta)
        self.rope_bases = resolve_rope_bases(
            rope_theta_time=rope_theta_time,
            rope_theta_camera=rope_theta_camera,
            rope_theta_type=rope_theta_type,
        )

        self.invisible_token = InvisibleTokenEmbedding(
            dim=hidden_dim, init_std=invisible_init_std
        )
        self.court_embed = CourtKPUVEmbedding(
            dim=hidden_dim,
            invisible_token=self.invisible_token,
        )
        self.ball_embed = BallUVEmbedding(
            dim=hidden_dim,
            invisible_token=self.invisible_token,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=hidden_dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        attention_type="mha",
                        n_kv_heads=None,
                        rope_base=self.rope_theta,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_dim)

        self.output_head = build_trajectory_output(
            input_dim=hidden_dim,
            dropout=dropout,
            predict_velocity=predict_velocity,
        )

        freqs_cis = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_rope_positions(),
            base=self.rope_bases,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @staticmethod
    def _validate_init_args(*, hidden_dim: int, num_heads: int) -> None:
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )

    @classmethod
    def from_config(cls, config: SingleModelConfig) -> BLCSModel:
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            BLCSModel: Initialized model.
        """
        return cls(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            rope_dim=config.rope_dim,
            rope_theta=config.rope_theta,
            rope_theta_time=config.rope_theta_time,
            rope_theta_camera=config.rope_theta_camera,
            rope_theta_type=config.rope_theta_type,
            ffn_type=config.ffn_type,
            predict_velocity=config.predict_velocity,
            max_seq_len=config.max_seq_len,
            invisible_init_std=config.invisible_init_std,
            num_court_tokens=config.num_court_tokens,
        )

    def _build_rope_positions(self) -> Tensor:
        """Build 3-axis RoPE positions for `[court, ball]` tokens."""
        court_idx = torch.arange(self.num_court_tokens, dtype=torch.long)
        ball_time = torch.arange(self.max_seq_len, dtype=torch.long) + 1

        court_pos = torch.stack(
            [
                torch.zeros_like(court_idx),
                court_idx,
                torch.zeros_like(court_idx),
            ],
            dim=-1,
        )
        ball_pos = torch.stack(
            [
                ball_time,
                torch.zeros_like(ball_time),
                torch.ones_like(ball_time),
            ],
            dim=-1,
        )
        return torch.cat([court_pos, ball_pos], dim=0)

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor,
        court_vis: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            ball_uv: Ball 2D positions, shape (B, T, 2).
            court_kp: Court keypoints, shape (B, 40) or (B, 20, 2).
            ball_vis: Ball visibility flags, shape (B, T).
            court_vis: Court visibility mask, shape (B, K).
            attention_mask: Adapter-prepared court/ball attention mask.

        Returns:
            dict: Dictionary with 'position' (B, T, 3) and optionally 'velocity'.
        """
        # Tokenize court and ball
        court_tok = self.court_embed(court_kp, court_vis)  # (B, K, D)
        ball_tok = self.ball_embed(ball_uv, ball_vis)  # (B, T, D)
        num_court_tokens = court_tok.shape[1]
        x = torch.cat([court_tok, ball_tok], dim=1)  # (B, S, D)
        freqs_cis = self.freqs_cis[: x.shape[1]]
        for blk in self.blocks:
            x = blk(
                x,
                freqs_cis=freqs_cis,
                attn_mask=attention_mask,
            )

        x = self.final_norm(x)
        ball_out = x[:, num_court_tokens:, :]  # (B, T, D)
        return cast("dict[str, Tensor]", self.output_head(ball_out))

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    freqs_cis: Tensor

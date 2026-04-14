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

from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.blcs.models.components.heads import Trajectory3DHead, VelocityHead
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis_nd,
)
from src.utils.models.embeddings import (
    BallUVEmbedding,
    CourtKPUVEmbedding,
    InvisibleTokenEmbedding,
)
from src.utils.schema.court import NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


# -------------------------
# Main model (legacy alias preserved)
# -------------------------


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
        - ball_mask: Ball padding mask, shape (B, T). Optional.
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
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_theta_time: float | None = None,
        rope_theta_camera: float | None = None,
        rope_theta_type: float = 100.0,
        ffn_type: Literal["swiglu", "mlp"] = "swiglu",
        predict_velocity: bool = False,
        max_seq_len: int = 120,
        invisible_init_std: float = 0.02,
        num_court_tokens: int = NUM_COURT_KP,
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
        self.predict_velocity = predict_velocity
        self.num_court_tokens = int(num_court_tokens)
        self.max_tokens = int(self.num_court_tokens + self.max_seq_len)

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        head_dim = hidden_dim // num_heads

        rope_dim = head_dim if rope_dim is None else rope_dim
        self.rope_dim = int(rope_dim)
        self.rope_theta = float(rope_theta)
        self.rope_bases = (
            float(self.rope_theta if rope_theta_time is None else rope_theta_time),
            float(self.rope_theta if rope_theta_camera is None else rope_theta_camera),
            float(rope_theta_type),
        )

        if ffn_dim is None:
            ffn_dim = int((8 * hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64  # Round to multiple of 64

        self.invisible_token = InvisibleTokenEmbedding(
            dim=hidden_dim, init_std=invisible_init_std
        )
        self.court_embed = CourtKPUVEmbedding(
            dim=hidden_dim,
            dropout=dropout,
            invisible_token=self.invisible_token,
        )
        self.ball_embed = BallUVEmbedding(
            dim=hidden_dim,
            dropout=dropout,
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
                        rope_base=self.rope_theta,
                        ffn_type=ffn_type,
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

        freqs_cis = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_rope_positions(),
            base=self.rope_bases,
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
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=model_cfg.get("dropout", 0.1),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=model_cfg.get("rope_theta", 10000.0),
            rope_theta_time=model_cfg.get("rope_theta_time", None),
            rope_theta_camera=model_cfg.get("rope_theta_camera", None),
            rope_theta_type=model_cfg.get("rope_theta_type", 100.0),
            ffn_type=cast(Literal["swiglu", "mlp"], str(model_cfg.get("ffn_type", "swiglu"))),
            predict_velocity=model_cfg.get("predict_velocity", False),
            max_seq_len=model_cfg.get(
                "max_seq_len", data_cfg.get("max_seq_len", 120)
            ),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            num_court_tokens=int(data_cfg.get("num_court_kp", NUM_COURT_KP)),
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
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            ball_uv: Ball 2D positions, shape (B, T, 2).
            court_kp: Court keypoints, shape (B, 40) or (B, 20, 2).
            ball_vis: Ball visibility flags, shape (B, T). Optional.
            ball_mask: Ball padding mask, shape (B, T). Optional.
            court_vis: Court visibility mask, shape (B, 20). Optional.

        Returns:
            dict: Dictionary with 'position' (B, T, 3) and optionally 'velocity'.
        """
        B, T, _ = ball_uv.shape

        # Tokenize court and ball
        court_tok = self.court_embed(court_kp, court_vis)  # (B, K, D)
        ball_tok = self.ball_embed(ball_uv, ball_vis)  # (B, T, D)
        if court_tok.shape[1] != self.num_court_tokens:
            raise ValueError(
                f"Expected {self.num_court_tokens} court tokens, got {court_tok.shape[1]}"
            )

        K = court_tok.shape[1]
        x = torch.cat([court_tok, ball_tok], dim=1)  # (B, S, D)
        S = x.shape[1]

        freqs_cis = cast(Tensor, self.freqs_cis)
        if freqs_cis.shape[0] < S:
            raise ValueError(
                f"Sequence length S={S} exceeds cached freqs_cis length {freqs_cis.shape[0]}. "
                "Increase max_seq_len."
            )
        freqs_cis = freqs_cis[:S]
        if freqs_cis.device != x.device:
            freqs_cis = freqs_cis.to(x.device)
        attn_mask: Tensor | None = None
        if ball_mask is not None:
            court_valid = torch.ones(B, K, device=x.device, dtype=torch.bool)
            ball_valid = ball_mask > 0
            key_padding_mask = torch.cat([court_valid, ball_valid], dim=1)
            attn_mask = key_padding_mask[:, None, :].expand(B, S, S)

        for blk in self.blocks:
            x = blk(
                x,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
            )

        x = self.final_norm(x)
        ball_out = x[:, K:, :]  # (B, T, D)

        out: dict[str, Tensor] = {"position": self.position_head(ball_out)}  # (B, T, 3)

        if self.predict_velocity and self.velocity_head is not None:
            out["velocity"] = self.velocity_head(ball_out)  # (B, T, 3)

        return out

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)

    model = BLCSModel(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        max_seq_len=16,
        predict_velocity=True,
    )

    B = 2
    T = 8
    ball_uv = torch.randn(B, T, 2)
    court_kp = torch.randn(B, NUM_COURT_KP, 2)
    ball_vis = (torch.rand(B, T) > 0.2).to(torch.float32)
    ball_mask = torch.ones(B, T)
    court_vis = (torch.rand(B, NUM_COURT_KP) > 0.1).to(torch.float32)

    with torch.no_grad():
        out = model(
            ball_uv=ball_uv,
            court_kp=court_kp,
            ball_vis=ball_vis,
            ball_mask=ball_mask,
            court_vis=court_vis,
        )

    print("BLCSModel:")
    for key, value in out.items():
        print(f"  {key}: {tuple(value.shape)}")

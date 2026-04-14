"""BLCS early-fusion model.

Builds a single per-frame token by concatenating:
    [ball_uv(2), court_kp_flat(20*2), court_vis(20)]
and applying an MLP embedding, then models only temporal tokens with a
decoder-style Transformer and regresses 3D trajectory coordinates.
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
    precompute_freqs_cis,
)
from src.utils.models.embeddings import InvisibleTokenEmbedding
from src.utils.schema.court import NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSEarlyFusionModel(nn.Module):
    """BLCS model with early fusion at embedding stage.

    Input token at each frame t:
        fused_t = MLP(cat(ball_uv_t, court_kp_flat))

    Temporal tokens are processed by Transformer blocks, and each frame token
    is decoded to 3D position (and optional velocity).
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
        ffn_type: Literal["swiglu", "mlp"] = "swiglu",
        predict_velocity: bool = False,
        max_seq_len: int = 120,
        invisible_init_std: float = 0.02,
    ) -> None:
        """Initialize early-fusion BLCS model."""
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.predict_velocity = bool(predict_velocity)

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        head_dim = hidden_dim // num_heads

        rope_dim = head_dim if rope_dim is None else int(rope_dim)
        self.rope_dim = int(rope_dim)
        self.rope_theta = float(rope_theta)
        self.rope_time_base = float(self.rope_theta if rope_theta_time is None else rope_theta_time)

        if ffn_dim is None:
            ffn_dim = int((8 * hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        self.fusion_input_dim = int(2 + NUM_COURT_KP * 2 + NUM_COURT_KP)
        self.invisible_token = InvisibleTokenEmbedding(
            dim=hidden_dim,
            init_std=float(invisible_init_std),
        )
        self.fusion_embed = nn.Sequential(
            nn.Linear(self.fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(float(dropout)),
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
                        rope_base=self.rope_time_base,
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

        freqs_cis = precompute_freqs_cis(
            dim=self.rope_dim,
            seqlen=self.max_seq_len,
            base=self.rope_time_base,
            device=None,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> BLCSEarlyFusionModel:
        """Create model from configuration."""
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            rope_theta_time=model_cfg.get("rope_theta_time", None),
            ffn_type=cast(Literal["swiglu", "mlp"], str(model_cfg.get("ffn_type", "swiglu"))),
            predict_velocity=bool(model_cfg.get("predict_velocity", False)),
            max_seq_len=int(model_cfg.get("max_seq_len", data_cfg.get("max_seq_len", 120))),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
        )

    @staticmethod
    def _flatten_court_kp(court_kp: Tensor) -> Tensor:
        """Flatten court keypoints to shape (B, 40)."""
        if court_kp.dim() == 2:
            if court_kp.shape[1] != NUM_COURT_KP * 2:
                raise ValueError(
                    f"court_kp with dim=2 must have shape (B, {NUM_COURT_KP * 2}), "
                    f"got {tuple(court_kp.shape)}"
                )
            return court_kp
        if court_kp.dim() == 3:
            if court_kp.shape[1:] != (NUM_COURT_KP, 2):
                raise ValueError(
                    f"court_kp with dim=3 must have shape (B, {NUM_COURT_KP}, 2), "
                    f"got {tuple(court_kp.shape)}"
                )
            return court_kp.reshape(court_kp.shape[0], NUM_COURT_KP * 2)
        raise ValueError(
            f"court_kp must be (B, {NUM_COURT_KP * 2}) or (B, {NUM_COURT_KP}, 2), "
            f"got {tuple(court_kp.shape)}"
        )

    @staticmethod
    def _prepare_court_vis(court_vis: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        """Prepare court visibility as shape (B, NUM_COURT_KP)."""
        if court_vis is None:
            return torch.ones(batch_size, NUM_COURT_KP, device=device)
        if court_vis.shape != (batch_size, NUM_COURT_KP):
            raise ValueError(
                f"court_vis must have shape {(batch_size, NUM_COURT_KP)}, "
                f"got {tuple(court_vis.shape)}"
            )
        return court_vis

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
        """
        batch_size, seq_len, input_dim = ball_uv.shape
        if input_dim != 2:
            raise ValueError(f"ball_uv last dim must be 2, got {input_dim}")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}. "
                "Increase model.max_seq_len."
            )

        court_flat = self._flatten_court_kp(court_kp)
        court_vis_feat = self._prepare_court_vis(
            court_vis=court_vis,
            batch_size=batch_size,
            device=ball_uv.device,
        ).to(dtype=ball_uv.dtype)
        court_rep = court_flat.unsqueeze(1).expand(-1, seq_len, -1)
        court_vis_rep = court_vis_feat.unsqueeze(1).expand(-1, seq_len, -1)
        fused_input = torch.cat([ball_uv, court_rep, court_vis_rep], dim=-1)
        x = self.fusion_embed(fused_input)

        if ball_vis is not None:
            visible = (ball_vis > 0).unsqueeze(-1)
            inv = self.invisible_token().to(dtype=x.dtype, device=x.device)
            inv = inv.view(1, 1, -1).expand_as(x)
            x = torch.where(visible, x, inv)

        freqs_cis = self.freqs_cis[:seq_len]
        if freqs_cis.device != x.device:
            freqs_cis = freqs_cis.to(x.device)

        attn_mask: Tensor | None = None
        if ball_mask is not None:
            valid = (ball_mask > 0).bool()
            fully_masked = ~valid.any(dim=1)
            if fully_masked.any():
                valid = valid.clone()
                valid[fully_masked, 0] = True
            attn_mask = valid[:, None, :].expand(batch_size, seq_len, seq_len)

        for blk in self.blocks:
            x = blk(
                x,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
            )

        x = self.final_norm(x)

        out: dict[str, Tensor] = {"position": self.position_head(x)}
        if self.predict_velocity and self.velocity_head is not None:
            out["velocity"] = self.velocity_head(x)
        return out

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

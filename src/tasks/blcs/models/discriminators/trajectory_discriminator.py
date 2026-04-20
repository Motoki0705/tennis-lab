"""Transformer discriminator for BLCS 3D trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn

from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis,
)
from src.utils.models.embeddings import Ball3DEmbedding, InvisibleTokenEmbedding

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSTrajectoryDiscriminator(nn.Module):
    """Transformer discriminator over 3D trajectory tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        ffn_type: str = "swiglu",
        max_seq_len: int = 120,
        invisible_init_std: float = 0.02,
        cls_init_std: float = 0.02,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        head_dim = hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else int(rope_dim)
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}")

        if ffn_dim is None:
            ffn_dim = int((8 * hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.rope_dim = int(rope_dim)

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim,
            init_std=float(invisible_init_std),
        )
        self.ball_embed = Ball3DEmbedding(
            dim=self.hidden_dim,
            dropout=float(dropout),
            invisible_token=self.invisible_token,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * float(cls_init_std))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        ffn_dim=int(ffn_dim),
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=float(dropout),
                        rope_base=float(rope_theta),
                        ffn_type=cast(Literal["swiglu", "mlp"], ffn_type),
                    )
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = RMSNorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, 1)

        freqs_cis = precompute_freqs_cis(
            dim=self.rope_dim,
            seqlen=self.max_seq_len + 1,
            base=float(rope_theta),
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> BLCSTrajectoryDiscriminator:
        """Build discriminator from training.gan.discriminator config."""
        train_cfg = config.get("training", {}) or {}
        gan_cfg = train_cfg.get("gan", {}) or {}
        disc_cfg = gan_cfg.get("discriminator", {}) or {}
        data_cfg = config.get("data", {}) or {}

        return cls(
            hidden_dim=int(disc_cfg.get("hidden_dim", 128)),
            num_layers=int(disc_cfg.get("num_layers", 4)),
            num_heads=int(disc_cfg.get("num_heads", 4)),
            ffn_dim=disc_cfg.get("ffn_dim", None),
            dropout=float(disc_cfg.get("dropout", 0.1)),
            rope_dim=disc_cfg.get("rope_dim", None),
            rope_theta=float(disc_cfg.get("rope_theta", 10000.0)),
            ffn_type=str(disc_cfg.get("ffn_type", "swiglu")),
            max_seq_len=int(disc_cfg.get("max_seq_len", data_cfg.get("max_seq_len", 120))),
            invisible_init_std=float(disc_cfg.get("invisible_init_std", 0.02)),
            cls_init_std=float(disc_cfg.get("cls_init_std", 0.02)),
        )

    def forward(
        self,
        position_3d: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Score 3D trajectories as real/fake.

        Args:
            position_3d: Trajectory tensor of shape (B, T, 3).
            mask: Valid-timestep mask of shape (B, T).

        Returns:
            Tensor of shape (B,) containing discriminator logits.
        """
        batch_size, seq_len, _ = position_3d.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        seq_mask = mask
        if seq_mask is None:
            seq_mask = torch.ones(batch_size, seq_len, device=position_3d.device, dtype=torch.bool)
        else:
            seq_mask = seq_mask > 0

        x = self.ball_embed(position_3d, seq_mask.to(dtype=position_3d.dtype))
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        freqs_cis = cast(Tensor, self.freqs_cis[: seq_len + 1])
        if freqs_cis.device != x.device:
            freqs_cis = freqs_cis.to(x.device)

        cls_valid = torch.ones(batch_size, 1, device=x.device, dtype=torch.bool)
        attn_valid = torch.cat([cls_valid, seq_mask], dim=1)
        attn_mask = attn_valid[:, None, :].expand(batch_size, seq_len + 1, seq_len + 1)

        for block in self.blocks:
            x = cast(Tensor, block(x, freqs_cis=freqs_cis, attn_mask=attn_mask))

        x = cast(Tensor, self.final_norm(x))
        logits = cast(Tensor, self.head(x[:, 0, :]))
        return logits.squeeze(-1)
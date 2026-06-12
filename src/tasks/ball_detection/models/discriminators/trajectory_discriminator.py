"""Transformer discriminator wrapper for ball 2D trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from src.utils.models.architectures import TransformerSequenceDiscriminator

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BallTrajectoryDiscriminator(TransformerSequenceDiscriminator):
    """Score normalized image-space ball trajectories as real or fake."""

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
        max_seq_len: int = 64,
        invisible_init_std: float = 0.02,
        cls_init_std: float = 0.02,
    ) -> None:
        super().__init__(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            rope_dim=rope_dim,
            rope_theta=rope_theta,
            ffn_type=ffn_type,
            max_seq_len=max_seq_len,
            invalid_init_std=invisible_init_std,
            cls_init_std=cls_init_std,
        )

    @classmethod
    def from_config(cls, config: DictConfig) -> BallTrajectoryDiscriminator:
        """Build discriminator from training.gan.discriminator config."""
        train_cfg = config.get("training", {}) or {}
        gan_cfg = train_cfg.get("gan", {}) or {}
        disc_cfg = gan_cfg.get("discriminator", {}) or {}
        model_cfg = config.get("model", {}) or {}

        return cls(
            hidden_dim=int(disc_cfg.get("hidden_dim", 128)),
            num_layers=int(disc_cfg.get("num_layers", 4)),
            num_heads=int(disc_cfg.get("num_heads", 4)),
            ffn_dim=disc_cfg.get("ffn_dim", None),
            dropout=float(disc_cfg.get("dropout", 0.1)),
            rope_dim=disc_cfg.get("rope_dim", None),
            rope_theta=float(disc_cfg.get("rope_theta", 10000.0)),
            ffn_type=str(disc_cfg.get("ffn_type", "swiglu")),
            max_seq_len=int(disc_cfg.get("max_seq_len", model_cfg.get("num_frames", 8))),
            invisible_init_std=float(disc_cfg.get("invisible_init_std", 0.02)),
            cls_init_std=float(disc_cfg.get("cls_init_std", 0.02)),
        )

    def forward(self, ball_xy: Tensor, *, mask: Tensor | None = None) -> Tensor:
        """Score normalized ball coordinate sequences as real or fake."""
        return super().forward(ball_xy, mask=mask)

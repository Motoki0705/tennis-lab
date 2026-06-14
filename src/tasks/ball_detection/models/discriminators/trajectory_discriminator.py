"""Transformer discriminator wrapper for ball 2D trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from src.utils.models.architectures import (
    TransformerSequenceDiscriminator,
    build_trajectory_discriminator,
)

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
    def from_config(cls, config: DictConfig) -> TransformerSequenceDiscriminator:
        """Build discriminator from ``training.gan.discriminator`` config.

        Delegates kwarg parsing to the shared
        :func:`build_trajectory_discriminator` factory (``input_dim=2``).
        """
        train_cfg = config.get("training", {}) or {}
        gan_cfg = train_cfg.get("gan", {}) or {}
        disc_cfg = gan_cfg.get("discriminator", {}) or {}
        model_cfg = config.get("model", {}) or {}

        return build_trajectory_discriminator(
            input_dim=2,
            disc_cfg=disc_cfg,
            default_max_seq_len=int(model_cfg.get("num_frames", 8)),
        )

    def forward(self, ball_xy: Tensor, *, mask: Tensor | None = None) -> Tensor:
        """Score normalized ball coordinate sequences as real or fake."""
        return super().forward(ball_xy, mask=mask)

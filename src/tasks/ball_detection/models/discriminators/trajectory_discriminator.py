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
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        rope_dim: int,
        rope_theta: float,
        ffn_type: str,
        max_seq_len: int,
        invalid_init_std: float,
        cls_init_std: float,
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
            invalid_init_std=invalid_init_std,
            cls_init_std=cls_init_std,
        )

    @classmethod
    def from_config(cls, config: DictConfig) -> TransformerSequenceDiscriminator:
        """Build discriminator from ``training.gan.discriminator`` config.

        Delegates kwarg parsing to the shared
        :func:`build_trajectory_discriminator` factory (``input_dim=2``).
        """
        return build_trajectory_discriminator(
            input_dim=2,
            disc_cfg=config.training.gan.discriminator,
        )

    def forward(self, ball_xy: Tensor, *, mask: Tensor | None = None) -> Tensor:
        """Score normalized ball coordinate sequences as real or fake."""
        return super().forward(ball_xy, mask=mask)

"""Transformer discriminator wrapper for BLCS 3D trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from src.utils.models.architectures import (
    TransformerSequenceDiscriminator,
    build_trajectory_discriminator,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSTrajectoryDiscriminator(TransformerSequenceDiscriminator):
    """BLCS-compatible wrapper over the shared sequence discriminator.

    Scores 3D ball trajectories (``input_dim=3``).
    """

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
        super().__init__(
            input_dim=3,
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
        :func:`build_trajectory_discriminator` factory (``input_dim=3``).
        """
        train_cfg = config.get("training", {}) or {}
        gan_cfg = train_cfg.get("gan", {}) or {}
        disc_cfg = gan_cfg.get("discriminator", {}) or {}
        data_cfg = config.get("data", {}) or {}

        return build_trajectory_discriminator(
            input_dim=3,
            disc_cfg=disc_cfg,
            default_max_seq_len=int(data_cfg.get("max_seq_len", 120)),
        )

    def forward(self, position_3d: Tensor, *, mask: Tensor | None = None) -> Tensor:
        """Score BLCS 3D trajectories as real or fake."""
        return super().forward(position_3d, mask=mask)

"""Transformer discriminator wrapper for BLCS 3D trajectories."""

from __future__ import annotations

from torch import Tensor

from src.tasks.base.configuration import as_config_mapping, require_config_mapping
from src.utils.models.architectures import (
    TransformerSequenceDiscriminator,
    build_trajectory_discriminator,
)


class BLCSTrajectoryDiscriminator(TransformerSequenceDiscriminator):
    """BLCS-compatible wrapper over the shared sequence discriminator.

    Scores 3D ball trajectories (``input_dim=3``).
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
        ffn_type: str,
        max_seq_len: int,
        invalid_init_std: float,
        cls_init_std: float,
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
            invalid_init_std=invalid_init_std,
            cls_init_std=cls_init_std,
        )

    @classmethod
    def from_config(cls, config: object) -> TransformerSequenceDiscriminator:
        """Build discriminator from ``training.gan.discriminator`` config.

        Delegates kwarg parsing to the shared
        :func:`build_trajectory_discriminator` factory (``input_dim=3``).
        """
        root = as_config_mapping(config, path="configuration")
        training = require_config_mapping(root, "training", path="configuration")
        gan = require_config_mapping(training, "gan", path="training")
        disc_cfg = require_config_mapping(gan, "discriminator", path="training.gan")

        shared = {
            key: disc_cfg[key]
            for key in (
                "hidden_dim",
                "num_layers",
                "num_heads",
                "ffn_dim",
                "dropout",
                "rope_dim",
                "rope_theta",
                "ffn_type",
                "max_seq_len",
                "invalid_init_std",
                "cls_init_std",
            )
        }
        return build_trajectory_discriminator(
            input_dim=3,
            disc_cfg=shared,
        )

    def forward(self, position_3d: Tensor, *, mask: Tensor | None = None) -> Tensor:
        """Score BLCS 3D trajectories as real or fake."""
        return super().forward(position_3d, mask=mask)

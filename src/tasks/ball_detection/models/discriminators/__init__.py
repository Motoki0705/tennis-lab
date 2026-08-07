"""Ball detection discriminator modules."""

from __future__ import annotations

from omegaconf import DictConfig

from src.utils.models.architectures import (
    TransformerSequenceDiscriminator,
    build_trajectory_discriminator,
)


def build_ball_detection_discriminator(
    config: DictConfig,
) -> TransformerSequenceDiscriminator:
    """Build the configured ball detection discriminator."""
    disc_name = str(config.training.gan.discriminator.name)
    if disc_name != "trajectory_transformer":
        raise ValueError(
            "Unknown ball_detection discriminator name="
            f"'{disc_name}'. Supported: ['trajectory_transformer']"
        )
    return build_trajectory_discriminator(
        input_dim=2,
        disc_cfg=config.training.gan.discriminator,
    )


__all__ = [
    "build_ball_detection_discriminator",
]

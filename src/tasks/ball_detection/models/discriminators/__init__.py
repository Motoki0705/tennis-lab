"""Ball detection discriminator modules."""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.ball_detection.models.discriminators.trajectory_discriminator import (
    BallTrajectoryDiscriminator,
)
from src.utils.models.architectures import TransformerSequenceDiscriminator


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
    return BallTrajectoryDiscriminator.from_config(config)


__all__ = [
    "BallTrajectoryDiscriminator",
    "build_ball_detection_discriminator",
]

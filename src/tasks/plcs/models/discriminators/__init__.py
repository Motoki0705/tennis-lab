"""PLCS discriminator modules."""

from __future__ import annotations

from src.tasks.base.configuration import require_config_mapping, require_config_value
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.models.discriminators.pose_sequence_discriminator import (
    PLCSPoseSequenceDiscriminator,
)


def build_plcs_discriminator(
    config: PLCSTrainingConfig,
) -> PLCSPoseSequenceDiscriminator:
    """Build the configured PLCS discriminator."""
    training = require_config_mapping(config.raw, "training", path="configuration")
    gan = require_config_mapping(training, "gan", path="training")
    discriminator = require_config_mapping(gan, "discriminator", path="training.gan")
    disc_name = str(
        require_config_value(
            discriminator, "name", str, path="training.gan.discriminator"
        )
    )
    if disc_name != "pose_sequence_transformer":
        raise ValueError(
            "Unknown PLCS discriminator name="
            f"'{disc_name}'. Supported: ['pose_sequence_transformer']"
        )
    return PLCSPoseSequenceDiscriminator.from_config(discriminator)


__all__ = [
    "PLCSPoseSequenceDiscriminator",
    "build_plcs_discriminator",
]

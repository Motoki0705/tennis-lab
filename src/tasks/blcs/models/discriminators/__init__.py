"""BLCS discriminator modules."""

from __future__ import annotations

from src.tasks.base.configuration import as_config_mapping, require_config_mapping
from src.tasks.blcs.models.discriminators.trajectory_discriminator import (
    BLCSTrajectoryDiscriminator,
)
from src.utils.models.architectures import TransformerSequenceDiscriminator


def build_blcs_discriminator(config: object) -> TransformerSequenceDiscriminator:
    """Build the configured BLCS discriminator."""
    root = as_config_mapping(config, path="configuration")
    training = require_config_mapping(root, "training", path="configuration")
    gan = require_config_mapping(training, "gan", path="training")
    discriminator = require_config_mapping(gan, "discriminator", path="training.gan")
    disc_name = str(discriminator["name"])
    if disc_name != "trajectory_transformer":
        raise ValueError(
            "Unknown BLCS discriminator name="
            f"'{disc_name}'. Supported: ['trajectory_transformer']"
        )
    return BLCSTrajectoryDiscriminator.from_config(config)


__all__ = [
    "BLCSTrajectoryDiscriminator",
    "build_blcs_discriminator",
]

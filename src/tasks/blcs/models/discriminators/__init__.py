"""BLCS discriminator modules."""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.blcs.models.discriminators.trajectory_discriminator import (
    BLCSTrajectoryDiscriminator,
)
from src.utils.models.architectures import TransformerSequenceDiscriminator


def build_blcs_discriminator(config: DictConfig) -> TransformerSequenceDiscriminator:
    """Build the configured BLCS discriminator."""
    train_cfg = config.get("training", {}) or {}
    gan_cfg = train_cfg.get("gan", {}) or {}
    disc_name = str(gan_cfg.get("discriminator", {}).get("name", "trajectory_transformer"))
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
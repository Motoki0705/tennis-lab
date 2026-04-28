"""PLCS discriminator modules."""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.plcs.models.discriminators.pose_sequence_discriminator import (
    PLCSPoseSequenceDiscriminator,
)


def build_plcs_discriminator(config: DictConfig) -> PLCSPoseSequenceDiscriminator:
    """Build the configured PLCS discriminator."""
    train_cfg = config.get("training", {}) or {}
    gan_cfg = train_cfg.get("gan", {}) or {}
    disc_name = str(gan_cfg.get("discriminator", {}).get("name", "pose_sequence_transformer"))
    if disc_name != "pose_sequence_transformer":
        raise ValueError(
            "Unknown PLCS discriminator name="
            f"'{disc_name}'. Supported: ['pose_sequence_transformer']"
        )
    return PLCSPoseSequenceDiscriminator.from_config(config)


__all__ = [
    "PLCSPoseSequenceDiscriminator",
    "build_plcs_discriminator",
]
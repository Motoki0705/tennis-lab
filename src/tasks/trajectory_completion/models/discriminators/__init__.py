"""Trajectory completion discriminator modules."""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.trajectory_completion.models.discriminators.uv_sequence_discriminator import (
    UVTrajectorySequenceDiscriminator,
)


def build_trajectory_completion_discriminator(
    config: DictConfig,
) -> UVTrajectorySequenceDiscriminator:
    """Build the configured trajectory completion discriminator."""
    train_cfg = config.get("training", {}) or {}
    gan_cfg = train_cfg.get("gan", {}) or {}
    disc_name = str(gan_cfg.get("discriminator", {}).get("name", "uv_sequence_transformer"))
    if disc_name != "uv_sequence_transformer":
        raise ValueError(
            "Unknown trajectory completion discriminator name="
            f"'{disc_name}'. Supported: ['uv_sequence_transformer']"
        )
    return UVTrajectorySequenceDiscriminator.from_config(config)


__all__ = [
    "UVTrajectorySequenceDiscriminator",
    "build_trajectory_completion_discriminator",
]
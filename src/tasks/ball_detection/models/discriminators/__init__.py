"""Ball detection discriminator modules."""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.ball_detection.models.discriminators.trajectory_discriminator import (
    BallTrajectoryDiscriminator,
)


def build_ball_detection_discriminator(config: DictConfig) -> BallTrajectoryDiscriminator:
    """Build the configured ball detection discriminator."""
    train_cfg = config.get("training", {}) or {}
    gan_cfg = train_cfg.get("gan", {}) or {}
    disc_name = str(gan_cfg.get("discriminator", {}).get("name", "trajectory_transformer"))
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

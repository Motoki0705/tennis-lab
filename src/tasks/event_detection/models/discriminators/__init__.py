"""Event detection discriminator modules."""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.event_detection.models.discriminators.event_sequence_discriminator import (
    EventSequenceDiscriminator,
)


def build_event_detection_discriminator(config: DictConfig) -> EventSequenceDiscriminator:
    """Build the configured event detection discriminator."""
    train_cfg = config.get("training", {}) or {}
    gan_cfg = train_cfg.get("gan", {}) or {}
    disc_name = str(gan_cfg.get("discriminator", {}).get("name", "event_sequence_transformer"))
    if disc_name != "event_sequence_transformer":
        raise ValueError(
            "Unknown event detection discriminator name="
            f"'{disc_name}'. Supported: ['event_sequence_transformer']"
        )
    return EventSequenceDiscriminator.from_config(config)


__all__ = [
    "EventSequenceDiscriminator",
    "build_event_detection_discriminator",
]
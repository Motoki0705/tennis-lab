"""BLCS discriminator modules."""

from __future__ import annotations

from src.tasks.base.configuration import as_config_mapping, require_config_mapping
from src.utils.models.architectures import (
    TransformerSequenceDiscriminator,
    build_trajectory_discriminator,
)


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
    shared = {
        key: discriminator[key]
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
    return build_trajectory_discriminator(input_dim=3, disc_cfg=shared)


__all__ = [
    "build_blcs_discriminator",
]

"""PLCS discriminator modules."""

from __future__ import annotations

from src.tasks.base.configuration import require_config_mapping, require_config_value
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.utils.models.architectures import (
    TransformerSequenceDiscriminator,
    build_trajectory_discriminator,
)


def build_plcs_discriminator(
    config: PLCSTrainingConfig,
) -> TransformerSequenceDiscriminator:
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
    return build_trajectory_discriminator(input_dim=5, disc_cfg=shared)


__all__ = [
    "build_plcs_discriminator",
]

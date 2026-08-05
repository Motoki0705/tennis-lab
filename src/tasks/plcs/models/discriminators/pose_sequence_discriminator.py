"""Transformer discriminator for PLCS pose sequences."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from src.tasks.base.configuration import require_config_value
from src.utils.models.architectures import TransformerSequenceDiscriminator


class PLCSPoseSequenceDiscriminator(TransformerSequenceDiscriminator):
    """Discriminator over concatenated PLCS position and rotation outputs."""

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> PLCSPoseSequenceDiscriminator:
        return cls(
            input_dim=5,
            hidden_dim=cast(
                "int",
                require_config_value(
                    config, "hidden_dim", int, path="training.gan.discriminator"
                ),
            ),
            num_layers=cast(
                "int",
                require_config_value(
                    config, "num_layers", int, path="training.gan.discriminator"
                ),
            ),
            num_heads=cast(
                "int",
                require_config_value(
                    config, "num_heads", int, path="training.gan.discriminator"
                ),
            ),
            ffn_dim=cast(
                "int",
                require_config_value(
                    config,
                    "ffn_dim",
                    int,
                    path="training.gan.discriminator",
                ),
            ),
            dropout=float(
                cast(
                    "float | int",
                    require_config_value(
                        config,
                        "dropout",
                        (float, int),
                        path="training.gan.discriminator",
                    ),
                )
            ),
            rope_dim=cast(
                "int",
                require_config_value(
                    config,
                    "rope_dim",
                    int,
                    path="training.gan.discriminator",
                ),
            ),
            rope_theta=float(
                cast(
                    "float | int",
                    require_config_value(
                        config,
                        "rope_theta",
                        (float, int),
                        path="training.gan.discriminator",
                    ),
                )
            ),
            ffn_type=cast(
                "str",
                require_config_value(
                    config, "ffn_type", str, path="training.gan.discriminator"
                ),
            ),
            max_seq_len=cast(
                "int",
                require_config_value(
                    config, "max_seq_len", int, path="training.gan.discriminator"
                ),
            ),
            invalid_init_std=float(
                cast(
                    "float | int",
                    require_config_value(
                        config,
                        "invalid_init_std",
                        (float, int),
                        path="training.gan.discriminator",
                    ),
                )
            ),
            cls_init_std=float(
                cast(
                    "float | int",
                    require_config_value(
                        config,
                        "cls_init_std",
                        (float, int),
                        path="training.gan.discriminator",
                    ),
                )
            ),
        )

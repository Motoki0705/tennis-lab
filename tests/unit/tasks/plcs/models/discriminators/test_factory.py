"""Tests for the canonical shared PLCS discriminator composition."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.models.discriminators import build_plcs_discriminator
from src.utils.models.architectures import TransformerSequenceDiscriminator
from src.utils.paths import PROJECT_ROOT


def test_plcs_factory_uses_shared_transformer_builder_without_task_subclass() -> None:
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="train_chunked_gan")

    discriminator = build_plcs_discriminator(
        PLCSTrainingConfig.from_config(config)
    )

    assert type(discriminator) is TransformerSequenceDiscriminator
    assert discriminator.input_dim == 5
    old_module = Path(
        "src/tasks/plcs/models/discriminators/pose_sequence_discriminator.py"
    )
    assert not old_module.exists()

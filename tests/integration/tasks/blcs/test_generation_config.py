from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.blcs.configuration import (
    validate_generation_boundary,
    validate_training_boundary,
)
from src.utils.configuration import SemanticConfigurationError

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


@pytest.mark.parametrize(
    "config_name", ("generate_dataset", "train_tracking_chunked")
)
def test_multi_object_generation_has_explicit_bounded_physics_budget(
    config_name: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name=config_name,
            overrides=["generation=multi_object"],
        )

    assert config.generation.maximum_physics_attempts_per_object == 64
    if config_name == "generate_dataset":
        validate_generation_boundary(config)
    else:
        validate_training_boundary(config)


@pytest.mark.parametrize("maximum_attempts", (0, -1))
def test_multi_object_generation_rejects_nonpositive_physics_budget(
    maximum_attempts: int,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="generate_dataset",
            overrides=[
                "generation=multi_object",
                (
                    "generation.maximum_physics_attempts_per_object="
                    f"{maximum_attempts}"
                ),
            ],
        )

    with pytest.raises(
        SemanticConfigurationError,
        match="maximum_physics_attempts_per_object must be positive",
    ):
        validate_generation_boundary(config)

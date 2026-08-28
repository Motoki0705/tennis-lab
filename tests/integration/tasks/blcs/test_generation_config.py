from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.tasks.blcs.configuration import (
    parse_court_keypoint_contract,
    validate_generation_boundary,
    validate_training_boundary,
)
from src.utils.configuration import ConfigurationTypeError, SemanticConfigurationError

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


@pytest.mark.parametrize(
    ("selector", "contract_id"),
    [
        ("physical_v1", "physical_courtkp20_v1"),
        ("camera_view_v2", "camera_view_courtkp20_rzpi_v1"),
    ],
)
def test_generation_composes_explicit_court_keypoint_selector(
    selector: str,
    contract_id: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="generate_dataset",
            overrides=[f"court_keypoints={selector}"],
        )
    validate_generation_boundary(config)
    assert parse_court_keypoint_contract(config).contract_id == contract_id


def test_generation_default_court_keypoints_remains_physical_v1() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="generate_dataset")
    assert config.court_keypoints.selector == "physical_v1"


def test_generation_rejects_unknown_or_untyped_court_selector() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="generate_dataset")
    for invalid in ("v2", 2):
        with open_dict(config.court_keypoints):
            config.court_keypoints.selector = invalid
        with pytest.raises((SemanticConfigurationError, ConfigurationTypeError)):
            validate_generation_boundary(config)


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

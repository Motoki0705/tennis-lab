from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.generate_dataset.config import PLCSGenerationConfig
from src.utils.configuration import UnknownConfigurationKeyError


@pytest.mark.parametrize("court_selector", ["physical_v1", "camera_view_v2"])
def test_generation_composes_court_keypoint_contract(
    court_selector: str,
) -> None:
    config_dir = Path("src/tasks/plcs/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="generate_dataset",
            overrides=[
                f"court_keypoints={court_selector}",
                "run.device=cpu",
                "run.num_workers=1",
                "simulation.num_scenes=1",
            ],
        )
    runtime = PLCSGenerationConfig.from_config(config)
    assert runtime.court_keypoint_contract.selector == court_selector


def test_generation_rejects_unknown_typed_selector() -> None:
    config_dir = Path("src/tasks/plcs/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="generate_dataset",
            overrides=["court_keypoints.selector=unknown"],
        )
    with pytest.raises(ValueError, match="Unknown court keypoint selector"):
        PLCSGenerationConfig.from_config(config)


def test_generation_rejects_unknown_selector_field() -> None:
    config_dir = Path("src/tasks/plcs/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="generate_dataset",
            overrides=["+court_keypoints.fallback=true"],
        )
    with pytest.raises(UnknownConfigurationKeyError):
        PLCSGenerationConfig.from_config(config)


@pytest.mark.parametrize("config_name", ["train", "train_tracking"])
@pytest.mark.parametrize("court_selector", ["physical_v1", "camera_view_v2"])
def test_training_composes_court_keypoint_contract(
    config_name: str,
    court_selector: str,
) -> None:
    config_dir = Path("src/tasks/plcs/configs").resolve()
    overrides = [f"court_keypoints={court_selector}"]
    if config_name == "train_tracking" and court_selector == "camera_view_v2":
        overrides.append("model=track_query_reference")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name=config_name,
            overrides=overrides,
        )

    runtime = PLCSTrainingConfig.from_config(config)
    assert runtime.court_keypoint_contract.selector == court_selector

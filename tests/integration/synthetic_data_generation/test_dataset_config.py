"""Integration tests for the generic synthetic-data Hydra configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.synthetic_data_generation.dataset.pipeline import PathPipelineManifest
from src.utils.paths import PROJECT_ROOT


def test_dataset_config_composes_to_conventional_path_layout() -> None:
    config_dir = PROJECT_ROOT / "src/synthetic_data_generation/configs/dataset"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="pipeline")
        paths = OmegaConf.to_container(config.paths, resolve=True)
        assert isinstance(paths, dict)
        manifest = PathPipelineManifest.from_config(
            cast(Mapping[str, object], paths),
            project_root=PROJECT_ROOT,
        )

    assert manifest.source_root == PROJECT_ROOT / "third_party/nht/data"
    assert manifest.artifact_root == (
        PROJECT_ROOT / "third_party/nht/artifacts/synthetic-data"
    )
    assert manifest.execution_root == (
        PROJECT_ROOT / "outputs/synthetic_data_generation"
    )
    assert manifest.dataset_root == PROJECT_ROOT / "data/synthetic_data_generation"

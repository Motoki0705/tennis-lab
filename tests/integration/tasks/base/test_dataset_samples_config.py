"""Integration tests for PLCS/BLCS dataset-sample Hydra composition."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.base.generate_dataset.dataset_samples import DatasetSamplesConfig
from src.utils.configuration import SemanticConfigurationError


@pytest.mark.parametrize("task", ("plcs", "blcs"))
def test_default_sample_config_covers_all_five_canonical_datasets(
    task: str,
    tmp_path: Path,
) -> None:
    config_dir = Path(f"src/tasks/{task}/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="generate_dataset_samples",
            overrides=[f"paths.data_root={tmp_path.as_posix()}"],
        )

    runtime = DatasetSamplesConfig.from_config(config, task=task)  # type: ignore[arg-type]

    assert [spec.relative_path for spec in runtime.datasets] == [
        f"{task}/single_object",
        f"{task}/multi_object",
        f"{task}/single_object_broadcast",
        f"{task}/multi_object_broadcast",
        f"{task}/multi_object_camera_view_v2",
    ]
    assert [spec.mode for spec in runtime.datasets] == [
        "single",
        "multi",
        "single",
        "multi",
        "multi",
    ]
    assert [spec.court_keypoint_contract.selector for spec in runtime.datasets] == [
        "physical_v1",
        "physical_v1",
        "physical_v1",
        "physical_v1",
        "camera_view_v2",
    ]
    assert all(spec.root.is_relative_to(tmp_path) for spec in runtime.datasets)
    assert runtime.max_frames == 120
    assert runtime.view == "camera"


@pytest.mark.parametrize("task", ("plcs", "blcs"))
def test_sample_config_rejects_non_camera_rendering(task: str) -> None:
    config_dir = Path(f"src/tasks/{task}/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="generate_dataset_samples",
            overrides=["samples.view=3d"],
        )

    with pytest.raises(SemanticConfigurationError, match="must be 'camera'"):
        DatasetSamplesConfig.from_config(config, task=task)  # type: ignore[arg-type]

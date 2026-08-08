"""PLCS task configuration does not retain a second generation authority."""

from __future__ import annotations

import src.tasks.plcs.configuration_contracts as configuration_contracts
from src.utils.paths import PROJECT_ROOT


def _path_root() -> dict[str, object]:
    return {
        "paths": {
            "project_root": ".",
            "data_root": "data",
            "checkpoint_root": "ckpt",
            "artifact_root": "artifacts",
            "output_root": "outputs",
            "cache_root": ".cache",
            "external_asset_root": "/home/kamimura/projects",
        },
    }


def test_task_contract_exposes_paths_without_generation_compatibility() -> None:
    source = (PROJECT_ROOT / "src/tasks/plcs/configuration_contracts.py").read_text(
        encoding="utf-8"
    )

    paths = configuration_contracts.PLCSPathConfig.from_config(_path_root())

    assert paths.resolver is not None
    assert "PLCSGenerationComponents" not in source
    assert "CameraProfileConfig" not in source


def test_canonical_pipeline_owns_camera_and_motion_configuration() -> None:
    source = (
        PROJECT_ROOT / "src/synthetic_data_generation/configuration.py"
    ).read_text(encoding="utf-8")

    assert "CameraProfileConfig" in source
    assert "PLCSDatasetConfiguration" in source

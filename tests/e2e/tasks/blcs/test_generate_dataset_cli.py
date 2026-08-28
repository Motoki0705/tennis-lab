"""Asset-free Hydra CLI composition tests for BLCS dataset generation."""

from __future__ import annotations

import subprocess
import sys

import pytest
import yaml

from src.utils.paths import PROJECT_ROOT


@pytest.mark.parametrize("camera", ["default", "broadcast"])
@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("physical_v1", "physical_v1"),
        ("camera_view_v2", "camera_view_v2"),
    ],
)
def test_generate_dataset_cli_composes_independent_court_contracts(
    camera: str,
    selector: str,
    expected: str,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tasks.blcs.scripts.generate_dataset",
            f"court_keypoints={selector}",
            f"camera={camera}",
            "--cfg",
            "job",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = yaml.safe_load(completed.stdout)
    assert payload["court_keypoints"] == {"selector": expected}
    assert "court_coordinate_normalization" not in payload


def test_generate_dataset_cli_keeps_physical_v1_default() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tasks.blcs.scripts.generate_dataset",
            "--cfg",
            "job",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = yaml.safe_load(completed.stdout)
    assert payload["court_keypoints"] == {"selector": "physical_v1"}

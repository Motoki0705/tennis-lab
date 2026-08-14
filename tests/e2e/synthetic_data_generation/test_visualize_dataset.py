"""CLI composition test for canonical generated-dataset visualization."""

from __future__ import annotations

import subprocess
import sys

import yaml

from src.utils.paths import PROJECT_ROOT


def test_cli_composes_strict_production_visualization_config() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.synthetic_data_generation.scripts.visualize_dataset",
            "--cfg",
            "job",
            "--resolve",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = yaml.safe_load(completed.stdout)
    assert payload["visualization"] == {
        "domain": "court",
        "dataset_root": "???",
        "output_video": "synthetic_data_generation/dataset-visualization.mp4",
        "trajectory_id": None,
        "logical_scene_id": None,
        "camera_id": None,
        "fps": 30.0,
        "crf": 17,
        "history_frames": 12,
    }

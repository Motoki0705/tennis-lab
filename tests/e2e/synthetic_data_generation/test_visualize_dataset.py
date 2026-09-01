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
        "court_overlay": {
            "mode": "semantic",
            "color_rgb": [255, 96, 32],
            "background_color_rgb": [0, 0, 0],
            "opacity": 0.55,
            "depth_epsilon_m": 0.02,
            "near_plane_m": 0.05,
            "maximum_cells": 1_000_000,
            "maximum_surface_faces": 4_000_000,
            "maximum_projected_pixels": 100_000_000,
        },
    }


def test_cli_hydra_override_selects_strict_court_v4_aabb_mode() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.synthetic_data_generation.scripts.visualize_dataset",
            "--cfg",
            "job",
            "--resolve",
            "visualization.court_overlay.mode=trajectory_support_aabb",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = yaml.safe_load(completed.stdout)

    assert (
        payload["visualization"]["court_overlay"]["mode"]
        == "trajectory_support_aabb"
    )

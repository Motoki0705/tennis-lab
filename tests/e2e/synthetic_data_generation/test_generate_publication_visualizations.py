"""CLI composition coverage for publication visualization generation."""

from __future__ import annotations

import subprocess
import sys

import yaml

from src.synthetic_data_generation.visualization.publication.contracts import (
    REQUIRED_PUBLICATION_ARTIFACTS,
)
from src.utils.paths import PROJECT_ROOT


def test_cli_composes_one_complete_explicit_publication_request() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.synthetic_data_generation.scripts.generate_publication_visualizations",
            "publication.scene_id=scene-0",
            "publication.court.trajectory_id=trajectory-0",
            "publication.court.frame_indices=[0,2]",
            "publication.blcs.logical_scene_id=logical-0",
            "publication.blcs.camera_id=camera-0",
            "publication.blcs.frame_indices=[0,2]",
            "publication.blcs.camera_ids=[camera-0,camera-1]",
            "publication.plcs.logical_scene_id=logical-0",
            "publication.plcs.camera_id=camera-0",
            "publication.plcs.frame_indices=[0,2]",
            "publication.plcs.camera_ids=[camera-0,camera-1]",
            "publication.captured.camera_ids=[camera-0,camera-1]",
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
    assert set(payload) == {"roots", "publication"}
    publication = payload["publication"]
    assert publication["scene_id"] == "scene-0"
    assert publication["scene_root"] == "scenes/scene-0"
    assert publication["output_bundle"] == (
        "synthetic_data_generation/publication/scene-0"
    )
    assert publication["artifacts"] == [
        artifact.value for artifact in REQUIRED_PUBLICATION_ARTIFACTS
    ]
    assert publication["court"] == {
        "trajectory_id": "trajectory-0",
        "frame_indices": [0, 2],
    }
    assert publication["blcs"] == {
        "logical_scene_id": "logical-0",
        "camera_id": "camera-0",
        "frame_indices": [0, 2],
        "camera_ids": ["camera-0", "camera-1"],
    }
    assert publication["plcs"] == {
        "logical_scene_id": "logical-0",
        "camera_id": "camera-0",
        "frame_indices": [0, 2],
        "camera_ids": ["camera-0", "camera-1"],
    }
    assert publication["captured"] == {"camera_ids": ["camera-0", "camera-1"]}
    assert publication["drawing"] == {
        "dataset_size": [960, 540],
        "alignment_size": [1200, 900],
        "figure_size": [1800, 1350],
        "overview_size": [2700, 1800],
        "gif_duration_ms": 160,
        "frustum_depth_metres": 3.0,
        "line_width": 1.4,
        "font_size": 12,
        "history_frames": 12,
        "maximum_rendered_captured_cameras": 24,
        "coincident_centre_tolerance_metres": 1.0e-6,
        "coincident_forward_angle_tolerance_degrees": 1.0e-6,
        "maximum_artifact_bytes": 12_582_912,
        "maximum_bundle_bytes": 75_497_472,
    }

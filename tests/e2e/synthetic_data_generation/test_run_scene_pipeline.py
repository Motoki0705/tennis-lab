"""Public CLI composition tests for the sole scene-pipeline entrypoint."""

from __future__ import annotations

import subprocess
import sys

import yaml

from src.utils.paths import PROJECT_ROOT


def test_cli_resolves_one_full_b00_request_without_legacy_pipeline_fields() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.synthetic_data_generation.scripts.run_scene_pipeline",
            "--cfg",
            "job",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = yaml.safe_load(completed.stdout)
    assert payload["request"] == {
        "scene_id": "B00",
        "source_video": "neural-harmonic-textures/data/tennis_court.mp4",
        "targets": ["court", "blcs", "plcs"],
        "from_stage": "ingest",
    }
    assert payload["pipeline"]["config_schema"] == "canonical_scene_pipeline_v1"
    assert payload["camera"]["expected_camera_count"] == 6
    assert payload["dataset"]["court"]["sampling"]["proposal_budget"] == 4800
    assert payload["dataset"]["blcs"]["timeline"]["frame_selection"] == (
        "all_source_frames"
    )
    assert payload["dataset"]["plcs"]["multi_object_global_timeline"] is True
    serialized = completed.stdout
    for forbidden in (
        "artifact_ref",
        "fingerprint",
        "content_addressed",
        "immutable_publication",
        "selected_camera",
        "pose_indices",
    ):
        assert forbidden not in serialized.lower()

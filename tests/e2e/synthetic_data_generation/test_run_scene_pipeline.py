"""Public CLI composition tests for the sole scene-pipeline entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
        "source_video": "synthetic_data_generation/raw/tennis_court.mp4",
        "targets": ["court", "blcs", "plcs"],
        "from_stage": "ingest",
    }
    assert payload["pipeline"]["config_schema"] == "canonical_scene_pipeline_v1"
    assert payload["camera"]["expected_camera_count"] == 6
    assert payload["dataset"]["court"]["sampling"]["proposal_budget"] == 4800
    assert payload["dataset"]["blcs"]["timeline"]["frame_selection"] == (
        "all_source_frames"
    )
    assert payload["dataset"]["plcs"]["production_mode"] == (
        "multi_object_global_timeline"
    )
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


def test_cli_exposes_exact_v1_v2_v3_court_selectors_and_keeps_v1_default() -> None:
    payloads = {}
    for selector in (None, "v1", "v2", "v3"):
        argv = [
            sys.executable,
            "-m",
            "src.synthetic_data_generation.scripts.run_scene_pipeline",
        ]
        if selector is not None:
            argv.append(f"dataset/court={selector}")
        argv.extend(("--cfg", "job"))
        completed = subprocess.run(
            argv,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payloads[selector] = yaml.safe_load(completed.stdout)["dataset"]["court"]

    assert payloads[None] == payloads["v1"]
    assert payloads[None]["schema_version"] == "v1"
    assert payloads[None]["view"]["target_modes"] == [
        "court_center",
        "complex_center",
        "near_baseline",
        "far_baseline",
    ]
    assert payloads["v2"]["schema_version"] == "v2"
    assert payloads["v2"]["view"]["target_modes"] == ["court_center"]
    assert payloads["v3"]["schema_version"] == "v3"
    assert payloads["v3"]["view"]["target_modes"] == ["court_center"]


def test_court_readme_is_single_linked_and_documents_executable_selectors() -> None:
    parent = Path(PROJECT_ROOT / "src/synthetic_data_generation/README.md").read_text(
        encoding="utf-8"
    )
    contract = Path(
        PROJECT_ROOT / "src/synthetic_data_generation/dataset/court/README.md"
    ).read_text(encoding="utf-8")

    link = "[Court Detection dataset v1/v2/v3 contract](dataset/court/README.md)"
    assert parent.count(link) == 1
    assert parent.count("## Court dataset v1/v2/v3 contract") == 0
    assert contract.count("dataset/court=v1") >= 1
    assert contract.count("dataset/court=v2") >= 1
    assert contract.count("dataset/court=v3") >= 1
    assert "canonical_court_dataset_v1" in contract
    assert "canonical_court_dataset_v2" in contract
    assert "canonical_court_dataset_v3" in contract

"""Tests for immutable metric court-geometry artifacts."""

from pathlib import Path

import pytest

from src.synthetic_data_generation.alignment.artifacts.court_geometry import (
    COURT_GEOMETRY_SCHEMA,
    load_court_geometry_artifact,
    publish_court_geometry_artifact,
)


def _payload() -> dict[str, object]:
    return {
        "schema": COURT_GEOMETRY_SCHEMA,
        "artifact_id": "synthetic-court-fit-v1",
        "created_at_utc": "2026-07-25T00:00:00+00:00",
        "ground_line_artifact": {},
        "fit_settings": {},
        "candidates": [
            {
                "candidate_id": "court-0",
                "template_score": 1.0,
            }
        ],
        "selection": {
            "selected_candidate_id": "court-0",
            "rule": "highest score",
        },
        "acceptance_status": "fit_candidate_holdout_not_run",
        "provenance": {},
    }


def test_geometry_artifact_round_trip_and_holdout_status(tmp_path: Path) -> None:
    payload = _payload()

    path = publish_court_geometry_artifact(payload, output_dir=tmp_path)

    loaded = load_court_geometry_artifact(path)
    assert loaded["selection"]["selected_candidate_id"] == "court-0"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_court_geometry_artifact(payload, output_dir=tmp_path)


def test_geometry_artifact_rejects_missing_selected_candidate(
    tmp_path: Path,
) -> None:
    payload = _payload()
    selection = payload["selection"]
    assert isinstance(selection, dict)
    selection["selected_candidate_id"] = "court-missing"

    with pytest.raises(ValueError, match="does not exist"):
        publish_court_geometry_artifact(payload, output_dir=tmp_path)

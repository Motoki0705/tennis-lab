"""Tests for typed alignment stage and resume boundaries."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from src.synthetic_data_generation.alignment.stage_result import (
    AlignmentJob,
    AlignmentStageError,
    ArtifactHandle,
    StageResult,
    find_matching_artifact,
)


def _payload(created_at: str, fingerprint: str) -> dict[str, Any]:
    return {
        "schema": "test_v1",
        "artifact_id": "artifact-v1",
        "artifact_fingerprint": fingerprint,
        "created_at_utc": created_at,
    }


def test_alignment_job_is_immutable(tmp_path: Path) -> None:
    job = AlignmentJob(
        alignment_id="alignment-v1",
        scene_id="scene-v1",
        provider_bundle=tmp_path / "provider",
        artifact_root=tmp_path / "artifacts",
        config_overrides={},
    )

    with pytest.raises(FrozenInstanceError):
        job.scene_id = "changed"  # type: ignore[misc]


def test_stage_result_serializes_paths(tmp_path: Path) -> None:
    result = StageResult(
        stage="ground_line",
        status="executed",
        artifact_paths=(tmp_path / "artifact",),
        primary_artifact=tmp_path / "artifact",
        fingerprint="a" * 64,
        metadata={"count": 1},
    )

    serialized = result.to_dict()

    assert serialized["primary_artifact"] == str(tmp_path / "artifact")
    assert serialized["artifact_paths"] == [str(tmp_path / "artifact")]


def test_stage_error_preserves_published_artifacts(tmp_path: Path) -> None:
    paths = (tmp_path / "calibration.json",)

    error = AlignmentStageError(
        "fit gates failed",
        stage="calibration",
        job_id="alignment-v1",
        preserved_artifacts=paths,
    )

    assert error.stage == "calibration"
    assert error.job_id == "alignment-v1"
    assert error.preserved_artifacts == paths


def test_find_matching_artifact_strict_loads_and_selects_newest(
    tmp_path: Path,
) -> None:
    older = tmp_path / "older.json"
    newer = tmp_path / "newer.json"
    older.write_text("older", encoding="utf-8")
    newer.write_text("newer", encoding="utf-8")
    payloads = {
        older: _payload("2026-07-25T00:00:00+00:00", "a" * 64),
        newer: _payload("2026-07-26T00:00:00+00:00", "b" * 64),
    }
    loaded: list[Path] = []

    def load(path: Path) -> dict[str, Any]:
        loaded.append(path)
        return payloads[path]

    match = find_matching_artifact(
        (newer, older),
        load=load,
        matches=lambda payload: payload["schema"] == "test_v1",
    )

    assert loaded == [newer, older]
    assert match is not None
    assert match[0] == newer


def test_artifact_handle_serializes_strict_identity(tmp_path: Path) -> None:
    handle = ArtifactHandle(
        path=tmp_path / "artifact.json",
        artifact_id="artifact-v1",
        fingerprint="a" * 64,
        file_sha256="b" * 64,
        schema="test_v1",
    )

    assert handle.to_dict() == {
        "path": str(tmp_path / "artifact.json"),
        "artifact_id": "artifact-v1",
        "fingerprint": "a" * 64,
        "file_sha256": "b" * 64,
        "schema": "test_v1",
    }

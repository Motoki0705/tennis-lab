"""Tests for the shared path-only pipeline manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.synthetic_data_generation.dataset.pipeline import (
    PATH_PIPELINE_SCHEMA,
    PathPipelineManifest,
)


def _configured_paths(root: Path) -> dict[str, object]:
    return {
        "source_root": "third_party/nht/data",
        "artifact_root": "third_party/nht/artifacts/synthetic-data",
        "execution_root": "outputs/synthetic_data_generation",
        "dataset_root": "data/synthetic_data_generation",
        "alignment_observations": ("third_party/nht/data/alignment-observations.json"),
        "render_jobs": "third_party/nht/data/render-jobs.json",
        "pipeline_manifest": ("outputs/synthetic_data_generation/path-manifest.json"),
        "alignment_metrics": (
            "third_party/nht/artifacts/synthetic-data/alignment-metrics.json"
        ),
        "dataset_plan": ("third_party/nht/artifacts/synthetic-data/dataset-plan.json"),
        "render_manifest": (
            "third_party/nht/artifacts/synthetic-data/render-manifest.json"
        ),
        "quality_metrics": (
            "third_party/nht/artifacts/synthetic-data/quality-metrics.json"
        ),
        "visualization": ("outputs/synthetic_data_generation/pipeline-summary.html"),
    }


def test_path_manifest_round_trip_uses_configured_layout(tmp_path: Path) -> None:
    manifest = PathPipelineManifest.from_config(
        _configured_paths(tmp_path),
        project_root=tmp_path,
    )

    manifest.write()
    loaded = PathPipelineManifest.read(manifest.pipeline_manifest)

    assert loaded == manifest
    assert loaded.to_dict()["schema"] == PATH_PIPELINE_SCHEMA
    assert loaded.source_root == (tmp_path / "third_party/nht/data").resolve()
    assert (
        loaded.artifact_root
        == (tmp_path / "third_party/nht/artifacts/synthetic-data").resolve()
    )
    assert (
        loaded.execution_root
        == (tmp_path / "outputs/synthetic_data_generation").resolve()
    )
    assert (
        loaded.dataset_root == (tmp_path / "data/synthetic_data_generation").resolve()
    )


def test_path_manifest_rejects_malformed_json_and_unrelated_fields(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        PathPipelineManifest.read(malformed)

    payload = PathPipelineManifest.from_config(
        _configured_paths(tmp_path),
        project_root=tmp_path,
    ).to_dict()
    payload["approval"] = True
    with pytest.raises(ValueError, match="only schema and paths"):
        PathPipelineManifest.from_dict(payload)


def test_path_manifest_requires_every_named_path(tmp_path: Path) -> None:
    paths = _configured_paths(tmp_path)
    del paths["render_jobs"]

    with pytest.raises(ValueError, match="missing=\\['render_jobs'\\]"):
        PathPipelineManifest.from_config(paths, project_root=tmp_path)

    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        PathPipelineManifest.read(missing)

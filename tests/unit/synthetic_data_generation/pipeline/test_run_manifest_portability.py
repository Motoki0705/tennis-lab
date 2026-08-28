"""Portable source-authority checks for canonical scene manifests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.synthetic_data_generation.pipeline.contracts import (
    DatasetTarget,
    ScenePipelineRequest,
)
from src.synthetic_data_generation.pipeline.run_manifest import MutableRunManifest

pytestmark = pytest.mark.unit


def _request(source: Path) -> ScenePipelineRequest:
    return ScenePipelineRequest(
        scene_id="scene-a",
        source_video=source.resolve(),
        targets=frozenset({DatasetTarget.COURT}),
        from_stage="court_dataset",
        config_schema="canonical_scene_pipeline_v1",
    )


def test_relocated_source_is_accepted_when_ingested_copy_is_identical(
    tmp_path: Path,
) -> None:
    requested = tmp_path / "mounted" / "source.mp4"
    requested.parent.mkdir()
    requested.write_bytes(b"same-video-bytes")
    canonical = tmp_path / "scene" / "source" / "video.mp4"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(requested.read_bytes())
    request = _request(requested)
    manifest = MutableRunManifest.create(request)
    manifest.source_video = "/old-host/project/data/source.mp4"

    manifest.assert_request_compatible(
        request,
        canonical_source_video=canonical,
    )


def test_relocated_source_is_rejected_when_ingested_copy_differs(
    tmp_path: Path,
) -> None:
    requested = tmp_path / "mounted" / "source.mp4"
    requested.parent.mkdir()
    requested.write_bytes(b"requested-video")
    canonical = tmp_path / "scene" / "source" / "video.mp4"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"different-video")
    request = _request(requested)
    manifest = MutableRunManifest.create(request)
    manifest.source_video = "/old-host/project/data/source.mp4"

    with pytest.raises(ValueError, match="source video disagrees"):
        manifest.assert_request_compatible(
            request,
            canonical_source_video=canonical,
        )


def test_matching_manifest_path_does_not_require_ingested_copy(
    tmp_path: Path,
) -> None:
    requested = tmp_path / "source.mp4"
    requested.write_bytes(b"video")
    request = _request(requested)
    manifest = MutableRunManifest.create(request)

    manifest.assert_request_compatible(
        request,
        canonical_source_video=tmp_path / "missing.mp4",
    )

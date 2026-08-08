"""Tests for mutable current-state run manifests."""

import pytest

from src.synthetic_data_generation.pipeline import DatasetTarget, StageName
from src.synthetic_data_generation.pipeline.contracts import (
    ScenePipelineRequest,
    StageStatus,
)
from src.synthetic_data_generation.pipeline.run_manifest import MutableRunManifest


def _request(tmp_path, *, schema: str = "scene_pipeline_v1") -> ScenePipelineRequest:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    return ScenePipelineRequest(
        scene_id="scene-a",
        source_video=source,
        targets=frozenset({DatasetTarget.COURT}),
        from_stage=StageName.INGEST,
        config_schema=schema,
    )


def test_manifest_round_trip_and_transitions(tmp_path) -> None:
    request = _request(tmp_path)
    manifest = MutableRunManifest.create(request)
    path = tmp_path / "run.json"

    manifest.begin(StageName.INGEST)
    manifest.complete(StageName.INGEST, {"frames": 10})
    manifest.invalidate(StageName.COURT_DATASET)
    manifest.skip(StageName.BLCS_DATASET)
    manifest.save(path)

    loaded = MutableRunManifest.load(path)
    assert loaded.stages[StageName.INGEST].status is StageStatus.COMPLETED
    assert loaded.stages[StageName.INGEST].summary == {"frames": 10}
    assert loaded.stages[StageName.COURT_DATASET].status is StageStatus.INVALIDATED
    assert loaded.stages[StageName.BLCS_DATASET].status is StageStatus.SKIPPED


def test_manifest_rejects_incompatible_request_before_mutation(tmp_path) -> None:
    request = _request(tmp_path)
    manifest = MutableRunManifest.create(request)
    incompatible = ScenePipelineRequest(
        scene_id=request.scene_id,
        source_video=request.source_video,
        targets=request.targets,
        from_stage=request.from_stage,
        config_schema="scene_pipeline_v2",
    )

    with pytest.raises(ValueError, match="config schema"):
        manifest.assert_request_compatible(incompatible)

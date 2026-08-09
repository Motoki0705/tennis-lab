"""Tests for mutable current-state run manifests."""

from pathlib import Path

import pytest

from src.synthetic_data_generation.pipeline import DatasetTarget, StageName
from src.synthetic_data_generation.pipeline.contracts import (
    ScenePipelineRequest,
    StageStatus,
)
from src.synthetic_data_generation.pipeline.run_manifest import MutableRunManifest


def _request(tmp_path: Path, *, schema: str = "scene_pipeline_v1") -> ScenePipelineRequest:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    return ScenePipelineRequest(
        scene_id="scene-a",
        source_video=source,
        targets=frozenset({DatasetTarget.COURT}),
        from_stage=StageName.INGEST,
        config_schema=schema,
    )


def test_manifest_round_trip_and_transitions(tmp_path: Path) -> None:
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


def test_manifest_rejects_incompatible_request_before_mutation(tmp_path: Path) -> None:
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


@pytest.mark.parametrize(
    "status",
    [
        StageStatus.PENDING,
        StageStatus.FAILED,
        StageStatus.INVALIDATED,
        StageStatus.SKIPPED,
    ],
)
def test_manifest_begin_accepts_only_retryable_states(
    tmp_path: Path,
    status: StageStatus,
) -> None:
    manifest = MutableRunManifest.create(_request(tmp_path))
    record = manifest.stages[StageName.INGEST]
    record.status = status
    record.attempt = 3
    record.summary = {"stale": True}
    record.error = "stale error"

    manifest.begin(StageName.INGEST)

    assert record.status is StageStatus.RUNNING
    assert record.attempt == 4
    assert record.summary == {}
    assert record.error is None


@pytest.mark.parametrize("status", [StageStatus.RUNNING, StageStatus.COMPLETED])
def test_manifest_begin_rejects_non_retryable_states_without_mutation(
    tmp_path: Path,
    status: StageStatus,
) -> None:
    manifest = MutableRunManifest.create(_request(tmp_path))
    record = manifest.stages[StageName.INGEST]
    record.status = status
    record.attempt = 3
    record.summary = {"preserved": True}
    before = record.to_dict()

    with pytest.raises(ValueError, match=f"cannot begin from {status.value}"):
        manifest.begin(StageName.INGEST)

    assert record.to_dict() == before


def test_completed_stage_can_begin_only_after_invalidation(tmp_path: Path) -> None:
    manifest = MutableRunManifest.create(_request(tmp_path))
    manifest.begin(StageName.INGEST)
    manifest.complete(StageName.INGEST, {"frames": 10})

    with pytest.raises(ValueError, match="completed stages require explicit invalidation"):
        manifest.begin(StageName.INGEST)

    manifest.invalidate(StageName.INGEST)
    manifest.begin(StageName.INGEST)
    assert manifest.stages[StageName.INGEST].status is StageStatus.RUNNING
    assert manifest.stages[StageName.INGEST].attempt == 2

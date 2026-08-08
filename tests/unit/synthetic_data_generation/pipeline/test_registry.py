"""Tests for the sole canonical stage graph."""

from pathlib import Path

from src.synthetic_data_generation.pipeline import DatasetTarget, StageName
from src.synthetic_data_generation.pipeline.contracts import ScenePipelineRequest
from src.synthetic_data_generation.pipeline.registry import canonical_registry


def test_registry_derives_all_alignment_descendants(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    request = ScenePipelineRequest(
        scene_id="scene-a",
        source_video=source,
        targets=frozenset({DatasetTarget.COURT}),
        from_stage=StageName.INGEST,
        config_schema="scene_pipeline_v1",
    )
    registry = canonical_registry()

    assert registry.selected_for_request(request) == (
        StageName.INGEST,
        StageName.RECONSTRUCTION,
        StageName.ALIGNMENT,
        StageName.COURT_DATASET,
        StageName.REPORT,
    )
    assert set(registry.descendants(StageName.ALIGNMENT)) == {
        StageName.COURT_DATASET,
        StageName.BLCS_DATASET,
        StageName.PLCS_DATASET,
        StageName.REPORT,
    }
    assert registry.spec(StageName.PLCS_DATASET).required_outputs == (
        Path("dataset.json"),
        Path("backgrounds"),
        Path("scenes"),
        Path("diagnostics"),
    )

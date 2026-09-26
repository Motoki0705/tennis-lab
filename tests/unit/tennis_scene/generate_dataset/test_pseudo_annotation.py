"""Publication of completion markers over the clip store's immutable scene export."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from src.tennis_scene.generate_dataset.pseudo_annotation import (
    SceneRunner,
    generate_pseudo_annotations,
)
from src.tennis_scene.schema import SceneResult
from src.utils.io import load_json
from tests.support.tennis_scene.annotations import publish_scene_to_clip_store

CLIP_ID = "video_000/clip_000"
IDENTITY = {"checkpoint": "sha-one"}


def _clip(dataset: Path) -> Path:
    return dataset / "videos/video_000/clips/clip_000"


def _publishing_runner(scene: SceneResult, calls: list[int] | None = None) -> SceneRunner:
    """Stand in for the orchestrator: leave the export in the clip store, return the scene."""
    def runner(video_paths: Sequence[Path], camera_ids: Sequence[str], clip_dir: Path) -> SceneResult:
        assert [path.name for path in video_paths] == ["cam0.mp4"] and list(camera_ids) == ["cam0"]
        if calls is not None:
            calls.append(1)
        publish_scene_to_clip_store(clip_dir, CLIP_ID, scene)
        return scene
    return runner


def test_marker_points_at_the_export_and_is_reused_for_the_same_identity(
    structured_dataset: Path, valid_scene_result: SceneResult
) -> None:
    calls: list[int] = []
    runner = _publishing_runner(valid_scene_result, calls)
    first = generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml="device: cpu\n", publication_identity=IDENTITY)
    assert first[0].status == "generated" and first[0].annotation_path is not None
    marker = load_json(first[0].annotation_path)
    assert marker["clip_id"] == CLIP_ID and marker["scene_index"] == "scene.json"
    assert marker["scene_result"].startswith("exports/") and marker["arrays"]["ball_3d"]["shape"] == [3, 3]
    second = generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml="device: cpu\n", publication_identity=IDENTITY)
    assert second[0].status == "skipped" and len(calls) == 1


def test_config_is_content_addressed_and_the_export_stays_immutable(
    structured_dataset: Path, valid_scene_result: SceneResult
) -> None:
    outcome = generate_pseudo_annotations(structured_dataset, _publishing_runner(valid_scene_result),
        pipeline_config_yaml="device: cpu\n", publication_identity=IDENTITY)[0]
    assert outcome.annotation_path is not None
    marker = load_json(outcome.annotation_path)
    annotations = outcome.annotation_path.parent
    export = (annotations / marker["scene_result"]).parent
    assert sorted(path.name for path in export.iterdir()) == ["scene.metadata.json", "scene.npz"]
    config = annotations / marker["pipeline_config"]
    assert config.parent.name == "configs" and config.read_text() == "device: cpu\n"


def test_completed_annotation_cannot_hide_changed_weights(
    structured_dataset: Path, valid_scene_result: SceneResult,
) -> None:
    calls: list[int] = []
    runner = _publishing_runner(valid_scene_result, calls)
    assert generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml="test: true", publication_identity=IDENTITY)[0].status == "generated"
    changed = generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml="test: true", publication_identity={"checkpoint": "sha-two"})
    assert changed[0].status == "failed" and "Stale annotation" in str(changed[0].error)
    assert len(calls) == 1


def test_historical_marker_without_identity_is_never_silently_reused(
    structured_dataset: Path, valid_scene_result: SceneResult
) -> None:
    from src.utils.io import save_json_atomic
    save_json_atomic({"version": 1, "scene_result": "scene.npz"}, _clip(structured_dataset) / "annotations/tennis_scene/annotation.json")
    outcome = generate_pseudo_annotations(structured_dataset, _publishing_runner(valid_scene_result),
        pipeline_config_yaml="x: 1", publication_identity=IDENTITY)[0]
    assert outcome.status == "failed" and "use overwrite=true" in str(outcome.error)


@pytest.mark.parametrize("change,message", [
    (lambda s: setattr(s, "fps", 25.0), "fps 25.0 != 30.0"),
    (lambda s: s.metadata.update(pipeline_contract="legacy_stages"), "declared_components_v1"),
    (lambda s: setattr(s, "ball_3d", None), "Invalid v2 ball_3d"),
])
def test_contract_violation_records_failure_without_completion_marker(
    structured_dataset: Path, valid_scene_result: SceneResult, change, message: str
) -> None:
    change(valid_scene_result)
    outcomes = generate_pseudo_annotations(structured_dataset, lambda *_: valid_scene_result,
        pipeline_config_yaml="device: cpu\n", publication_identity=IDENTITY)
    assert outcomes[0].status == "failed" and message in str(outcomes[0].error)
    annotations = _clip(structured_dataset) / "annotations"
    assert not (annotations / "tennis_scene/annotation.json").exists()
    assert message in load_json(annotations / "tennis_scene.failure.json")["error"]


def test_integrity_failure_propagates_without_completion_marker(structured_dataset: Path) -> None:
    from src.utils.checksum import FileIntegrityError

    error = FileIntegrityError("providers disagree", details={"path": "checkpoint"})

    def runner(_videos: Sequence[Path], _cameras: Sequence[str], _clip: Path) -> SceneResult:
        raise error

    with pytest.raises(FileIntegrityError) as caught:
        generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml="device: cpu\n",
            publication_identity=IDENTITY, continue_on_error=False)
    assert caught.value is error
    annotations = _clip(structured_dataset) / "annotations"
    assert not (annotations / "tennis_scene/annotation.json").exists()
    assert "FileIntegrityError" in load_json(annotations / "tennis_scene.failure.json")["error"]


def test_publication_preserves_the_store_and_slcs_follows_the_index(
    structured_dataset: Path, valid_scene_result: SceneResult,
) -> None:
    from src.tasks.slcs.data.annotation import load_slcs_annotation
    from src.tennis_scene.generate_dataset.manifest import ClipManifest
    clip = _clip(structured_dataset)
    export = publish_scene_to_clip_store(clip, CLIP_ID, valid_scene_result)
    index_before = (clip / "annotations/tennis_scene/scene.json").read_bytes()
    outcomes = generate_pseudo_annotations(structured_dataset, lambda *_: valid_scene_result,
        pipeline_config_yaml="device: cpu", publication_identity=IDENTITY)
    assert outcomes[0].status == "generated" and outcomes[0].annotation_path is not None
    assert (clip / "annotations/tennis_scene/scene.json").read_bytes() == index_before
    marker = load_json(outcomes[0].annotation_path)
    assert marker["scene_result"] == str(export.relative_to(clip / "annotations/tennis_scene"))
    loaded = load_slcs_annotation(ClipManifest.load(clip))
    assert loaded.player_position.shape == valid_scene_result.player_position.shape

"""Tests for incremental pseudo annotation publication."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.generate_dataset.pseudo_annotation import (
    generate_pseudo_annotations,
)
from src.tennis_scene.schema import SceneResult
from src.utils.io import load_json


def test_generate_publishes_complete_annotation_and_then_skips(
    structured_dataset: Path, valid_scene_result: SceneResult
) -> None:
    calls = 0

    def runner(video_paths: Sequence[Path], camera_ids: Sequence[str]) -> SceneResult:
        nonlocal calls
        calls += 1
        assert [path.name for path in video_paths] == ["cam0.mp4"]
        assert list(camera_ids) == ["cam0"]
        return valid_scene_result

    first = generate_pseudo_annotations(
        structured_dataset,
        runner,
        pipeline_config_yaml="device: cpu\n",
    )
    assert first[0].status == "generated"
    assert calls == 1
    assert first[0].annotation_path is not None
    annotation = load_json(first[0].annotation_path)
    assert annotation["clip_id"] == "video_000/clip_000"
    assert annotation["arrays"]["ball_3d"]["shape"] == [3, 3]
    assert annotation["arrays"]["gvhmr_aligned_player_position"]["shape"] == [
        2,
        3,
        3,
    ]
    loaded = load_scene_result(first[0].annotation_path.parent / "scene.npz")
    assert loaded.metadata["dataset_clip_id"] == "video_000/clip_000"

    second = generate_pseudo_annotations(
        structured_dataset,
        runner,
        pipeline_config_yaml="device: cpu\n",
    )
    assert second[0].status == "skipped"
    assert calls == 1


def test_completed_annotation_cannot_hide_changed_weights(
    structured_dataset: Path, valid_scene_result: SceneResult,
) -> None:
    calls = []
    def runner(_paths: Sequence[Path], _ids: Sequence[str]) -> SceneResult:
        calls.append(1)
        return valid_scene_result
    first = generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml="test: true", publication_identity={"checkpoint": "sha-one"})
    assert first[0].status == "generated"
    same = generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml="test: true", publication_identity={"checkpoint": "sha-one"})
    assert same[0].status == "skipped"
    changed = generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml="test: true", publication_identity={"checkpoint": "sha-two"})
    assert changed[0].status == "failed"
    assert "Stale annotation" in str(changed[0].error)
    assert len(calls) == 1


def test_contract_mismatch_records_failure_without_completion_marker(
    structured_dataset: Path, valid_scene_result: SceneResult
) -> None:
    valid_scene_result.num_frames = 2

    def runner(_video_paths: Sequence[Path], _camera_ids: Sequence[str]) -> SceneResult:
        return valid_scene_result

    outcomes = generate_pseudo_annotations(
        structured_dataset,
        runner,
        pipeline_config_yaml="device: cpu\n",
    )
    assert outcomes[0].status == "failed"
    assert "num_frames 2 != 3" in str(outcomes[0].error)
    annotation_root = (
        structured_dataset
        / "videos"
        / "video_000"
        / "clips"
        / "clip_000"
        / "annotations"
    )
    assert not (annotation_root / "tennis_scene" / "annotation.json").exists()
    failure = load_json(annotation_root / "tennis_scene.failure.json")
    assert "num_frames 2 != 3" in failure["error"]


def test_missing_blcs_labels_is_explicit_failure(
    structured_dataset: Path, valid_scene_result: SceneResult
) -> None:
    valid_scene_result.ball_3d = None

    def runner(_video_paths: Sequence[Path], _camera_ids: Sequence[str]) -> SceneResult:
        return valid_scene_result

    outcomes = generate_pseudo_annotations(
        structured_dataset,
        runner,
        pipeline_config_yaml="device: cpu\n",
    )
    assert outcomes[0].status == "failed"
    assert "required pseudo-label array 'ball_3d' is missing" in str(outcomes[0].error)


def test_plcs_only_result_does_not_require_disabled_ball_stages(
    structured_dataset: Path, valid_scene_result: SceneResult
) -> None:
    valid_scene_result.ball_uv = None
    valid_scene_result.ball_vis = None
    valid_scene_result.ball_3d = None
    valid_scene_result.metadata["enabled_stages"] = ["court_kp", "gvhmr", "plcs"]

    def runner(_video_paths: Sequence[Path], _camera_ids: Sequence[str]) -> SceneResult:
        return valid_scene_result

    outcomes = generate_pseudo_annotations(
        structured_dataset,
        runner,
        pipeline_config_yaml="device: cpu\n",
    )

    assert outcomes[0].status == "generated"
    assert outcomes[0].annotation_path is not None
    annotation = load_json(outcomes[0].annotation_path)
    assert "ball_uv" not in annotation["arrays"]
    assert "ball_vis" not in annotation["arrays"]
    assert "ball_3d" not in annotation["arrays"]


def test_integrity_failure_propagates_without_completion_marker(
    structured_dataset: Path,
) -> None:
    import pytest

    from src.utils.checksum import FileIntegrityError

    error = FileIntegrityError("providers disagree", details={"path": "checkpoint"})

    def runner(_videos: Sequence[Path], _cameras: Sequence[str]) -> SceneResult:
        raise error

    with pytest.raises(FileIntegrityError) as caught:
        generate_pseudo_annotations(
            structured_dataset,
            runner,
            pipeline_config_yaml="device: cpu\n",
            continue_on_error=False,
        )
    assert caught.value is error
    annotations = structured_dataset / "videos/video_000/clips/clip_000/annotations"
    assert not (annotations / "tennis_scene/annotation.json").exists()
    failure = load_json(annotations / "tennis_scene.failure.json")
    assert "FileIntegrityError" in failure["error"]

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

    def runner(
        video_paths: Sequence[Path], camera_ids: Sequence[str]
    ) -> SceneResult:
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
    assert annotation["clip_id"] == "match-001/clip_000"
    assert annotation["arrays"]["ball_3d"]["shape"] == [3, 3]
    loaded = load_scene_result(first[0].annotation_path.parent / "scene.npz")
    assert loaded.metadata["dataset_clip_id"] == "match-001/clip_000"

    second = generate_pseudo_annotations(
        structured_dataset,
        runner,
        pipeline_config_yaml="device: cpu\n",
    )
    assert second[0].status == "skipped"
    assert calls == 1


def test_contract_mismatch_records_failure_without_completion_marker(
    structured_dataset: Path, valid_scene_result: SceneResult
) -> None:
    valid_scene_result.num_frames = 2

    def runner(
        _video_paths: Sequence[Path], _camera_ids: Sequence[str]
    ) -> SceneResult:
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
        / "clips"
        / "match-001"
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

    def runner(
        _video_paths: Sequence[Path], _camera_ids: Sequence[str]
    ) -> SceneResult:
        return valid_scene_result

    outcomes = generate_pseudo_annotations(
        structured_dataset,
        runner,
        pipeline_config_yaml="device: cpu\n",
    )
    assert outcomes[0].status == "failed"
    assert "required pseudo-label array 'ball_3d' is missing" in str(outcomes[0].error)

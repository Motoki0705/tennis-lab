"""Unit tests for shared stratified dataset-sample contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.tasks.base.generate_dataset.dataset_samples import (
    DatasetSampleCandidate,
    assign_tercile,
    bounded_playback_fps,
    choose_camera_index,
    evenly_spaced_frame_indices,
    remap_sample_track_instances,
    select_stratified_samples,
    take_temporal_sample,
    tercile_boundaries,
    track_lifecycle_metrics,
    validate_sample_frame_indices,
)


def _candidates() -> tuple[DatasetSampleCandidate, ...]:
    candidates: list[DatasetSampleCandidate] = []
    for primary_index, primary in enumerate(("alpha", "beta", "gamma")):
        for duration_index, duration in enumerate((10.0, 20.0, 30.0)):
            candidates.append(
                DatasetSampleCandidate(
                    scene_id=f"scene_{primary_index}_{duration_index}",
                    primary_group=primary,
                    duration_value=duration,
                    visibility_value=(primary_index * 3 + duration_index + 1) / 10.0,
                    auxiliary_value=float((primary_index + duration_index) % 3),
                    camera_visibilities=(0.1, 0.5, 0.9),
                    metrics={"duration": duration},
                )
            )
    return tuple(candidates)


def test_even_frame_sampling_is_bounded_unique_and_endpoint_inclusive() -> None:
    indices = evenly_spaced_frame_indices(1024, 120)

    assert len(indices) == 120
    assert indices.dtype == np.int64
    assert indices[0] == 0
    assert indices[-1] == 1023
    assert np.all(np.diff(indices) > 0)


def test_even_frame_sampling_keeps_short_timelines_complete() -> None:
    assert evenly_spaced_frame_indices(4, 120).tolist() == [0, 1, 2, 3]


def test_shared_track_lifecycle_metrics_capture_duration_and_concurrency() -> None:
    tracks = (
        {"track_id": 0, "birth_frame": 0, "death_frame": 3},
        {"track_id": 1, "birth_frame": 2, "death_frame": 6},
    )

    assert track_lifecycle_metrics(tracks, num_frames=6, location="meta") == (7, 2)


def test_shared_temporal_helpers_slice_and_remap_one_timeline() -> None:
    indices = np.array([0, 2, 5], dtype=np.int64)
    validate_sample_frame_indices(indices, 6, task="PLCS")
    values = np.arange(12).reshape(6, 2)
    meta = {
        "track_instances": [
            {"track_id": 0, "birth_frame": 0, "death_frame": 3},
            {"track_id": 1, "birth_frame": 2, "death_frame": 6},
        ]
    }
    present = np.array(
        [
            [True, False],
            [True, True],
            [False, True],
        ]
    )

    sampled = take_temporal_sample(
        values,
        indices=indices,
        source_num_frames=6,
        location="scene.values",
    )
    remap_sample_track_instances(meta, present, task="PLCS")

    np.testing.assert_array_equal(sampled, values[indices])
    assert meta["track_instances"] == [
        {"track_id": 0, "birth_frame": 0, "death_frame": 2},
        {"track_id": 1, "birth_frame": 1, "death_frame": 3},
    ]


def test_shared_temporal_helpers_reject_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="endpoint-inclusive"):
        validate_sample_frame_indices(
            np.array([1, 3, 5], dtype=np.int64),
            6,
            task="BLCS",
        )
    with pytest.raises(ValueError, match="must start with T=6"):
        take_temporal_sample(
            np.zeros((5, 2)),
            indices=np.array([0, 5], dtype=np.int64),
            source_num_frames=6,
            location="scene.values",
        )


def test_bounded_playback_fps_preserves_duration_with_explicit_limits() -> None:
    assert (
        bounded_playback_fps(
            source_fps=30.0,
            source_num_frames=301,
            rendered_num_frames=101,
            min_fps=8,
            max_fps=20,
        )
        == 10
    )
    assert (
        bounded_playback_fps(
            source_fps=120.0,
            source_num_frames=121,
            rendered_num_frames=121,
            min_fps=8,
            max_fps=20,
        )
        == 20
    )


def test_stratified_selection_covers_every_three_by_three_cell() -> None:
    selection = select_stratified_samples(
        _candidates(),
        primary_order=("alpha", "beta", "gamma"),
        samples_per_stratum=1,
    )

    assert len(selection.selected) == 9
    assert {sample.stratum_key for sample in selection.selected} == {
        f"{primary}:{duration}"
        for primary in ("alpha", "beta", "gamma")
        for duration in ("short", "medium", "long")
    }
    assert set(selection.stratum_population.values()) == {1}
    assert sorted(
        {sample.visibility_target_quantile for sample in selection.selected}
    ) == pytest.approx([1.0 / 6.0, 0.5, 5.0 / 6.0])
    assert {sample.camera_index for sample in selection.selected} == {0, 1, 2}


def test_stratified_selection_fails_when_a_cell_is_missing() -> None:
    candidates = tuple(
        candidate
        for candidate in _candidates()
        if not (candidate.primary_group == "alpha" and candidate.duration_value == 30.0)
    )

    with pytest.raises(ValueError, match="alpha duration"):
        select_stratified_samples(
            candidates,
            primary_order=("alpha", "beta", "gamma"),
            samples_per_stratum=1,
        )


def test_tercile_helpers_reject_degenerate_metrics() -> None:
    with pytest.raises(ValueError, match="distinct tercile"):
        tercile_boundaries([1.0, 1.0, 1.0], metric_name="constant")

    boundaries = tercile_boundaries([1.0, 2.0, 3.0], metric_name="ordered")
    assert assign_tercile(1.0, boundaries, labels=("low", "mid", "high")) == "low"
    assert assign_tercile(3.0, boundaries, labels=("low", "mid", "high")) == "high"


def test_camera_selection_rejects_invisible_only_scenes() -> None:
    with pytest.raises(ValueError, match="visibility"):
        choose_camera_index((0.0, 0.01), target_quantile=0.5)


def test_camera_selection_uses_within_scene_visibility_rank() -> None:
    visibilities = (0.8, 0.2, 0.5)

    assert choose_camera_index(visibilities, target_quantile=1.0 / 6.0) == 1
    assert choose_camera_index(visibilities, target_quantile=0.5) == 2
    assert choose_camera_index(visibilities, target_quantile=5.0 / 6.0) == 0


def test_sample_output_paths_are_dataset_local(tmp_path: Path) -> None:
    # Regression guard for the requested sibling layout: scenes/ and samples/.
    dataset_root = tmp_path / "plcs" / "single_object"
    scenes_dir = dataset_root / "scenes"
    scenes_dir.mkdir(parents=True)

    assert scenes_dir.parent / "samples" == dataset_root / "samples"

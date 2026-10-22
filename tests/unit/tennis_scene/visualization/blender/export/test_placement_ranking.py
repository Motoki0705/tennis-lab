"""Placement ranking: windows, t0 tie-breaks, yaw variation, quota, output."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.submodules.configuration import SmplAssetPaths
from src.tennis_scene.motion_alignment.temporal import TemporalPlacementConfig
from src.tennis_scene.pipeline.body_types import BodyGeometry, BodyParameters
from src.tennis_scene.pipeline.components.gvhmr import (
    GVHMROutput,
    RecoveredBody,
)
from src.tennis_scene.pipeline.components.player_reconstruction import (
    PlacementTrace,
    PlayerSkeleton,
    ReconstructedPlayers,
)
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.observation_types import GroupedObservations
from src.tennis_scene.visualization.blender.export import placement_ranking
from src.tennis_scene.visualization.blender.export.placement import PlacementInputs
from src.tennis_scene.visualization.blender.export.placement_ranking import (
    EXCLUSION_LOW_VALID_RATIO,
    EXCLUSION_NO_T0,
    RANKING_SCHEMA,
    PlacementRankCandidate,
    PlacementRankConfig,
    _heading_degrees,
    _segment_windows,
    _select_t0,
    _window_plan,
    _yaw_variation_deg,
    rank_placement_candidates,
    rank_trace_candidates,
    select_top_candidates,
    sort_candidates,
)
from src.utils.checksum import dual_sha256
from src.utils.video.writer import save_video_rgb

IDENTITY = 7
FRAMES = 10
FPS = 10.0
SIZE = (32, 24)
VERTICES = 3


def _trace(
    *,
    frames: NDArray[np.int64] | None = None,
    segment_ids: NDArray[np.int64] | None = None,
    valid: NDArray[np.bool_] | None = None,
    joint_counts: NDArray[np.int64] | None = None,
    weight_sums: NDArray[np.float64] | None = None,
    headings_deg: NDArray[np.float64] | None = None,
    yaw_correction_deg: NDArray[np.float64] | None = None,
) -> PlacementTrace:
    if frames is None:
        frames = np.arange(0, 8, dtype=np.int64)
    count = len(frames)
    if segment_ids is None:
        segment_ids = np.zeros(count, np.int64)
    if valid is None:
        valid = np.ones(count, bool)
    if joint_counts is None:
        joint_counts = np.full(count, 17, np.int64)
    if weight_sums is None:
        weight_sums = np.ones(count, np.float64)
    if headings_deg is None:
        headings_deg = np.zeros(count, np.float64)
    if yaw_correction_deg is None:
        yaw_correction_deg = np.zeros(count, np.float64)
    target: NDArray[np.float32] = np.full((count, 17, 3), np.nan, np.float32)
    for index in range(count):
        target[index, : int(joint_counts[index])] = np.array([index, 0, 1], np.float32)
    weights: NDArray[np.float64] = np.zeros((count, 17), np.float64)
    weights[:, 0] = weight_sums
    joints_local: NDArray[np.float64] = np.zeros((count, 17, 3), np.float64)
    raw = np.radians(headings_deg - yaw_correction_deg)
    joints_local[:, 11, 0] = np.cos(raw)
    joints_local[:, 11, 1] = np.sin(raw)
    joints_local[:, 12] = 0.0
    last_segment = int(segment_ids.max()) if count else -1
    return PlacementTrace(
        identity=0,
        frames=frames,
        segment_ids=segment_ids,
        segment_camera_ids=tuple(f"cam{value}" for value in range(last_segment + 1)),
        vertices_local=np.zeros((count, 3, 3), np.float32),
        joints_local=joints_local,
        target_joints=target,
        target_weights=weights,
        scale=1.0,
        hip_position=np.zeros((count, 3), np.float64),
        yaw_correction=np.radians(yaw_correction_deg),
        valid=valid,
        reasons=np.zeros(count, np.uint8),
    )


def _candidate(
    identity: int,
    start: int,
    *,
    yaw_variation_deg: float,
    valid_ratio: float,
    segment_id: int = 0,
) -> PlacementRankCandidate:
    return PlacementRankCandidate(
        identity=identity,
        segment_id=segment_id,
        gvhmr_camera_id=f"cam{segment_id}",
        start=start,
        stop=start + 4,
        t0=start + 1,
        t0_valid_joints=17,
        valid_ratio=valid_ratio,
        yaw_variation_deg=yaw_variation_deg,
        yaw_correction_range_deg=0.0,
        scale=1.0,
        shortened=False,
    )


def test_window_enumeration_stays_inside_segments_and_shortens() -> None:
    frames = np.concatenate(
        [
            np.arange(0, 6),
            np.arange(10, 18),
            np.arange(20, 23),
        ]
    ).astype(np.int64)
    segment_ids = np.concatenate(
        [
            np.zeros(6, np.int64),
            np.ones(8, np.int64),
            np.full(3, 2, np.int64),
        ]
    )
    trace = _trace(frames=frames, segment_ids=segment_ids)
    plan = _window_plan(
        PlacementRankConfig(window_seconds=4.0, stride_seconds=2.0), 1.0
    )

    windows = list(_segment_windows(trace, plan))

    assert windows == [
        (0, 4, False),
        (2, 6, False),
        (10, 14, False),
        (12, 16, False),
        (14, 18, False),
        (20, 23, True),
    ]
    # Every window is inside one segment and uses stop-exclusive bounds.
    for start, stop, shortened in windows:
        assert start < stop
        if shortened:
            assert (start, stop) == (20, 23)


def test_short_segment_is_one_shortened_candidate() -> None:
    trace = _trace(
        frames=np.arange(10, 13, dtype=np.int64),
        segment_ids=np.zeros(3, np.int64),
    )
    result = rank_trace_candidates(
        trace,
        config=PlacementRankConfig(
            window_seconds=4.0, stride_seconds=1.0, min_valid_ratio=0.0
        ),
        fps=1.0,
    )

    assert result.windows == 1
    assert result.shortened_windows == 1
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert (candidate.start, candidate.stop, candidate.shortened) == (10, 13, True)


def test_rounding_window_or_stride_to_zero_frames_is_rejected() -> None:
    with pytest.raises(ValueError, match="window and stride"):
        _window_plan(
            PlacementRankConfig(window_seconds=0.4, stride_seconds=1.0), 1.0
        )
    with pytest.raises(ValueError, match="window and stride"):
        _window_plan(
            PlacementRankConfig(window_seconds=4.0, stride_seconds=0.4), 1.0
        )


def test_t0_requires_a_valid_frame_with_fifteen_valid_joints() -> None:
    trace = _trace(
        frames=np.arange(0, 4, dtype=np.int64),
        joint_counts=np.array([14, 15, 10, 17], np.int64),
        valid=np.array([True, True, True, False], bool),
    )
    assert _select_t0(trace, np.arange(4), start=0, stop=4) == (1, 15)

    short = _trace(
        frames=np.arange(0, 3, dtype=np.int64),
        joint_counts=np.array([14, 14, 14], np.int64),
    )
    assert _select_t0(short, np.arange(3), start=0, stop=3) is None


def test_t0_tie_break_count_then_weight_then_centre_then_earliest() -> None:
    trace = _trace(
        frames=np.arange(0, 5, dtype=np.int64),
        joint_counts=np.array([16, 16, 16, 16, 15], np.int64),
        weight_sums=np.array([1.0, 5.0, 5.0, 5.0, 10.0], np.float64),
    )
    # Highest joint count first, then the largest weight sum, then the frame
    # nearest the window centre (frame 2).
    assert _select_t0(trace, np.arange(5), start=0, stop=5) == (2, 16)

    even = _trace(
        frames=np.arange(0, 4, dtype=np.int64),
        joint_counts=np.full(4, 16, np.int64),
        weight_sums=np.ones(4, np.float64),
    )
    # Centre is 1.5; frames 1 and 2 tie and the earlier frame wins.
    assert _select_t0(even, np.arange(4), start=0, stop=4) == (1, 16)


def test_t0_weight_sum_ignores_weights_at_nan_joints() -> None:
    trace = _trace(
        frames=np.arange(0, 3, dtype=np.int64),
        joint_counts=np.full(3, 15, np.int64),
        weight_sums=np.array([1.0, 5.0, 5.0], np.float64),
    )
    weights = np.asarray(trace.target_weights, np.float64).copy()
    weights[:, 15] = np.array([100.0, 1.0, 1.0])
    trace = replace(trace, target_weights=weights)

    # The huge weight sits on a NaN joint and must not win the tie-break.
    assert _select_t0(trace, np.arange(3), start=0, stop=3) == (1, 15)


def test_yaw_variation_does_not_cross_invalid_frames() -> None:
    heading = np.array([0.0, 10.0, 0.0, 0.0, 100.0, 110.0])
    valid = np.array([True, True, False, False, True, True])

    assert _yaw_variation_deg(heading, valid) == pytest.approx(20.0)


def test_yaw_variation_unwraps_angle_wrap() -> None:
    heading = np.array([170.0, -170.0])
    valid = np.array([True, True])

    assert _yaw_variation_deg(heading, valid) == pytest.approx(20.0)


def test_heading_applies_yaw_correction_to_the_hip_vector() -> None:
    joints: NDArray[np.float64] = np.zeros((1, 17, 3), np.float64)
    joints[0, 11] = np.array([1.0, 0.0, 0.0])

    no_correction = _heading_degrees(joints, np.array([0.0]))
    quarter_turn = _heading_degrees(joints, np.array([np.pi / 2.0]))

    assert no_correction[0] == pytest.approx(0.0)
    assert quarter_turn[0] == pytest.approx(90.0)


def test_yaw_variation_comes_from_hips_not_only_yaw_correction() -> None:
    rotating_hips = _trace(
        frames=np.arange(0, 2, dtype=np.int64),
        headings_deg=np.array([0.0, 45.0]),
        yaw_correction_deg=np.zeros(2),
    )
    result = rank_trace_candidates(
        rotating_hips,
        config=PlacementRankConfig(
            window_seconds=2.0, stride_seconds=1.0, min_valid_ratio=0.0
        ),
        fps=1.0,
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].yaw_variation_deg == pytest.approx(45.0)
    assert result.candidates[0].yaw_correction_range_deg == pytest.approx(0.0)

    correction_only = _trace(
        frames=np.arange(0, 2, dtype=np.int64),
        headings_deg=np.zeros(2),
        yaw_correction_deg=np.array([0.0, 45.0]),
    )
    result = rank_trace_candidates(
        correction_only,
        config=PlacementRankConfig(
            window_seconds=2.0, stride_seconds=1.0, min_valid_ratio=0.0
        ),
        fps=1.0,
    )
    # The correction range is reported separately; the heading itself does not
    # change because the hip vector was rotated by the inverse amount.
    assert result.candidates[0].yaw_variation_deg == pytest.approx(0.0)
    assert result.candidates[0].yaw_correction_range_deg == pytest.approx(45.0)


def test_window_exclusions_report_no_t0_and_low_valid_ratio() -> None:
    valid = np.array([True, False, False, False, False, False, False, False])
    trace = _trace(
        frames=np.arange(0, 8, dtype=np.int64),
        valid=valid,
        joint_counts=np.full(8, 17, np.int64),
    )
    result = rank_trace_candidates(
        trace,
        config=PlacementRankConfig(
            window_seconds=4.0, stride_seconds=4.0, min_valid_ratio=0.9
        ),
        fps=1.0,
    )

    assert result.candidates == ()
    assert result.windows == 2
    assert result.exclusions == {
        EXCLUSION_NO_T0: 1,
        EXCLUSION_LOW_VALID_RATIO: 1,
    }


def test_sort_order_is_yaw_then_valid_ratio_then_start() -> None:
    low_ratio = _candidate(0, 0, yaw_variation_deg=10.0, valid_ratio=0.8)
    high_ratio_late = _candidate(0, 3, yaw_variation_deg=10.0, valid_ratio=0.9)
    high_ratio_early = _candidate(0, 1, yaw_variation_deg=10.0, valid_ratio=0.9)
    highest_yaw = _candidate(0, 2, yaw_variation_deg=20.0, valid_ratio=0.5)

    ordered = sort_candidates(
        [low_ratio, high_ratio_late, high_ratio_early, highest_yaw]
    )

    assert [candidate.start for candidate in ordered] == [2, 1, 3, 0]


def test_selection_unions_global_top_k_with_each_identity_top3() -> None:
    candidates = [
        *[
            _candidate(0, start, yaw_variation_deg=100.0 - start, valid_ratio=0.9)
            for start in range(10)
        ],
        *[
            _candidate(1, start, yaw_variation_deg=10.0 - start, valid_ratio=0.9)
            for start in range(10)
        ],
    ]
    # The global top 10 all belong to identity 0.  Identity 1's top 3 are added
    # instead of replacing slots in the global top 10.
    selected, quota = select_top_candidates(
        candidates, identities=(0, 1), top_k=10
    )

    assert [candidate.identity for candidate in selected] == [0] * 10 + [1] * 3
    assert [candidate.yaw_variation_deg for candidate in selected] == [
        100.0,
        99.0,
        98.0,
        97.0,
        96.0,
        95.0,
        94.0,
        93.0,
        92.0,
        91.0,
        10.0,
        9.0,
        8.0,
    ]
    assert [candidate.start for candidate in selected] == list(range(10)) + [0, 1, 2]
    assert [candidate.overall_rank for candidate in selected] == list(range(1, 14))
    assert [candidate.selected_by for candidate in selected] == (
        [("overall", "identity_quota")] * 3
        + [("overall",)] * 7
        + [("identity_quota",)] * 3
    )
    assert len({(candidate.identity, candidate.start) for candidate in selected}) == 13
    assert quota["top_k"] == 10
    assert quota["identity_quota"] == 3
    assert quota["candidate_count"] == 20
    assert quota["selected_count"] == 13
    assert quota["selected_by_overall"] == 10
    assert quota["added_by_identity_quota"] == 3
    assert quota["capacity_shortage"] == 0
    assert quota["identity_shortages"] == []
    assert quota["identities"] == [
        {"identity": 0, "candidates": 10, "selected": 10},
        {"identity": 1, "candidates": 10, "selected": 3},
    ]
    assert set(quota) == {
        "top_k",
        "identity_quota",
        "candidate_count",
        "selected_count",
        "selected_by_overall",
        "added_by_identity_quota",
        "capacity_shortage",
        "identity_shortages",
        "identities",
    }


def test_duplicate_input_candidates_are_rejected() -> None:
    candidate = _candidate(2, 5, yaw_variation_deg=10.0, valid_ratio=0.9)
    same_window = _candidate(2, 5, yaw_variation_deg=99.0, valid_ratio=0.5)

    for duplicate in (candidate, same_window):
        with pytest.raises(ValueError, match="Duplicate rank candidates") as error:
            select_top_candidates(
                [candidate, duplicate], identities=(2,), top_k=3
            )
        message = str(error.value)
        assert "identity=2" in message
        assert "segment_id=0" in message
        assert "start=5" in message
        assert "stop=9" in message


def test_top_k_smaller_than_the_identity_quota_still_adds_each_top3() -> None:
    candidates = [
        _candidate(identity, start, yaw_variation_deg=10.0, valid_ratio=0.9)
        for identity in (0, 1)
        for start in range(4)
    ]
    selected, quota = select_top_candidates(
        candidates, identities=(0, 1), top_k=4
    )

    # The global top 4 still wins its slots; each identity's top 3 is added on
    # top of that, so the output exceeds top_k without dropping a global pick.
    assert [(candidate.identity, candidate.start) for candidate in selected] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
    ]
    assert [candidate.overall_rank for candidate in selected] == list(range(1, 7))
    assert [candidate.selected_by for candidate in selected] == [
        ("overall", "identity_quota"),
        ("overall", "identity_quota"),
        ("overall", "identity_quota"),
        ("overall", "identity_quota"),
        ("identity_quota",),
        ("identity_quota",),
    ]
    assert quota["selected_count"] == 6
    assert quota["selected_by_overall"] == 4
    assert quota["added_by_identity_quota"] == 2
    assert quota["capacity_shortage"] == 0
    assert quota["identities"] == [
        {"identity": 0, "candidates": 4, "selected": 3},
        {"identity": 1, "candidates": 4, "selected": 3},
    ]


def test_identity_candidate_shortage_is_reported() -> None:
    candidates = [
        *[
            _candidate(0, start, yaw_variation_deg=10.0, valid_ratio=0.9)
            for start in range(2)
        ],
        *[
            _candidate(1, start, yaw_variation_deg=20.0, valid_ratio=0.9)
            for start in range(4)
        ],
    ]
    selected, quota = select_top_candidates(
        candidates, identities=(0, 1), top_k=5
    )

    # The global top 5 keeps identity 1's four and identity 0's first; the
    # identity quota adds identity 0's remaining candidate.
    assert len(selected) == 6
    assert [candidate.overall_rank for candidate in selected] == list(range(1, 7))
    assert [candidate.selected_by for candidate in selected] == [
        ("overall", "identity_quota"),
        ("overall", "identity_quota"),
        ("overall", "identity_quota"),
        ("overall",),
        ("overall", "identity_quota"),
        ("identity_quota",),
    ]
    assert quota["selected_by_overall"] == 5
    assert quota["added_by_identity_quota"] == 1
    assert quota["identity_shortages"] == [
        {"identity": 0, "available": 2, "required": 3, "missing": 1}
    ]
    assert quota["capacity_shortage"] == 0
    assert quota["identities"] == [
        {"identity": 0, "candidates": 2, "selected": 2},
        {"identity": 1, "candidates": 4, "selected": 4},
    ]


def test_capacity_shortage_when_candidates_are_fewer_than_top_k() -> None:
    candidates = [
        _candidate(0, start, yaw_variation_deg=10.0 - start, valid_ratio=0.9)
        for start in range(3)
    ]
    selected, quota = select_top_candidates(
        candidates, identities=(0, 1), top_k=10
    )

    assert len(selected) == 3
    assert [candidate.overall_rank for candidate in selected] == [1, 2, 3]
    assert all(
        candidate.selected_by == ("overall", "identity_quota")
        for candidate in selected
    )
    assert quota["selected_count"] == 3
    assert quota["selected_by_overall"] == 3
    assert quota["added_by_identity_quota"] == 0
    assert quota["capacity_shortage"] == 7
    assert quota["identity_shortages"] == [
        {"identity": 1, "available": 0, "required": 3, "missing": 3}
    ]
    assert quota["identities"] == [
        {"identity": 0, "candidates": 3, "selected": 3},
        {"identity": 1, "candidates": 0, "selected": 0},
    ]


class _UnusedBody:
    """The ranking tests replace trace_player_placement, so no body is read."""

    @property
    def root_regressor(self) -> NDArray[np.float64]:
        raise AssertionError("ranking must reuse trace_player_placement")

    def reconstruct(self, parameters: BodyParameters) -> BodyGeometry:
        raise AssertionError("ranking must not reconstruct a body directly")

    def unload(self) -> None:
        raise AssertionError("test body is not owned by ranking")


def _integration_inputs(
    tmp_path: Path,
) -> tuple[PlacementInputs, PlacementTrace, Path]:
    video = tmp_path / "cam0.mp4"
    save_video_rgb(
        np.zeros((FRAMES, SIZE[1], SIZE[0], 3), np.uint8), video, fps=FPS
    )
    source = ClipSource(
        "synthetic",
        (
            SourceVideo(
                "cam0", video, dual_sha256(video), FRAMES, FPS, *SIZE
            ),
        ),
    )
    grouped = GroupedObservations(
        np.array([IDENTITY], np.int64),
        np.zeros((1, 1, FRAMES, 17, 2), np.float32),
        np.zeros((1, 1, FRAMES, 17), np.float32),
        np.zeros((1, 1, FRAMES, 17), bool),
        np.full((1, 1, FRAMES), -1, np.int64),
    )
    skeleton = PlayerSkeleton(
        np.zeros((1, FRAMES, 17, 3), np.float32),
        np.zeros((1, FRAMES, 17), bool),
        np.zeros((1, FRAMES, 17), np.uint8),
        np.zeros((1, 1, FRAMES, 17), bool),
        np.zeros((1, 1, FRAMES, 17), np.float32),
    )
    trace = replace(
        _trace(
            frames=np.arange(0, FRAMES, dtype=np.int64),
            valid=np.ones(FRAMES, bool),
            joint_counts=np.full(FRAMES, 17, np.int64),
            headings_deg=np.linspace(0.0, 90.0, FRAMES),
            yaw_correction_deg=np.linspace(0.0, 10.0, FRAMES),
        ),
        identity=IDENTITY,
    )
    placed = ReconstructedPlayers(
        position=np.zeros((1, FRAMES, 3), np.float32),
        yaw=np.zeros((1, FRAMES), np.float32),
        root_valid=np.asarray(trace.valid)[None].copy(),
        heading_valid=np.zeros((1, FRAMES), bool),
        smpl_valid=np.ones((1, FRAMES), bool),
        body_pose=np.zeros((1, FRAMES, 63), np.float32),
        global_orient=np.zeros((1, FRAMES, 3), np.float32),
        betas=np.zeros((1, 10), np.float32),
        vertices_local=np.zeros((1, FRAMES, VERTICES, 3), np.float32),
        root_reasons=np.zeros((1, FRAMES), np.uint8),
        mesh_reprojection_px=np.zeros((1, FRAMES), np.float32),
        diagnostics={},
    )
    joint_regressor: NDArray[np.float32] = np.zeros((1, VERTICES), np.float32)
    assets = SmplAssetPaths(
        hmr2_mean_params=tmp_path / "hmr2_mean_params.npz",
        smplx_to_smpl=tmp_path / "smplx_to_smpl.pt",
        smpl_coco17_regressor=tmp_path / "smpl_coco17.pt",
        smplx_verts437=tmp_path / "smplx_verts437.pt",
        smpl_neutral_joint_regressor=tmp_path / "joint_regressor.pt",
    )
    inputs = PlacementInputs(
        source=source,
        cameras=(),
        grouped=grouped,
        skeleton=skeleton,
        recovered=GVHMROutput((RecoveredBody(IDENTITY, "cam0", ()),)),
        placed=placed,
        placement_config=TemporalPlacementConfig(),
        joint_regressor=joint_regressor,
        body_assets=assets,
        body_models_dir=tmp_path / "body_models",
        asset_provenance={},
        artifact_provenance={
            "body_placement": {
                "artifact_id": "artifact-1",
                "sha256": "b" * 64,
                "schema": "placed_bodies",
                "version": 1,
                "path": "body_placement.npz",
            }
        },
        store_provenance={
            "scene_index": str(tmp_path / "store/scene.json"),
            "index_sha256": "a" * 64,
            "source_sha256": "c" * 64,
            "clip_id": "synthetic",
            "camera_ids": ["cam0"],
        },
    )
    return inputs, trace, video


def _rank_config() -> PlacementRankConfig:
    return PlacementRankConfig(
        window_seconds=0.4,
        stride_seconds=0.1,
        min_valid_ratio=0.5,
        top_k=10,
    )


def _patch_trace(
    monkeypatch: pytest.MonkeyPatch, trace: PlacementTrace
) -> None:
    def fake_trace(*args: Any, **kwargs: Any) -> PlacementTrace:
        return trace

    monkeypatch.setattr(placement_ranking, "trace_player_placement", fake_trace)


def _write_ranking(
    inputs: PlacementInputs,
    trace: PlacementTrace,
    video: Path,
    output: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    _patch_trace(monkeypatch, trace)
    ranking: Path = rank_placement_candidates(
        inputs,
        body=_UnusedBody(),
        config=_rank_config(),
        plate_camera_id="cam0",
        plate_video_path=video,
        output_path=output,
        resolved_config={"mode": "rank", "rank": {"top_k": 10}},
        git_commit="d" * 40,
    )
    return ranking


def test_rank_writes_only_ranking_json_with_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, trace, video = _integration_inputs(tmp_path)
    output = tmp_path / "run" / "ranking.json"

    result = _write_ranking(inputs, trace, video, output, monkeypatch)

    assert result == output
    assert not (tmp_path / "run" / "bundle").exists()
    document = json.loads(output.read_text())
    assert document["schema"] == RANKING_SCHEMA
    assert document["store"]["index_sha256"] == "a" * 64
    assert document["artifacts"]["body_placement"]["artifact_id"] == "artifact-1"
    assert document["config"]["resolved"]["mode"] == "rank"
    assert document["plate"] == {
        "camera_id": "cam0",
        "video": str(video),
        "video_sha256": dual_sha256(video),
    }
    assert document["parameters"] == {
        "fps": FPS,
        "window_frames": 4,
        "stride_frames": 1,
    }
    assert document["counts"] == {
        "identities": 1,
        "windows": 7,
        "shortened_windows": 0,
        "candidates": 7,
        "selected": 7,
    }
    assert document["exclusions"][EXCLUSION_NO_T0] == 0
    assert document["exclusions"][EXCLUSION_LOW_VALID_RATIO] == 0
    assert document["quota"] == {
        "top_k": 10,
        "identity_quota": 3,
        "candidate_count": 7,
        "selected_count": 7,
        "selected_by_overall": 7,
        "added_by_identity_quota": 0,
        "capacity_shortage": 3,
        "identity_shortages": [],
        "identities": [{"identity": IDENTITY, "candidates": 7, "selected": 7}],
    }
    assert document["verification"][str(IDENTITY)]["status"] == "ok"
    assert document["verification"][str(IDENTITY)]["saved_placement_check"] == "ok"
    assert (
        document["verification"][str(IDENTITY)]["max_abs_difference_m"] == 0.0
    )
    assert document["git_commit"] == "d" * 40

    candidate = document["candidates"][0]
    assert set(candidate) == {
        "identity",
        "segment_id",
        "gvhmr_camera_id",
        "start",
        "stop",
        "t0",
        "t0_valid_joints",
        "valid_ratio",
        "yaw_variation_deg",
        "yaw_correction_range_deg",
        "scale",
        "shortened",
        "overall_rank",
        "selected_by",
    }
    assert candidate["identity"] == IDENTITY
    assert candidate["gvhmr_camera_id"] == "cam0"
    assert candidate["t0_valid_joints"] == 17
    assert candidate["shortened"] is False
    assert [item["overall_rank"] for item in document["candidates"]] == list(
        range(1, 8)
    )
    assert document["candidates"][0]["selected_by"] == [
        "overall",
        "identity_quota",
    ]
    assert document["candidates"][2]["selected_by"] == [
        "overall",
        "identity_quota",
    ]
    assert document["candidates"][3]["selected_by"] == ["overall"]
    yaw_values = [
        item["yaw_variation_deg"] for item in document["candidates"]
    ]
    assert yaw_values == sorted(yaw_values, reverse=True)


def test_rank_refuses_to_overwrite_ranking_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, trace, video = _integration_inputs(tmp_path)
    output = tmp_path / "ranking.json"
    _write_ranking(inputs, trace, video, output, monkeypatch)
    original = output.read_bytes()

    with pytest.raises(FileExistsError):
        _write_ranking(inputs, trace, video, output, monkeypatch)

    assert output.read_bytes() == original

    existing_dir = tmp_path / "already-a-directory"
    existing_dir.mkdir()
    with pytest.raises(FileExistsError):
        _write_ranking(inputs, trace, video, existing_dir, monkeypatch)
    assert existing_dir.is_dir()


def test_rank_rejects_plate_video_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, trace, video = _integration_inputs(tmp_path)
    other = tmp_path / "other.mp4"
    frames: NDArray[np.uint8] = np.zeros(
        (FRAMES, SIZE[1], SIZE[0], 3), np.uint8
    )
    frames[..., 0] = 9
    save_video_rgb(frames, other, fps=FPS)
    _patch_trace(monkeypatch, trace)
    output = tmp_path / "ranking.json"

    with pytest.raises(ValueError, match="Plate video SHA-256 mismatch"):
        rank_placement_candidates(
            inputs,
            body=_UnusedBody(),
            config=_rank_config(),
            plate_camera_id="cam0",
            plate_video_path=other,
            output_path=output,
        )

    assert not output.exists()


def test_rank_rejects_a_saved_placement_that_disagrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, trace, video = _integration_inputs(tmp_path)
    assert inputs.placed.vertices_local is not None
    inputs.placed.vertices_local[0, 3, 0, 0] = 1.0
    _patch_trace(monkeypatch, trace)
    output = tmp_path / "ranking.json"

    with pytest.raises(ValueError, match="differs from the saved body_placement") as error:
        rank_placement_candidates(
            inputs,
            body=_UnusedBody(),
            config=_rank_config(),
            plate_camera_id="cam0",
            plate_video_path=video,
            output_path=output,
        )

    assert "identity=7" in str(error.value)
    assert "source_frame=3" in str(error.value)
    assert "max_abs_difference_m=" in str(error.value)
    assert not output.exists()


def test_rank_rejects_missing_scale_with_saved_valid_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, trace, video = _integration_inputs(tmp_path)
    trace = replace(
        trace,
        scale=None,
        valid=np.zeros(len(trace.frames), bool),
    )
    inputs.placed.root_valid[:] = False
    _patch_trace(monkeypatch, trace)
    output = tmp_path / "ranking.json"

    with pytest.raises(ValueError, match="no fitted scale") as error:
        rank_placement_candidates(
            inputs,
            body=_UnusedBody(),
            config=_rank_config(),
            plate_camera_id="cam0",
            plate_video_path=video,
            output_path=output,
        )

    message = str(error.value)
    assert "identity=7" in message
    assert "10 valid SMPL" in message
    assert "source_frames=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]" in message
    assert not output.exists()


def test_rank_reports_no_scale_when_saved_mesh_has_no_valid_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, trace, video = _integration_inputs(tmp_path)
    trace = replace(
        trace,
        scale=None,
        valid=np.zeros(len(trace.frames), bool),
    )
    inputs.placed.root_valid[:] = False
    inputs.placed.smpl_valid[:] = False
    output = tmp_path / "ranking.json"

    result = _write_ranking(inputs, trace, video, output, monkeypatch)

    document = json.loads(result.read_text())
    verification = document["verification"][str(IDENTITY)]
    assert verification["status"] == "no_scale"
    assert verification["saved_placement_check"] == "skipped_no_scale"
    assert verification["smpl_valid_frames"] == 0
    assert verification["max_abs_difference_m"] is None
    assert document["counts"]["candidates"] == 0
    assert document["counts"]["selected"] == 0
    assert document["exclusions"][EXCLUSION_NO_T0] == 7
    assert document["quota"] == {
        "top_k": 10,
        "identity_quota": 3,
        "candidate_count": 0,
        "selected_count": 0,
        "selected_by_overall": 0,
        "added_by_identity_quota": 0,
        "capacity_shortage": 10,
        "identity_shortages": [
            {"identity": IDENTITY, "available": 0, "required": 3, "missing": 3}
        ],
        "identities": [{"identity": IDENTITY, "candidates": 0, "selected": 0}],
    }


def test_rank_rejects_root_validity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, trace, video = _integration_inputs(tmp_path)
    inputs.placed.root_valid[0, 3] = False
    _patch_trace(monkeypatch, trace)
    output = tmp_path / "ranking.json"

    with pytest.raises(ValueError, match="root_valid disagrees"):
        rank_placement_candidates(
            inputs,
            body=_UnusedBody(),
            config=_rank_config(),
            plate_camera_id="cam0",
            plate_video_path=video,
            output_path=output,
        )

    assert not output.exists()

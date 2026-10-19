"""Independent track contracts, interpolation and detection filtering."""

import numpy as np
import pytest
import torch

import src.submodules.models.tracker.common as tracker_common
from src.submodules.models import PersonDetectionResult
from src.submodules.models.tracker.common import (
    TrackRequest,
    TrackResult,
    build_track_tensor,
    resolve_track_frame_count,
    select_and_complete_tracks,
    sort_tracks,
    stitch_single_subject_tracklets,
)


def test_track_frame_limit_is_explicit_and_never_exceeds_video() -> None:
    request = TrackRequest("video.mp4", num_tracks=1, interactive=False, max_frames=8)

    assert resolve_track_frame_count(744, request) == 8
    assert resolve_track_frame_count(4, request) == 4
    assert (
        resolve_track_frame_count(
            744, TrackRequest("video.mp4", num_tracks=1, interactive=False)
        )
        == 744
    )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_track_frame_limit_rejects_invalid_values(value: object) -> None:
    request = TrackRequest(
        "video.mp4",
        num_tracks=1,
        interactive=False,
        max_frames=value,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="max_frames"):
        resolve_track_frame_count(10, request)


def make_history() -> list[list[dict]]:
    """Two ids: id 1 large boxes (frames 0,1,3), id 2 small boxes (all frames)."""
    big = lambda x: np.array([x, 0.0, x + 100.0, 200.0])  # noqa: E731
    small = lambda x: np.array([x, 0.0, x + 10.0, 20.0])  # noqa: E731
    return [
        [{"id": 1, "bbx_xyxy": big(0.0)}, {"id": 2, "bbx_xyxy": small(0.0)}],
        [{"id": 1, "bbx_xyxy": big(10.0)}, {"id": 2, "bbx_xyxy": small(1.0)}],
        [{"id": 2, "bbx_xyxy": small(2.0)}],
        [{"id": 1, "bbx_xyxy": big(30.0)}, {"id": 2, "bbx_xyxy": small(3.0)}],
    ]


class TestSortTracks:
    def test_orders_by_accumulated_area(self):
        id_to_frame_ids, id_to_bbx_xyxys, ids_by_area = sort_tracks(make_history())
        # id 1: 3 frames of 100x200 >> id 2: 4 frames of 10x20
        assert ids_by_area == [1, 2]
        assert id_to_frame_ids[1] == [0, 1, 3]
        assert id_to_frame_ids[2] == [0, 1, 2, 3]
        assert id_to_bbx_xyxys[1].shape == (3, 4)

    def test_empty_history(self):
        id_to_frame_ids, _, ids_by_area = sort_tracks([[], []])
        assert ids_by_area == []
        assert id_to_frame_ids == {}


class TestBuildTrackTensor:
    def test_interpolates_and_covers_all_frames(self):
        id_to_frame_ids, id_to_bbx_xyxys, _ = sort_tracks(make_history())
        track = build_track_tensor(id_to_frame_ids[1], id_to_bbx_xyxys[1], num_frames=4)

        assert track.shape == (4, 4)
        assert track.dtype == torch.float32
        assert (track.sum(dim=1) != 0).all()
        # Values stay within the range of observed boxes (smoothing is convex).
        assert track[:, 0].min() >= 0.0 - 1e-5
        assert track[:, 0].max() <= 30.0 + 1e-5


class TestTrackResult:
    def test_bbx_xys_square_and_enlarged(self):
        # Static 100x200 box centered at (100, 150) over 3 frames.
        boxes = torch.tensor([[50.0, 50.0, 150.0, 250.0]]).repeat(3, 1)
        result = TrackResult(tracks={7: boxes}, num_frames=3)

        assert result.track_ids == [7]
        xys = result.bbx_xys(7, base_enlarge=1.2)
        assert xys.shape == (3, 3)
        torch.testing.assert_close(xys[:, 0], torch.full((3,), 100.0))
        torch.testing.assert_close(xys[:, 1], torch.full((3,), 150.0))
        # w=100 < h*192/256=150 -> w:=150; size = max(h, w) * 1.2 = 240
        torch.testing.assert_close(xys[:, 2], torch.full((3,), 240.0))

    def test_observed_mask_distinguishes_interpolated_frames(self):
        result = select_and_complete_tracks(
            make_history(),
            TrackRequest("video.mp4", num_tracks=1, interactive=False),
            num_frames=4,
        )

        torch.testing.assert_close(
            result.observed_mask(1),
            torch.tensor([True, True, False, True]),
        )


def test_single_subject_stitching_preserves_observations_across_id_switch() -> None:
    history = [
        [{"id": 4, "bbx_xyxy": np.asarray([0.0, 0.0, 20.0, 40.0])}],
        [{"id": 9, "bbx_xyxy": np.asarray([1.0, 0.0, 21.0, 40.0])}],
        [],
    ]

    stitched = stitch_single_subject_tracklets(history)
    result = select_and_complete_tracks(
        stitched,
        TrackRequest("video.mp4", num_tracks=1, interactive=False),
        num_frames=3,
    )

    assert [frame[0]["source_track_id"] for frame in stitched[:2]] == [4, 9]
    assert result.track_ids == [0]
    torch.testing.assert_close(
        result.observed_mask(0), torch.tensor([True, True, False])
    )


def test_requested_track_count_must_be_available() -> None:
    one_track_history = [[make_history()[0][0]], [make_history()[1][0]]]
    with pytest.raises(RuntimeError, match="Requested 2 person tracks"):
        select_and_complete_tracks(
            one_track_history,
            TrackRequest("video.mp4", num_tracks=2, interactive=False),
            num_frames=2,
        )


def test_completed_track_interpolates_then_smooths_missing_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoothing_inputs: list[torch.Tensor] = []

    def _fake_smooth(
        track: torch.Tensor, *, window_size: int, dim: int
    ) -> torch.Tensor:
        assert window_size == 5
        assert dim == 0
        smoothing_inputs.append(track.clone())
        return track + 0.25

    monkeypatch.setattr(tracker_common, "moving_average_smooth", _fake_smooth)
    history = [
        [{"id": 7, "bbx_xyxy": np.array([10, 20, 30, 50], dtype=np.float32)}],
        [],
        [{"id": 7, "bbx_xyxy": np.array([14, 24, 34, 54], dtype=np.float32)}],
    ]

    result = select_and_complete_tracks(
        history,
        TrackRequest("camera-near.mp4", num_tracks=1, interactive=False),
        num_frames=3,
    )

    interpolated = torch.tensor(
        [
            [10.0, 20.0, 30.0, 50.0],
            [12.0, 22.0, 32.0, 52.0],
            [14.0, 24.0, 34.0, 54.0],
        ]
    )
    assert result.track_ids == [7]
    assert result.tracks[7].shape == (3, 4)
    torch.testing.assert_close(smoothing_inputs[0], interpolated)
    torch.testing.assert_close(smoothing_inputs[1], interpolated + 0.25)
    torch.testing.assert_close(result.tracks[7], interpolated + 0.5)


def test_playing_area_filters_by_feet_and_preserves_scores() -> None:
    from src.submodules.models.tracker.common import (
        filter_detections_by_footpoint,
    )

    detections = PersonDetectionResult(
        np.array([[1, -5, 3, 5], [8, 8, 12, 10], [2, 0, 4, 11]], np.float32),
        np.array([0.8, 0.7, 0.9], np.float32),
    )
    polygon = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    selected = filter_detections_by_footpoint(detections, polygon)
    np.testing.assert_array_equal(selected.boxes_xyxy, detections.boxes_xyxy[:2])
    np.testing.assert_array_equal(selected.scores, detections.scores[:2])
    empty = filter_detections_by_footpoint(
        PersonDetectionResult(np.empty((0, 4), np.float32), np.empty(0, np.float32)),
        polygon,
    )
    assert empty.boxes_xyxy.shape == (0, 4)
    with pytest.raises(ValueError, match="finite polygon"):
        filter_detections_by_footpoint(detections, ((0.0, 0.0), (1.0, 1.0)))

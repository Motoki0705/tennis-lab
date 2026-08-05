"""Tests for src/submodules/models/tracker/yolo_tracker.py (pure helpers)."""

import numpy as np
import pytest
import torch

from src.submodules.models.tracker.common import (
    TrackRequest,
    TrackResult,
    build_track_tensor,
    select_and_complete_tracks,
    sort_tracks,
)


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


def test_requested_track_count_must_be_available() -> None:
    one_track_history = [[make_history()[0][0]], [make_history()[1][0]]]
    with pytest.raises(RuntimeError, match="Requested 2 person tracks"):
        select_and_complete_tracks(
            one_track_history,
            TrackRequest("video.mp4", num_tracks=2, interactive=False),
            num_frames=2,
        )

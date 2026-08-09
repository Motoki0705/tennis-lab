"""Unit tests for BLCS event extraction and bounce-frame resolution."""

from __future__ import annotations

import numpy as np

from src.tasks.blcs.visualization.rendering.scene_renderer import (
    bounce_frames_from_events,
    extract_ball_events,
    extract_ball_track_events,
    split_ball_tracks,
)
from src.utils.rendering.ball_renderer import BallEventType


class TestExtractBallEvents:
    def test_extracts_bounces_and_net_hits(self) -> None:
        meta = {
            "shots": [
                {
                    "shot_index": 0,
                    "t_start": 0,
                    "t_bounce1": 5,
                    "t_bounce2": -1,
                    "t_net": -1,
                },
                {
                    "shot_index": 1,
                    "t_start": 8,
                    "t_bounce1": 12,
                    "t_bounce2": 15,
                    "t_net": 18,
                },
            ]
        }

        events = extract_ball_events(meta)

        bounces = [e.frame_idx for e in events if e.event_type is BallEventType.BOUNCE]
        assert bounces == [5, 12, 15]
        net_hits = [
            e.frame_idx for e in events if e.event_type is BallEventType.NET_HIT
        ]
        assert net_hits == [18]
        # The first shot's start is the rally start, not a shot boundary.
        boundaries = [
            e.frame_idx for e in events if e.event_type is BallEventType.SHOT_BOUNDARY
        ]
        assert boundaries == [8]

    def test_empty_meta_yields_no_events(self) -> None:
        assert extract_ball_events({}) == []


class TestBounceFramesFromEvents:
    def test_prefers_event_metadata_over_detection(self) -> None:
        events = extract_ball_events(
            {"shots": [{"shot_index": 0, "t_start": 0, "t_bounce1": 3}]}
        )

        frames = bounce_frames_from_events(events)

        # Only the metadata bounce is reported, never the detected one too.
        assert frames.tolist() == [3]

    def test_events_without_bounces_remain_empty(self) -> None:
        events = extract_ball_events(
            {"shots": [{"shot_index": 0, "t_start": 0, "t_net": 4}]}
        )

        frames = bounce_frames_from_events(events)

        assert frames.tolist() == []


class TestMultiBallSceneHelpers:
    def test_splits_only_real_tracks_from_padded_object_axis(self) -> None:
        positions: np.ndarray = np.zeros((5, 3, 3), dtype=np.float32)

        tracks = split_ball_tracks({"ball_pos_world": positions, "num_balls": 2})

        assert len(tracks) == 2
        assert all(track.shape == (5, 3) for track in tracks)

    def test_extracts_events_for_requested_ball(self) -> None:
        meta = {
            "shots": [
                {
                    "ball_index": 0,
                    "shots": [{"shot_index": 0, "t_bounce1": 1}],
                },
                {
                    "ball_index": 1,
                    "shots": [{"shot_index": 0, "t_bounce1": 4}],
                },
            ]
        }

        events = extract_ball_track_events(meta, 1)

        assert [event.frame_idx for event in events] == [4]

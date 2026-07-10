"""Unit tests for BLCS event extraction and bounce-frame resolution."""

from __future__ import annotations

import numpy as np

from src.tasks.blcs.visualization.rendering.scene_renderer import (
    extract_ball_events,
    resolve_bounce_frames,
)
from src.utils.rendering.ball_renderer import BallEventType


def _bouncy_trajectory(num_frames: int = 20) -> np.ndarray:
    """Trajectory with a clear detectable ground bounce at frame 10."""
    positions: np.ndarray = np.zeros((num_frames, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(-2.0, 2.0, num_frames)
    positions[:, 2] = np.abs(np.linspace(-1.0, 1.0, num_frames))
    return positions


class TestExtractBallEvents:
    def test_extracts_bounces_and_net_hits(self) -> None:
        meta = {
            "shots": [
                {"shot_index": 0, "t_start": 0, "t_bounce1": 5, "t_bounce2": -1, "t_net": -1},
                {"shot_index": 1, "t_start": 8, "t_bounce1": 12, "t_bounce2": 15, "t_net": 18},
            ]
        }

        events = extract_ball_events(meta)

        bounces = [e.frame_idx for e in events if e.event_type is BallEventType.BOUNCE]
        assert bounces == [5, 12, 15]
        net_hits = [e.frame_idx for e in events if e.event_type is BallEventType.NET_HIT]
        assert net_hits == [18]
        # The first shot's start is the rally start, not a shot boundary.
        boundaries = [
            e.frame_idx for e in events if e.event_type is BallEventType.SHOT_BOUNDARY
        ]
        assert boundaries == [8]

    def test_empty_meta_yields_no_events(self) -> None:
        assert extract_ball_events({}) == []


class TestResolveBounceFrames:
    def test_prefers_event_metadata_over_detection(self) -> None:
        positions = _bouncy_trajectory()
        events = extract_ball_events(
            {"shots": [{"shot_index": 0, "t_start": 0, "t_bounce1": 3}]}
        )

        frames = resolve_bounce_frames(positions, events)

        # Only the metadata bounce is reported, never the detected one too.
        assert frames.tolist() == [3]

    def test_falls_back_to_detection_without_events(self) -> None:
        positions = _bouncy_trajectory()

        frames = resolve_bounce_frames(positions, None)

        assert frames.tolist() == [10]

    def test_falls_back_when_events_carry_no_bounces(self) -> None:
        positions = _bouncy_trajectory()
        events = extract_ball_events(
            {"shots": [{"shot_index": 0, "t_start": 0, "t_net": 4}]}
        )

        frames = resolve_bounce_frames(positions, events)

        assert frames.tolist() == [10]

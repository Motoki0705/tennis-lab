"""Artifact joins refuse incompatible camera, time and carrier axes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_points import (
    BallPointsInput,
    BallPointsModule,
)
from src.tennis_scene.pipeline.components.identity import PlayerIdentitiesOutput
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.input_assembly.observations import (
    gather_balls,
    gather_people,
    identified_people,
)
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.utils.geometry.temporal_smoothing import TemporalSmoothingConfig


def source() -> ClipSource:
    return ClipSource("join", tuple(SourceVideo(c, Path(f"{c}.mp4"), "test", 3, 30., 640, 480) for c in ("west", "east")))


def pose(camera: str, ids: list[int]) -> ObjectObservations:
    code: NDArray[np.float32] = np.arange(3 * len(ids) * 17, dtype=np.float32).reshape(1, 3, len(ids), 17)
    return ObjectObservations(
        (camera,), (640, 480), 30., np.stack((code + 1, code + 2), -1),
        np.full(code.shape, .8, np.float32), np.ones(code.shape[:3], bool),
        np.array([ids], np.int64), np.full((*code.shape[:3], 3), 12., np.float32),
    )


def test_gather_people_pads_carriers_without_sorting_tracks_or_cameras() -> None:
    west, east = pose("west", [41]), pose("east", [88, 6])
    west.observed[0, 1, 0] = False
    raw = gather_people(source(), {"pose_east": east, "pose_west": west})
    assert raw.camera_ids == ("west", "east")
    np.testing.assert_array_equal(raw.local_track_ids, [[41, -1], [88, 6]])
    np.testing.assert_array_equal(raw.observed, [[[True, False], [False, False], [True, False]], [[True, True]] * 3])
    for name in ("uv_px", "confidence", "boxes_xys"):
        np.testing.assert_array_equal(getattr(raw, name)[0, :, :1], getattr(west, name)[0])
        np.testing.assert_array_equal(getattr(raw, name)[0, :, 1:], 0)
        np.testing.assert_array_equal(getattr(raw, name)[1], getattr(east, name)[0])


@pytest.mark.parametrize("mismatch", ["camera", "fps", "frames", "size", "boxes"])
def test_gather_people_rejects_incompatible_artifacts(mismatch: str) -> None:
    row = pose("west", [41])
    if mismatch == "camera":
        row = replace(row, camera_ids=("east",))
    elif mismatch == "fps":
        row = replace(row, fps=29.97)
    elif mismatch == "frames":
        assert row.boxes_xys is not None
        row = replace(row, uv_px=row.uv_px[:, :2], confidence=row.confidence[:, :2], observed=row.observed[:, :2], boxes_xys=row.boxes_xys[:, :2])
    elif mismatch == "size":
        row = replace(row, size=(641, 480))
    else:
        row = replace(row, boxes_xys=None)
    with pytest.raises(ValueError, match="camera/timeline mismatch|source boxes"):
        gather_people(source(), {"pose_west": row, "pose_east": pose("east", [88, 6])})


@pytest.mark.parametrize("mismatch", ["cameras", "track_order", "frames"])
def test_identified_people_rejects_incompatible_identity_axes(mismatch: str) -> None:
    raw = gather_people(source(), {"pose_west": pose("west", [41]), "pose_east": pose("east", [88, 6])})
    tracks = raw.local_track_ids.copy()
    cameras = raw.camera_ids
    players: NDArray[np.int64] = np.full((2, 2, 3), -1, np.int64)
    if mismatch == "cameras":
        cameras = cameras[::-1]
    elif mismatch == "track_order":
        tracks[1] = tracks[1, ::-1]
    else:
        players = players[:, :, :2]
    identities = PlayerIdentitiesOutput(cameras, tracks, players, {})
    with pytest.raises(ValueError, match={"cameras": "cameras", "track_order": "track order", "frames": "timeline"}[mismatch]):
        identified_people(raw, identities, .3)


@pytest.mark.parametrize("mismatch", ["camera", "frame_order", "frame_count"])
def test_gather_balls_rejects_camera_and_explicit_frame_mismatch(mismatch: str) -> None:
    artifacts = {}
    for camera in source().camera_ids:
        count = 2 if camera == "west" and mismatch == "frame_count" else 3
        detections = BallDetectionOutput(camera, np.arange(count, dtype=np.int64), np.full((count, 2), 12., np.float32),
            np.full(count, .8, np.float32), np.ones(count, bool), np.ones(count, np.uint8), "annotation_acceptance_not_probability", None)
        row = BallPointsModule(TemporalSmoothingConfig(enabled=False)).process(BallPointsInput(detections, (640, 480), 30.))
        artifacts[f"ball_{camera}"] = row
    west = artifacts["ball_west"]
    if mismatch == "camera":
        artifacts["ball_west"] = replace(west, camera_id="east")
    elif mismatch == "frame_order":
        # Simulate an inconsistent deserialized artifact with the right length.
        west.frame_indices[:] = [0, 2, 1]
    with pytest.raises(ValueError, match="camera/timeline mismatch"):
        gather_balls(source(), artifacts)

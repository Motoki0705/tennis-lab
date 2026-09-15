"""The entity binary codec must be exact and reject malformed buffers."""

from __future__ import annotations

import numpy as np
import pytest

from src.tasks.base.visualization.review.payload import (
    ScenePayload,
    pack_entity_frames,
)


def test_layout_is_frames_then_orientation_then_presence() -> None:
    frames: np.ndarray = np.arange(2 * 3 * 1 * 3, dtype=np.float32).reshape(2, 3, 1, 3)
    orientation: np.ndarray = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    presence = np.asarray([[True, False, True], [False, True, True]])
    data = pack_entity_frames(frames, orientation=orientation, presence=presence)
    frames_bytes = 2 * 3 * 1 * 3 * 4
    orientation_bytes = 2 * 3 * 2 * 4
    assert len(data) == frames_bytes + orientation_bytes + 2 * 3
    decoded = np.frombuffer(data, dtype="<f4", count=2 * 3 * 1 * 3).reshape(frames.shape)
    assert np.array_equal(decoded, frames)
    decoded_orientation = np.frombuffer(
        data, dtype="<f4", count=2 * 3 * 2, offset=frames_bytes
    ).reshape(orientation.shape)
    assert np.array_equal(decoded_orientation, orientation)
    decoded_presence = np.frombuffer(
        data, dtype=np.uint8, count=6, offset=frames_bytes + orientation_bytes
    ).reshape(2, 3)
    assert np.array_equal(decoded_presence.astype(bool), presence)


def test_non_float32_frames_are_rejected() -> None:
    frames: np.ndarray = np.zeros((1, 2, 1, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        pack_entity_frames(frames)


def test_nan_frames_are_rejected() -> None:
    frames: np.ndarray = np.zeros((1, 2, 1, 3), dtype=np.float32)
    frames[0, 1, 0, 0] = np.nan
    with pytest.raises(ValueError):
        pack_entity_frames(frames)


def test_mismatched_presence_shape_is_rejected() -> None:
    frames: np.ndarray = np.zeros((2, 3, 1, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        pack_entity_frames(frames, presence=np.ones((3, 2), dtype=bool))


def test_scene_payload_json_bytes_are_compact_utf8() -> None:
    payload = ScenePayload(document={"task": "blcs", "name": "シーン"}, binary=b"")
    encoded = payload.json_bytes()
    assert encoded.decode("utf-8") == '{"task":"blcs","name":"シーン"}'

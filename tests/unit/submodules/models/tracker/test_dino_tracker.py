"""Contract tests for DINO association and completed person tracks."""

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

import src.submodules.models.tracker.common as tracker_common
from src.submodules.models import PersonDetectionResult
from src.submodules.models.tracker.common import (
    TrackRequest,
    select_and_complete_tracks,
)
from src.submodules.models.tracker.dino_tracker import BotSortAssociator


def _detections(x_offset: float) -> PersonDetectionResult:
    return PersonDetectionResult(
        boxes_xyxy=np.array(
            [[10 + x_offset, 20, 50 + x_offset, 100]], dtype=np.float32
        ),
        scores=np.array([0.95], dtype=np.float32),
    )


class _FakeBotSortBackend:
    def __init__(self, track_id: int) -> None:
        self.track_id = track_id
        self.inputs: list[NDArray[np.float32]] = []

    def update(self, boxes, _frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        box_data = np.asarray(boxes.data, dtype=np.float32).copy()
        self.inputs.append(box_data)
        if box_data.shape[0] == 0:
            return np.empty((0, 8), dtype=np.float32)
        return np.array(
            [[*box_data[0, :4], float(self.track_id), box_data[0, 4], 0.0, 0.0]],
            dtype=np.float32,
        )


def _fake_associator(backend: _FakeBotSortBackend) -> BotSortAssociator:
    associator = BotSortAssociator.__new__(BotSortAssociator)
    associator._tracker = backend
    return associator


def test_bot_sort_exposes_integer_ids_and_pixel_boxes_per_local_run() -> None:
    frame: NDArray[np.uint8] = np.zeros((120, 160, 3), dtype=np.uint8)
    near_backend = _FakeBotSortBackend(track_id=3)
    far_backend = _FakeBotSortBackend(track_id=3)
    near_camera = _fake_associator(near_backend)
    far_camera = _fake_associator(far_backend)

    near = near_camera.update(_detections(0), frame)
    far = far_camera.update(_detections(40), frame)

    assert near[0]["id"] == far[0]["id"] == 3
    assert type(near[0]["id"]) is int
    assert near[0]["bbx_xyxy"].dtype == np.float32
    np.testing.assert_array_equal(near[0]["bbx_xyxy"], [10, 20, 50, 100])
    np.testing.assert_array_equal(far[0]["bbx_xyxy"], [50, 20, 90, 100])
    np.testing.assert_array_equal(
        near_backend.inputs[0],
        np.array([[10, 20, 50, 100, 0.95, 0.0]], dtype=np.float32),
    )


def test_completed_dino_track_interpolates_then_smooths_missing_frames(
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

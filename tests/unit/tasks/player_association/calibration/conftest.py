"""Small three-camera scenes; no dataset, labels, checkpoints or GPU."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from src.tasks.player_association.appearance.sampling import TrackAppearance
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.calibration.samples import CalibrationClip
from src.utils.geometry.triangulation import PinholeCamera


@pytest.fixture
def scene() -> Callable[..., CalibrationClip]:
    def make(key: str = 'video_000/clip_010', frames: int = 24) -> CalibrationClip:
        cameras = []
        for camera in range(3):
            rotation = np.diag([1., -1., -1.])
            model = PinholeCamera(f'cam{camera}', np.array([[600., 0., 960.], [0., 600., 540.], [0., 0., 1.]]),
                                  rotation, -rotation @ np.array([0., 0., 30.]))
            # The camera-dependent x offset supplies a small nonzero geometric residual.
            ground = np.array([[.4 * camera, -12., 0.], [.4 * camera, 12., 0.]])
            pixels = model.project(ground)[0]
            boxes = np.repeat(np.column_stack((pixels[:, 0] - 20, pixels[:, 1] - 100,
                                               pixels[:, 0] + 20, pixels[:, 1]))[:, None], frames, axis=1)
            appearance = tuple(TrackAppearance(np.arange(frames, dtype=np.int64),
                np.repeat(np.eye(2, dtype=np.float32)[row:row + 1], frames, axis=0)) for row in range(2))
            cameras.append(CameraTracks(model, (1920, 1080), np.array([1, 2], np.int64), boxes,
                                        np.ones((2, frames), bool), appearance))
        return CalibrationClip(key, 2., tuple(cameras), tuple(np.zeros_like(c.observed) for c in cameras))
    return make

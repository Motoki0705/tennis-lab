from __future__ import annotations

import numpy as np

from src.tasks.player_association.appearance.sampling import (
    CropSamplingConfig,
    crop,
    sample_tracks,
)

SIZE = (200, 100)  # width, height


def _boxes() -> tuple[np.ndarray, np.ndarray]:
    """Track 0: clean for 10 frames. Track 1: small (0-2), at the border (3-5), overlapping track 0 (6-9)."""
    boxes: np.ndarray = np.zeros((2, 10, 4), np.float32)
    boxes[0] = (20, 10, 40, 80)
    boxes[1, :3] = (100, 10, 110, 30)
    boxes[1, 3:6] = (180, 10, 200, 80)
    boxes[1, 6:] = (22, 12, 42, 82)
    return boxes, np.ones((2, 10), bool)


def test_rejections_are_counted_by_reason_and_samples_are_even() -> None:
    boxes, observed = _boxes()
    config = CropSamplingConfig(min_height_px=40, border_px=4, max_overlap_iou=.1, max_samples=4)
    clean, other = sample_tracks(boxes, observed, SIZE, config)
    # Track 0 overlaps track 1 in frames 6-9, so only 0-5 remain.
    assert clean.frames.tolist() == [0, 2, 3, 5] and clean.rejected == {"overlapped": 4}
    assert other.frames.tolist() == [] and other.rejected == {"small": 3, "truncated": 3, "overlapped": 4}


def test_unobserved_frames_are_never_sampled() -> None:
    boxes, observed = _boxes()
    observed[0, :8] = False
    observed[1] = False
    (track, _) = sample_tracks(boxes, observed, SIZE, CropSamplingConfig(min_height_px=40))
    assert track.frames.tolist() == [8, 9] and track.rejected == {}


def test_crop_is_rgb_chw_in_unit_range() -> None:
    frame: np.ndarray = np.zeros((100, 200, 3), np.uint8)
    frame[10:80, 20:40] = (255, 0, 0)  # blue in BGR
    image = crop(frame, np.array([20, 10, 40, 80.0]), (64, 32))
    assert image.shape == (3, 64, 32) and image.dtype == np.float32
    np.testing.assert_allclose(image[:, 32, 16], [0, 0, 1])

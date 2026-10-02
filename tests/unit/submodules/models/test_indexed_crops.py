"""Noncontiguous source frame crops remain bounded and correctly indexed."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.submodules.models._base.crops import iter_frame_person_crops, iter_person_crops
from src.submodules.models.tracker.common import (
    TrackRequest,
    select_and_complete_tracks,
)
from src.submodules.vendor.gvhmr.hmr2.preproc import IMAGE_MEAN, IMAGE_STD
from src.utils.video import FramePacket


def test_indexed_crops_use_requested_frames_and_bound_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    decoded = []
    def frames(*args: Any, **kwargs: Any) -> Any:
        for index in range(5):
            decoded.append(index)
            yield FramePacket(index, np.full((50, 50, 3), 20 * index, np.uint8), (50, 50))
    monkeypatch.setattr("src.submodules.models._base.crops.OpenCVVideoFrameReader", frames)
    batches = list(iter_person_crops(Path("video.mp4"), torch.tensor([[25., 25., 20.]] * 3), torch.tensor([1, 1, 3]), batch_size=2))
    assert [len(images) for images, _ in batches] == [2, 1]
    images = torch.cat([images for images, _ in batches]).permute(0, 2, 3, 1)
    pixels = ((images * IMAGE_STD + IMAGE_MEAN) * 255).mean((1, 2, 3))
    torch.testing.assert_close(pixels, torch.tensor([20., 20., 60.]))
    assert decoded == [0, 1, 2, 3]


def test_all_tracks_mode_preserves_empty_observations_without_ui() -> None:
    result = select_and_complete_tracks([[], [], []], TrackRequest("empty.mp4", None, False), 3)
    assert result.track_ids == [] and result.num_frames == 3
    with pytest.raises(ValueError, match="interactive"):
        select_and_complete_tracks([[]], TrackRequest("empty.mp4", None, True), 1)


def test_frame_stream_matches_video_crops_and_preserves_colour(monkeypatch: pytest.MonkeyPatch) -> None:
    image: np.ndarray = np.zeros((48, 64, 3), np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = 20, 60, 200
    packets = [FramePacket(i, image, (64, 48)) for i in range(4)]
    monkeypatch.setattr("src.submodules.models._base.crops.OpenCVVideoFrameReader", lambda *a, **k: iter(packets))
    indices = torch.tensor([1, 1, 3])
    boxes = torch.tensor([[28., 24., 20.], [32., 24., 24.], [32., 20., 16.]])
    video = list(iter_person_crops("unused.mp4", boxes, indices, batch_size=2))
    stream = list(iter_frame_person_crops(((p.index, p.frame) for p in packets), boxes, indices, batch_size=2))
    for (image_a, box_a), (image_b, box_b) in zip(video, stream, strict=True):
        torch.testing.assert_close(image_a, image_b, rtol=0, atol=0)
        torch.testing.assert_close(box_a, box_b, rtol=0, atol=0)
        rgb = ((image_b.permute(0, 2, 3, 1) * IMAGE_STD + IMAGE_MEAN) * 255).mean((0, 1, 2))
        torch.testing.assert_close(rgb, torch.tensor([200., 60., 20.]))


@pytest.mark.parametrize("failure", ["missing", "short", "duplicate", "negative", "order", "dtype", "size", "channels"])
def test_frame_stream_rejects_missing_or_inconsistent_images(failure: str) -> None:
    frames: list[tuple[int, np.ndarray]] = [(i, np.zeros((48, 64, 3), np.uint8)) for i in range(4)]
    indices = torch.tensor([1, 3])
    if failure == "missing":
        frames.pop(1)
    elif failure == "short":
        frames.pop()
    elif failure == "duplicate":
        frames[1] = frames[0]
    elif failure == "negative":
        frames[0] = (-1, frames[0][1])
    elif failure == "order":
        frames[1], frames[2] = frames[2], frames[1]
    elif failure == "dtype":
        frames[1] = (1, np.zeros((48, 64, 3), np.float32))
    elif failure == "size":
        frames[1] = (1, np.zeros((49, 64, 3), np.uint8))
    else:
        frames[1] = (1, np.zeros((48, 64, 4), np.uint8))
    with pytest.raises(ValueError):
        list(iter_frame_person_crops(frames, torch.tensor([[32., 24., 20.]] * 2), indices, batch_size=2))


def test_empty_crop_request_does_not_consume_source() -> None:
    class Unreadable:
        def __iter__(self):
            raise AssertionError("Empty request consumed its source")
    assert list(iter_frame_person_crops(Unreadable(), torch.empty(0, 3), torch.empty(0, dtype=torch.int64), batch_size=2)) == []

"""Noncontiguous source frame crops remain bounded and correctly indexed."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.submodules.models._base.crops import iter_person_crops
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

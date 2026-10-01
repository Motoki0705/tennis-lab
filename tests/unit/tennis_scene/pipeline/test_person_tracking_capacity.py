"""The per-camera cumulative track cap of person_tracking is configurable and stops explicitly."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.tennis_scene.pipeline.components import person_tracking
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_tracking import (
    PersonTrackingInput,
    PersonTrackingModule,
)
from src.tennis_scene.pipeline.components.tracking_identity import TrackletLinkPolicy
from src.tennis_scene.pipeline.contracts import SourceVideo
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable

FRAMES = 6
PEOPLE = 3


class _Tracker:
    """Reports ``PEOPLE`` well separated, never-moving people with fixed IDs."""

    def update(self, detection: Any, frame: Any) -> list[dict[str, Any]]:
        return [{"id": i + 1, "bbx_xyxy": np.array([100. + 300 * i, 100., 200. + 300 * i, 400.], np.float32)} for i in range(PEOPLE)]


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PersonTrackingInput:
    frame: np.ndarray = np.full((720, 1280, 3), 128, np.uint8)
    monkeypatch.setattr(person_tracking, "BotSortAssociator", _Tracker)
    monkeypatch.setattr(person_tracking, "OpenCVVideoFrameReader",
                        lambda path, max_frames: [SimpleNamespace(index=i, frame=frame) for i in range(FRAMES)])
    boxes = np.tile(np.array([[100., 100., 200., 400.]], np.float32), (FRAMES * PEOPLE, 1))
    detections = PersonDetectionOutput("cam0", np.arange(FRAMES + 1, dtype=np.int64) * PEOPLE, boxes, np.ones(FRAMES * PEOPLE, np.float32))
    video = SourceVideo("cam0", tmp_path / "cam0.mp4", "0" * 64, FRAMES, 30., 1280, 720)
    return PersonTrackingInput(video, detections)


def test_tracks_within_the_cap_are_returned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = PersonTrackingModule(TrackletLinkPolicy(), max_tracks=PEOPLE).process(_inputs(tmp_path, monkeypatch))
    assert output.track_ids.tolist() == [1, 2, 3]
    assert output.observed.all()


def test_tracks_over_the_cap_stop_with_their_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ReconstructionUnavailable) as stopped:
        PersonTrackingModule(TrackletLinkPolicy(), max_tracks=PEOPLE - 1).process(_inputs(tmp_path, monkeypatch))
    assert stopped.value.reason == "person_capacity_exceeded"
    assert stopped.value.diagnostics == {"track_ids": [1, 2, 3], "observed_frames": [FRAMES] * PEOPLE}


def test_cap_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        PersonTrackingModule(TrackletLinkPolicy(), max_tracks=0)

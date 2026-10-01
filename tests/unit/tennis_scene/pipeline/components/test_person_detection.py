"""Person detection is always restricted to the calibrated court ROI."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tennis_scene.pipeline.components.person_detection import (
    PersonDetectionInput,
    PersonDetectionModule,
)
from src.tennis_scene.pipeline.contracts import SourceVideo
from tests.unit.tennis_scene.pipeline.config_factories import make_people_config


def test_camera_without_court_roi_is_not_detected_unfiltered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("an excluded camera must not load the detector")

    monkeypatch.setattr("src.tennis_scene.pipeline.components.person_detection.DinoPersonDetector", forbidden)
    video = SourceVideo("cam2", tmp_path / "cam2.mp4", "hash", 5, 30.0, 1280, 720)
    output = PersonDetectionModule(make_people_config(tmp_path)).process(PersonDetectionInput(video, None))
    assert output.camera_id == "cam2"
    assert output.frame_offsets.tolist() == [0] * 6
    assert output.boxes_xyxy.shape == (0, 4)

"""Progress must describe consumed camera frames without altering tracking."""

from __future__ import annotations

import io
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tasks.person_tracking import sequence
from src.tasks.person_tracking.sequence import TrackingConfig
from src.tennis_scene.pipeline.components import person_tracking as module
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.contracts import SourceVideo


class _TrackingHarness:
    """Real CPU tracker on saved features; progress no longer depends on decoding."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.clock = 0.0
        self.fail_at: int | None = None
        self.tmp_path = tmp_path
        self.video = SourceVideo('camera-east', tmp_path / 'unused.mp4', 'hash', 12, 30., 64, 64)
        present = [i not in {2, 6} for i in range(self.video.num_frames)]
        offsets = np.array([0, *np.cumsum(present)], dtype=np.int64)
        boxes = np.array([[10 + i % 2, 10, 30 + i % 2, 45] for i, found in enumerate(present) if found], np.float32)
        rows: np.ndarray = np.arange(len(boxes), dtype=np.int64) * 3 + 7
        self.detections = PersonDetectionOutput(self.video.camera_id, offsets, boxes, np.full(len(boxes), .9, np.float32), rows)

        class Link:
            def __init__(link_self, path: Path) -> None:
                pass

            def links(link_self, boxes: np.ndarray, observed: np.ndarray) -> tuple[dict[int, int], list[Any]]:
                self.events.append(('link',))
                return {i: i for i in range(len(boxes))}, []

        original_update = sequence.StrongSort.update

        def update(tracker: Any, features: Any) -> Any:
            self.events.append(('track', features.frame, features.rows.tolist()))
            if features.frame == self.fail_at:
                raise RuntimeError('tracker failed')
            result = original_update(tracker, features)
            self.clock += 6
            return result

        monkeypatch.setattr(module, 'AFLink', Link)
        monkeypatch.setattr(module, 'monotonic', lambda: self.clock)
        monkeypatch.setattr(sequence.StrongSort, 'update', update)

    def run(self, *, show_progress: bool = True) -> module.PersonTrackingOutput:
        from src.tennis_scene.pipeline.components.person_features import (
            PersonAppearanceFeaturesOutput,
            PersonPoseFeaturesOutput,
        )
        detections = self.detections
        n = len(detections.confidence)
        poses: np.ndarray = np.zeros((n, 17, 3), np.float32)
        poses[..., :2] = (detections.boxes_xyxy[:, None, :2] + detections.boxes_xyxy[:, None, 2:]) * .5
        poses[..., 2] = .8
        embeddings: np.ndarray = np.zeros((n, 1280), np.float32)
        embeddings[:, 0] = 1
        pose_features = PersonPoseFeaturesOutput(self.video.camera_id, detections.frame_offsets, detections.source_rows, poses)
        appearance = PersonAppearanceFeaturesOutput(self.video.camera_id, detections.frame_offsets, detections.source_rows,
            embeddings, np.ones(n, bool), TrackingConfig().encoder, "a" * 64)
        tracker = module.PersonTrackingModule(TrackingConfig(), aflink_checkpoint=self.tmp_path / 'af.pth', show_progress=show_progress)
        return tracker.process(module.PersonTrackingInput(self.video, detections, pose_features, appearance))



def test_redirected_progress_counts_consumed_frames_and_preserves_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    harness = _TrackingHarness(tmp_path, monkeypatch)
    displayed = harness.run()
    output = capsys.readouterr().err
    events = harness.events.copy()
    assert output.splitlines() == [
        'person_tracking/camera-east: 0/12 frames',
        'person_tracking/camera-east: 5/12 frames',
        'person_tracking/camera-east: 10/12 frames',
        'person_tracking/camera-east: 12/12 frames',
    ]
    assert '\r' not in output and 'ViTPose:' not in output
    assert [event[1] for event in events if event[0] == 'track'] == list(range(12))
    assert harness.detections.source_rows is not None
    assert [event[2] for event in events if event[0] == 'track'] == [
        harness.detections.source_rows[harness.detections.frame_offsets[i]:harness.detections.frame_offsets[i + 1]].tolist()
        for i in range(12)
    ]
    assert displayed.evidence is not None and displayed.observed.any()
    harness.events.clear()
    harness.clock = 0
    hidden = harness.run(show_progress=False)
    assert capsys.readouterr().err == ''
    assert harness.events == events
    np.testing.assert_equal(asdict(displayed), asdict(hidden))


def test_interrupted_progress_never_claims_unprocessed_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    harness = _TrackingHarness(tmp_path, monkeypatch)
    harness.fail_at = 5
    expected, error = 5, "tracker failed"
    with pytest.raises((ValueError, RuntimeError), match=error):
        harness.run()
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == 'person_tracking/camera-east: 0/12 frames'
    assert lines[-1] == f'person_tracking/camera-east: {expected}/12 frames'
    assert not any('12/12' in line for line in lines)


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_terminal_has_one_camera_bar_with_final_frame_total(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _TrackingHarness(tmp_path, monkeypatch)
    terminal = _Terminal()
    with monkeypatch.context() as context:
        context.setattr(sys, 'stderr', terminal)
        harness.run()
    assert 'person_tracking/camera-east:' in terminal.getvalue()
    assert '12/12' in terminal.getvalue()
    assert '\r' in terminal.getvalue() and terminal.getvalue().endswith('\n')
    assert 'ViTPose:' not in terminal.getvalue()

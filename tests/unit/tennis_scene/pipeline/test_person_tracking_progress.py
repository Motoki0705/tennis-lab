"""Progress must describe consumed camera frames without altering tracking."""

from __future__ import annotations

import io
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.submodules.models.vitpose import pose2d as vitpose
from src.tasks.person_tracking import sequence
from src.tasks.person_tracking.sequence import TrackingConfig
from src.tennis_scene.pipeline.components import person_tracking as module
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.contracts import SourceVideo
from tests.unit.tennis_scene.pipeline.config_factories import make_people_config


class _TrackingHarness:
    """Real feature extraction/ViTPose wrapper/tracker, with deterministic CPU boundaries."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.clock = 0.0
        self.read_limit = 12
        self.fail_at: int | None = None
        self.pose_progress: list[bool] = []
        self.tmp_path = tmp_path
        self.people = make_people_config(tmp_path)
        self.video = SourceVideo('camera-east', tmp_path / 'unused.mp4', 'hash', 12, 30., 64, 64)
        present = [i not in {2, 6} for i in range(self.video.num_frames)]
        offsets = np.array([0, *np.cumsum(present)], dtype=np.int64)
        boxes = np.array([[10 + i % 2, 10, 30 + i % 2, 45] for i, found in enumerate(present) if found], np.float32)
        rows: np.ndarray = np.arange(len(boxes), dtype=np.int64) * 3 + 7
        self.detections = PersonDetectionOutput(self.video.camera_id, offsets, boxes, np.full(len(boxes), .9, np.float32), rows)

        class Network(torch.nn.Module):
            def forward(network_self, images: torch.Tensor) -> torch.Tensor:
                self.events.append(('pose_forward', len(images)))
                return torch.zeros((len(images), 17, 1, 1))

        class Encoder:
            name = 'clipreid_vitb16_market1501'
            input_size = (8, 8)

            def embed(encoder_self, crops: torch.Tensor) -> torch.Tensor:
                self.events.append(('embed', len(crops)))
                result = torch.zeros((len(crops), 1280))
                result[:, 0] = 1
                return result

        class Link:
            def __init__(link_self, path: Path) -> None:
                pass

            def links(link_self, boxes: np.ndarray, observed: np.ndarray) -> tuple[dict[int, int], list[Any]]:
                self.events.append(('link',))
                return {i: i for i in range(len(boxes))}, []

        real_pose = module.ViTPosePose2D

        def pose_factory(*args: Any, **kwargs: Any) -> vitpose.ViTPosePose2D:
            self.pose_progress.append(kwargs['show_progress'])
            model: vitpose.ViTPosePose2D = real_pose(*args, **kwargs)
            model._pose, model._loaded = Network(), True
            return model

        def crops(frames: Any, boxes: torch.Tensor, indices: torch.Tensor, *, batch_size: int) -> Any:
            frame, image = next(iter(frames))
            self.events.append(('pose_request', frame, indices.tolist(), int(image[0, 0, 0])))
            yield torch.zeros((len(boxes), 3, 1, 256)), boxes

        def decode(*, heatmaps: np.ndarray, center: np.ndarray, scale: np.ndarray, use_udp: bool) -> tuple[np.ndarray, np.ndarray]:
            return np.repeat(center[:, None, :], 17, axis=1), np.full((len(center), 17, 1), .8, np.float32)

        def read(*args: Any, **kwargs: Any) -> Any:
            assert kwargs['max_frames'] == self.video.num_frames
            for i in range(self.read_limit):
                self.events.append(('decode', i))
                yield SimpleNamespace(index=i, frame=np.full((64, 64, 3), i, np.uint8))

        original_update = sequence.StrongSort.update

        def update(tracker: Any, features: Any) -> Any:
            self.events.append(('track', features.frame, features.rows.tolist()))
            if features.frame == self.fail_at:
                raise RuntimeError('tracker failed')
            result = original_update(tracker, features)
            self.clock += 6
            return result

        monkeypatch.setattr(module, 'ViTPosePose2D', pose_factory)
        monkeypatch.setattr(module, 'AFLink', Link)
        monkeypatch.setattr(module, 'OpenCVVideoFrameReader', read)
        monkeypatch.setattr(module, 'release_inference_memory', lambda device: None)
        monkeypatch.setattr(module, 'monotonic', lambda: self.clock)
        monkeypatch.setattr(vitpose, 'iter_frame_person_crops', crops)
        monkeypatch.setattr(vitpose, 'keypoints_from_heatmaps', decode)
        monkeypatch.setattr(sequence.StrongSort, 'update', update)
        self.encoder = Encoder

    def run(self, *, show_progress: bool = True) -> module.PersonTrackingOutput:
        tracker = module.PersonTrackingModule(TrackingConfig(), people=self.people, encoder=self.encoder,
            aflink_checkpoint=self.tmp_path / 'af.pth', show_progress=show_progress)
        return tracker.process(module.PersonTrackingInput(self.video, self.detections))


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
    assert [event[1] for event in events if event[0] == 'pose_request'] == [0, 1, 3, 4, 5, 7, 8, 9, 10, 11]
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
    assert harness.pose_progress == [False, False]
    np.testing.assert_equal(asdict(displayed), asdict(hidden))


@pytest.mark.parametrize(('read_limit', 'fail_at', 'expected', 'error'), [
    (7, None, 7, 'complete source timeline'),
    (12, 5, 5, 'tracker failed'),
])
def test_interrupted_progress_never_claims_unprocessed_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    read_limit: int, fail_at: int | None, expected: int, error: str,
) -> None:
    harness = _TrackingHarness(tmp_path, monkeypatch)
    harness.read_limit, harness.fail_at = read_limit, fail_at
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

"""SLCS rendering streams frames and publishes only complete videos."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.tasks.slcs.data.dataset import ClipArrays
from src.tasks.slcs.visualization import _video, overlay_2d
from src.tasks.slcs.visualization.renderer_3d import (
    SceneRenderInputs,
    SLCSSceneRenderer,
)
from src.utils.video.reader import probe_video_info, read_video_rgb
from src.utils.video.types import FramePacket
from src.utils.video.writer import VideoWriter, save_video_rgb


def _scene(n: int = 3) -> SceneRenderInputs:
    return SceneRenderInputs(
        player_position_m=np.zeros((1, n, 3), np.float32),
        player_yaw_rad=np.zeros((1, n), np.float32),
        ball_position_m=np.zeros((n, 3), np.float32),
        gt_player_position_m=None,
        gt_player_yaw_rad=None,
        gt_ball_position_m=None,
        gt_player_valid=None,
        gt_ball_valid=None,
    )


def _clip(path: Path, n: int = 3) -> ClipArrays:
    # Only the public rendering inputs are needed; no label/model fixtures.
    return cast(
        ClipArrays,
        SimpleNamespace(
            manifest=SimpleNamespace(
                camera_ids=("cam0",),
                media_path=lambda _: path,
                width=96,
                height=64,
                clip_id="clip",
            ),
            fps=12.0,
            num_frames=n,
            court_kp=np.zeros((1, n, 14, 2), np.float32),
            court_vis=np.zeros((1, n, 14), np.float32),
            human_kp_2d=np.zeros((1, 1, n, 17, 2), np.float32),
            human_kp_vis=np.zeros((1, 1, n, 17), np.float32),
            ball_uv=np.zeros((1, n, 2), np.float32),
            ball_vis=np.zeros((1, n), bool),
        ),
    )


def _overlay(clip: ClipArrays, output: Path, n: int | None = None) -> tuple[Path, int]:
    scene = _scene(clip.num_frames if n is None else n)
    return overlay_2d.render_overlay_video(
        clip,
        0,
        player_position_m=scene.player_position_m,
        player_yaw_rad=scene.player_yaw_rad,
        ball_position_m=scene.ball_position_m,
        output_path=output,
        court_kp_indices=tuple(range(14)),
        min_homography_points=4,
        court_visibility_threshold=0.5,
    )


def _frames(n: int = 3) -> np.ndarray:
    frames: np.ndarray = np.zeros((n, 64, 96, 3), np.uint8)
    frames[..., 0] = np.arange(n)[:, None, None] * 35 + 60
    frames[..., 1] = 110
    return frames


def test_real_overlay_preserves_count_size_fps_rgb_and_missing_homographies(
    tmp_path: Path,
) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "overlay.mp4"
    frames = _frames(4)
    save_video_rgb(frames, source, fps=12)
    # An extra decoded source frame remains ignored as before.
    assert _overlay(_clip(source, 3), output) == (output, 3)
    info = probe_video_info(output)
    assert (info.frame_count, info.width, info.height, info.fps) == (3, 96, 64, 12)
    decoded = read_video_rgb(output)
    # Top of the image excludes the explicit homography warning at the bottom.
    assert np.abs(decoded[:, :20].astype(int) - frames[:3, :20].astype(int)).mean() < 8
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "overlay.mp4",
        "source.mp4",
    ]


def test_scene_real_video_matches_rendered_rgb(tmp_path: Path) -> None:
    renderer = SLCSSceneRenderer(figsize=(6, 3), dpi=80)
    scene = _scene(1)
    expected = renderer.render_frame(scene, 0)
    output = renderer.render_video(scene, tmp_path / "scene.mp4", fps=24, frame_step=1)
    decoded = read_video_rgb(output)
    assert decoded.shape == (1, *expected.shape)
    assert probe_video_info(output).fps == 24
    assert np.abs(decoded[0].astype(int) - expected.astype(int)).mean() < 10


def test_scene_frame_step_streaming_and_successful_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = SLCSSceneRenderer(figsize=(4, 2), dpi=80)
    frames = _frames(5)
    events: list[tuple[str, int]] = []
    original_write = VideoWriter.write_frame

    def render(inputs: SceneRenderInputs, t: int) -> np.ndarray:
        events.append(("render", t))
        return frames[t]

    def write(self: VideoWriter, frame: np.ndarray) -> None:
        events.append(("write", int((int(frame[0, 0, 0]) - 60) // 35)))
        original_write(self, frame)

    monkeypatch.setattr(renderer, "render_frame", render)
    monkeypatch.setattr(VideoWriter, "write_frame", write)
    output = tmp_path / "scene.mp4"
    output.write_bytes(b"previous")
    assert renderer.render_video(_scene(5), output, fps=24, frame_step=2) == output
    assert events == [(kind, t) for t in (0, 2, 4) for kind in ("render", "write")]
    info = probe_video_info(output)
    assert (info.frame_count, info.width, info.height, info.fps) == (3, 96, 64, 12)
    assert (
        np.abs(read_video_rgb(output).astype(int) - frames[::2].astype(int)).mean() < 8
    )


def test_overlay_writes_before_reading_next_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, int]] = []
    frames = _frames()
    original_write = VideoWriter.write_frame

    def reader(path: Path) -> Iterator[FramePacket[np.ndarray]]:
        for t, rgb in enumerate(frames):
            events.append(("read", t))
            yield FramePacket(
                index=t, frame=rgb[..., ::-1].copy(), original_size=(96, 64)
            )

    def write(self: VideoWriter, frame: np.ndarray) -> None:
        events.append(("write", int((int(frame[0, 0, 0]) - 60) // 35)))
        original_write(self, frame)

    monkeypatch.setattr(overlay_2d, "OpenCVVideoFrameReader", reader)
    monkeypatch.setattr(VideoWriter, "write_frame", write)
    _overlay(_clip(tmp_path / "unused"), tmp_path / "out.mp4")
    assert events == [(kind, t) for t in range(3) for kind in ("read", "write")]


@pytest.mark.parametrize("failure", ["timeline", "decode", "empty"])
def test_overlay_invalid_timeline_preserves_existing_output(
    tmp_path: Path, failure: str
) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "out.mp4"
    save_video_rgb(_frames(2), source)
    output.write_bytes(b"previous")
    with pytest.raises(
        (ValueError, RuntimeError), match="timeline|decoded 2 frames|empty"
    ):
        _overlay(
            _clip(source, 0 if failure == "empty" else 3),
            output,
            n=2 if failure == "timeline" else None,
        )
    assert output.read_bytes() == b"previous"
    assert {path.name for path in tmp_path.iterdir()} == {"out.mp4", "source.mp4"}


@pytest.mark.parametrize("kind", ["overlay", "scene"])
@pytest.mark.parametrize("failure", ["draw", "write", "close"])
def test_streaming_failures_close_encoder_and_preserve_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, failure: str
) -> None:
    output, source = tmp_path / "out.mp4", tmp_path / "source.mp4"
    save_video_rgb(_frames(), source)
    output.write_bytes(b"previous")
    closed: list[bool] = []
    original_close, original_write = VideoWriter.close, VideoWriter.write_frame

    def close(self: VideoWriter) -> None:
        assert output.read_bytes() == b"previous"
        original_close(self)
        closed.append(True)
        if failure == "close":
            raise RuntimeError("close failed")

    def write(self: VideoWriter, frame: np.ndarray) -> None:
        original_write(self, frame)
        if failure == "write":
            raise RuntimeError("write failed")

    draws = 0

    def fail(*args: Any, **kwargs: Any) -> None:
        nonlocal draws
        draws += 1
        if draws == 2:
            raise RuntimeError("draw failed")

    monkeypatch.setattr(VideoWriter, "close", close)
    monkeypatch.setattr(VideoWriter, "write_frame", write)
    renderer = SLCSSceneRenderer(figsize=(4, 2), dpi=80)
    if failure == "draw":
        monkeypatch.setattr(overlay_2d, "_draw_observations", fail)
        monkeypatch.setattr(renderer, "_draw_3d", fail)
    figures = plt.get_fignums()
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        if kind == "overlay":
            _overlay(_clip(source), output)
        else:
            renderer.render_video(_scene(), output, fps=12, frame_step=1)
    assert closed == [True]
    assert plt.get_fignums() == figures
    assert output.read_bytes() == b"previous"
    assert {path.name for path in tmp_path.iterdir()} == {"out.mp4", "source.mp4"}


def test_atomic_writer_preserves_crf_and_publishes_after_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out.mp4"
    output.write_bytes(b"previous")
    original_close = VideoWriter.close

    def create(path: Path, *, fps: float, crf: int) -> VideoWriter:
        assert path.parent == output.parent and path.suffix == ".mp4" and path != output
        assert (fps, crf) == (12, 17)
        return VideoWriter(path, fps=fps, crf=crf)

    def close(self: VideoWriter) -> None:
        assert output.read_bytes() == b"previous"
        original_close(self)
        assert output.read_bytes() == b"previous"

    monkeypatch.setattr(_video, "VideoWriter", create)
    monkeypatch.setattr(VideoWriter, "close", close)
    with _video.atomic_video_writer(output, fps=12) as writer:
        writer.write_frame(_frames(1)[0])
    assert probe_video_info(output).frame_count == 1
    assert list(tmp_path.iterdir()) == [output]


def test_writer_initialization_failure_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("init failed")

    monkeypatch.setattr(_video, "VideoWriter", fail)
    output = tmp_path / "out.mp4"
    with (
        pytest.raises(RuntimeError, match="init failed"),
        _video.atomic_video_writer(output, fps=12),
    ):
        pytest.fail("writer initialization must fail")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("frame_step,n", [(0, 1), (1, 0)])
def test_scene_rejects_empty_and_invalid_step(
    tmp_path: Path, frame_step: int, n: int
) -> None:
    renderer = SLCSSceneRenderer(figsize=(4, 2), dpi=80)
    with pytest.raises(ValueError, match="frame_step|empty"):
        renderer.render_video(
            _scene(n), tmp_path / "out.mp4", fps=12, frame_step=frame_step
        )
    assert list(tmp_path.iterdir()) == []

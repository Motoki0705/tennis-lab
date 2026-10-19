"""Exercise the actual H.264/GIF writers and lossless still exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image
from PIL.GifImagePlugin import GifImageFile

from src.tasks.base.visualization.presentation.export import render_presentation

pytestmark = pytest.mark.unit


def _frame(index: int) -> NDArray[np.uint8]:
    return np.full((24, 32, 3), index * 60, dtype=np.uint8)


def _arguments(output_dir: Path) -> dict[str, Any]:
    return dict(
        frame_count=3,
        fps=3.0,
        output_dir=output_dir,
        name="smoke",
        gif_width=16,
        gif_fps=3.0,
        stills={"first": 0, "last": 2},
        manifest={"provenance": {"kind": "schematic"}, "config": {"fps": 3}},
    )


def test_three_frame_export_produces_decodable_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "render"
    calls: list[int] = []

    def draw(index: int) -> NDArray[np.uint8]:
        calls.append(index)
        return _frame(index)

    arguments = _arguments(output)
    render_presentation(draw, **arguments)
    assert calls == [0, 1, 2]
    assert sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    ) == [
        "manifest.json",
        "smoke.gif",
        "smoke.mp4",
        "stills/first.png",
        "stills/last.png",
    ]
    assert json.loads((output / "manifest.json").read_text()) == arguments["manifest"]
    with av.open(str(output / "smoke.mp4")) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert (stream.width, stream.height) == (32, 24)
        assert float(stream.average_rate) == 3
        frames = list(container.decode(stream))
        assert len(frames) == 3
        for index, frame in enumerate(frames):
            assert float(frame.to_ndarray(format="rgb24").mean()) == pytest.approx(
                index * 60,
                abs=3,
            )
    with Image.open(output / "smoke.gif") as gif:
        assert isinstance(gif, GifImageFile)
        assert gif.size == (16, 12)
        assert gif.n_frames == 3
    with Image.open(output / "stills/last.png") as still:
        np.testing.assert_array_equal(np.array(still), _frame(2))


@pytest.mark.parametrize("kind", ["directory", "file", "dangling_symlink"])
def test_existing_destination_is_never_modified(tmp_path: Path, kind: str) -> None:
    output = tmp_path / "exists"
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_text("keep me")
    else:
        output.symlink_to(tmp_path / "absent", target_is_directory=True)

    def never_draw(index: int) -> NDArray[np.uint8]:
        raise AssertionError("must reject existing output before drawing")

    with pytest.raises(FileExistsError):
        render_presentation(never_draw, **_arguments(output))
    if kind == "directory":
        assert list(output.iterdir()) == []
    elif kind == "file":
        assert output.read_text() == "keep me"
    else:
        assert output.is_symlink()


def test_gif_sampling_uses_video_timestamps_and_copies_reused_frames(
    tmp_path: Path,
) -> None:
    output = tmp_path / "sampled"
    buffer = _frame(0)

    def draw(index: int) -> NDArray[np.uint8]:
        buffer.fill(index * 30)
        return buffer

    arguments = _arguments(output)
    arguments.update(frame_count=7, fps=5.0, gif_fps=3.0, stills={"unsampled": 4})
    render_presentation(draw, **arguments)
    with Image.open(output / "smoke.gif") as gif:
        assert isinstance(gif, GifImageFile)
        assert gif.n_frames == 5
        for gif_index, source_index in enumerate([0, 1, 3, 5, 6]):
            gif.seek(gif_index)
            assert np.array(gif.convert("RGB")).mean() == source_index * 30
    with Image.open(output / "stills/unsampled.png") as still:
        assert np.array(still).mean() == 120


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"frame_count": 0}, ValueError),
        ({"frame_count": 1.5}, ValueError),
        ({"frame_count": True}, ValueError),
        ({"fps": 0}, ValueError),
        ({"fps": np.nan}, ValueError),
        ({"gif_fps": np.inf}, ValueError),
        ({"gif_fps": -1}, ValueError),
        ({"gif_width": 0}, ValueError),
        ({"gif_width": 1.5}, ValueError),
        ({"name": "../escape"}, ValueError),
        ({"stills": {"../escape": 0}}, ValueError),
        ({"stills": {"late": 3}}, ValueError),
        ({"stills": {"float": 0.5}}, TypeError),
        ({"manifest": {"metric": np.nan}}, ValueError),
        ({"manifest": {"path": Path("input")}}, TypeError),
    ],
)
def test_invalid_export_options_do_not_create_output(
    tmp_path: Path,
    overrides: dict[str, Any],
    error: type[Exception],
) -> None:
    output = tmp_path / "invalid"
    arguments = _arguments(output)
    arguments.update(overrides)
    with pytest.raises(error):
        render_presentation(_frame, **arguments)
    assert not output.exists()


@pytest.mark.parametrize(
    "frame,error",
    [
        (np.zeros((24, 32, 3), dtype=np.float32), TypeError),
        (np.zeros((24, 32), dtype=np.uint8), ValueError),
        (np.zeros((24, 32, 4), dtype=np.uint8), ValueError),
        (np.zeros((23, 32, 3), dtype=np.uint8), ValueError),
        (np.zeros((0, 32, 3), dtype=np.uint8), ValueError),
    ],
)
def test_frame_contract_is_not_silently_converted(
    tmp_path: Path,
    frame: Any,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        render_presentation(lambda _: frame, **_arguments(tmp_path / "invalid"))


def test_changed_frame_size_is_rejected_and_partial_output_is_preserved(
    tmp_path: Path,
) -> None:
    output = tmp_path / "partial"

    def draw(index: int) -> NDArray[np.uint8]:
        return np.zeros((24 + 2 * index, 32, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="size changed"):
        render_presentation(draw, **_arguments(output))
    assert output.is_dir()
    assert not (output / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        render_presentation(_frame, **_arguments(output))

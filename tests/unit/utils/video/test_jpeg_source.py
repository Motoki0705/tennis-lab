"""Video seeks must reproduce the sequential publication's exact JPEG bytes."""

from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from src.utils.data.verification import file_version
from src.utils.video.encoding import encode_jpeg
from src.utils.video.jpeg_source import jpeg_video_runtime, read_video_jpegs


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, tuple[bytes, ...], dict[str, Any]]:
    path = tmp_path / "vfr.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.time_base = Fraction(1, 90000)
        stream.codec_context.time_base = stream.time_base
        stream.options = {"crf": "18", "g": "12", "bf": "2"}
        rng = np.random.default_rng(42)
        pts = 0
        for i in range(140):
            frame = av.VideoFrame.from_ndarray(
                rng.integers(0, 256, (64, 96, 3), np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = pts, stream.time_base
            pts += 3000 if i % 3 else 4500
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    payloads, points = [], []
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        video.codec_context.thread_count = 2
        for frame in container.decode(video):
            points.append(frame.pts)
            payloads.append(encode_jpeg(frame.to_ndarray(format="bgr24"), quality=90))
        tb = Fraction(video.time_base)
    options = dict(
        pts=tuple(points),
        time_base=tb,
        width=96,
        height=64,
        quality=90,
        runtime=jpeg_video_runtime(),
        frame_hashes=tuple(hashlib.sha256(x).hexdigest() for x in payloads),
        version=file_version(path),
    )
    return path, tuple(payloads), options


@pytest.mark.parametrize("step", [1, 2, 4])
def test_vfr_seek_matches_sequential_jpeg_bytes(
    source: tuple[Path, tuple[bytes, ...], dict[str, Any]], step: int
) -> None:
    path, expected, options = source
    indices = tuple(5 + i * step for i in range(32))
    assert read_video_jpegs(path, indices, **options) == tuple(
        expected[i] for i in indices
    )


@pytest.mark.parametrize(
    "fault,match",
    [
        ("runtime", "runtime"),
        ("hash", "JPEG"),
        ("pts", "PTS"),
        ("geometry", "dimensions"),
        ("file", "changed"),
    ],
)
def test_video_reader_refuses_changed_inputs(
    source: tuple[Path, tuple[bytes, ...], dict[str, Any]], fault: str, match: str
) -> None:
    path, _, options = source
    if fault == "runtime":
        options["runtime"] = "0" * 64
    elif fault == "hash":
        options["frame_hashes"] = ("0" * 64,) * 140
    elif fault == "pts":
        options["pts"] = tuple(p + 1 for p in options["pts"])
    elif fault == "geometry":
        options["width"] = 95
    else:
        with path.open("ab") as output:
            output.write(b"changed")
    with pytest.raises(ValueError, match=match):
        read_video_jpegs(path, tuple(range(5, 37)), **options)


@pytest.mark.parametrize("indices", [(), (2, 1), (-1,), (140,), tuple(range(33))])
def test_video_reader_bounds_selection(
    source: tuple[Path, tuple[bytes, ...], dict[str, Any]], indices: tuple[int, ...]
) -> None:
    path, _, options = source
    with pytest.raises(ValueError, match="selection"):
        read_video_jpegs(path, indices, **options)

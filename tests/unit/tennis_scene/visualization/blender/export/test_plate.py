"""Plate intervals retain source indices, colors and aspect ratio."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.tennis_scene.visualization.blender.export.bundle import _write_plate
from src.utils.video.encoding import encode_jpeg
from src.utils.video.reader import OpenCVVideoFrameReader
from src.utils.video.writer import save_video_rgb


@pytest.fixture
def video(tmp_path: Path) -> Path:
    rgb: np.ndarray = np.zeros((6, 48, 64, 3), dtype=np.uint8)
    rgb[..., 1] = np.arange(6, dtype=np.uint8)[:, None, None] * 30
    path = tmp_path / "source.mp4"
    save_video_rgb(rgb, path, fps=30.0)
    return path


def test_partial_plate_preserves_source_indices_and_resizes(
    video: Path, tmp_path: Path
) -> None:
    output = tmp_path / "plates/cam1"
    _write_plate(video, output, frame_range=range(2, 5), max_width=16, quality=95)

    assert sorted(path.name for path in output.iterdir()) == [
        "00002.jpg",
        "00003.jpg",
        "00004.jpg",
    ]
    for index in range(2, 5):
        decoded = cv2.imread(str(output / f"{index:05d}.jpg"))
        assert decoded is not None
        assert decoded.shape == (12, 16, 3)
        np.testing.assert_allclose(
            decoded.mean(axis=(0, 1)), [0, index * 30, 0], atol=4
        )


def test_full_plate_matches_original_sequential_jpeg_bytes(
    video: Path, tmp_path: Path
) -> None:
    output = tmp_path / "plate"
    _write_plate(video, output, frame_range=range(6), max_width=None, quality=95)

    expected = {
        f"{packet.index:05d}.jpg": encode_jpeg(packet.frame, quality=95)
        for packet in OpenCVVideoFrameReader(video)
    }
    assert {path.name: path.read_bytes() for path in output.iterdir()} == expected


def test_plate_width_cap_does_not_upscale(video: Path, tmp_path: Path) -> None:
    output = tmp_path / "plate"
    _write_plate(video, output, frame_range=range(1, 2), max_width=128, quality=95)

    decoded = cv2.imread(str(output / "00001.jpg"))
    assert decoded is not None
    assert decoded.shape == (48, 64, 3)


@pytest.mark.parametrize(
    "frames", [range(0), range(-1, 2), range(1, 7), range(0, 6, 2)]
)
def test_plate_rejects_invalid_interval_before_writing(
    video: Path, tmp_path: Path, frames: range
) -> None:
    output = tmp_path / "plate"
    with pytest.raises(ValueError, match="frame_range"):
        _write_plate(video, output, frame_range=frames, max_width=None, quality=95)
    assert not output.exists()


@pytest.mark.parametrize("width", [0, -1])
def test_plate_rejects_invalid_width(video: Path, tmp_path: Path, width: int) -> None:
    output = tmp_path / "plate"
    with pytest.raises(ValueError, match="max_width"):
        _write_plate(video, output, frame_range=range(6), max_width=width, quality=95)
    assert not output.exists()


def test_plate_rejects_existing_directory(video: Path, tmp_path: Path) -> None:
    output = tmp_path / "plate"
    output.mkdir()
    sentinel = output / "00002.jpg"
    sentinel.write_bytes(b"existing plate")
    with pytest.raises(FileExistsError):
        _write_plate(video, output, frame_range=range(2, 3), max_width=None, quality=95)
    assert sentinel.read_bytes() == b"existing plate"

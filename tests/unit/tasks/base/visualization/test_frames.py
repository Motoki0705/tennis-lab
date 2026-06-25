"""Unit tests for image-source resolution and RGB loading."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.tasks.base.visualization.frames import (
    load_rgb_frames,
    read_rgb,
    resolve_image_paths,
)

pytestmark = pytest.mark.unit


def _write_img(path: Path, *, fill: tuple[int, int, int] = (10, 20, 30), h: int = 4, w: int = 5) -> None:
    """Write a solid BGR image (cv2 expects BGR)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img: np.ndarray = np.zeros((h, w, 3), np.uint8)
    img[:] = fill  # interpreted as BGR by imwrite
    cv2.imwrite(str(path), img)


def test_resolve_single_file(tmp_path: Path) -> None:
    p = tmp_path / "frame.png"
    _write_img(p)
    assert resolve_image_paths(p) == [p]


def test_resolve_directory_sorted_and_filtered(tmp_path: Path) -> None:
    _write_img(tmp_path / "b.png")
    _write_img(tmp_path / "a.jpg")
    (tmp_path / "notes.txt").write_text("ignore me")
    paths = resolve_image_paths(tmp_path)
    assert [p.name for p in paths] == ["a.jpg", "b.png"]


def test_resolve_glob_pattern(tmp_path: Path) -> None:
    _write_img(tmp_path / "f0.png")
    _write_img(tmp_path / "f1.png")
    _write_img(tmp_path / "other.png")
    paths = resolve_image_paths(str(tmp_path / "f*.png"))
    assert [p.name for p in paths] == ["f0.png", "f1.png"]


def test_resolve_max_frames_caps(tmp_path: Path) -> None:
    for i in range(5):
        _write_img(tmp_path / f"f{i}.png")
    paths = resolve_image_paths(tmp_path, max_frames=2)
    assert len(paths) == 2


def test_resolve_no_images_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No images found"):
        resolve_image_paths(tmp_path / "does_not_exist_*.png")


def test_read_rgb_converts_bgr_to_rgb(tmp_path: Path) -> None:
    p = tmp_path / "c.png"
    # write a blue image in BGR -> (255, 0, 0) bgr is pure blue
    _write_img(p, fill=(255, 0, 0))
    rgb = read_rgb(p)
    assert rgb.shape == (4, 5, 3)
    # after BGR->RGB the blue channel (index 2) should be 255
    assert rgb[0, 0, 2] == 255
    assert rgb[0, 0, 0] == 0


def test_read_rgb_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Failed to read"):
        read_rgb(tmp_path / "missing.png")


def test_load_rgb_frames_resizes(tmp_path: Path) -> None:
    _write_img(tmp_path / "a.png", h=8, w=10)
    _write_img(tmp_path / "b.png", h=8, w=10)
    frames = load_rgb_frames(tmp_path, resize_hw=(4, 6))
    assert [name for name, _ in frames] == ["a.png", "b.png"]
    for _, rgb in frames:
        assert rgb.shape == (4, 6, 3)

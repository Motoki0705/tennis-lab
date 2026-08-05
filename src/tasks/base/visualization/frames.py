"""Shared image-source loading for raster-family visualizations.

Resolves a visualization input source -- a single image file, a directory of
frames, or a glob pattern -- into an ordered list of RGB frames.  Used by the
ball-detection and court-detection clip/frame visualization pipelines.
"""

from __future__ import annotations

import glob as globlib
from pathlib import Path
from typing import cast

import cv2
import numpy as np

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def resolve_image_paths(
    source: str | Path,
    *,
    max_frames: int | None = None,
) -> list[Path]:
    """Resolve a source into an ordered list of image paths.

    Args:
        source: A single image file, a directory of frames, or a glob pattern.
        max_frames: Optional cap on the number of returned paths.

    Returns:
        Sorted list of image paths.
    """
    source_path = Path(source)
    if source_path.is_file():
        paths = [source_path]
    elif source_path.is_dir():
        paths = sorted(
            p for p in source_path.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
    else:
        paths = sorted(Path(p) for p in globlib.glob(str(source)))

    if not paths:
        raise FileNotFoundError(f"No images found for source: {source}")
    if max_frames is not None:
        paths = paths[:max_frames]
    return paths


def read_rgb(path: Path) -> np.ndarray:
    """Read an image file as an ``(H, W, 3)`` uint8 RGB array."""
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cast("np.ndarray", cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def load_rgb_frames(
    source: str | Path,
    *,
    resize_hw: tuple[int, int] | None = None,
    max_frames: int | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Load an image source into ``(name, rgb_uint8)`` frames.

    Args:
        source: A single image file, a directory of frames, or a glob pattern.
        resize_hw: Optional ``(height, width)`` to resize each frame to.
        max_frames: Optional cap on the number of frames.

    Returns:
        Ordered list of ``(file_name, rgb_uint8)`` tuples.
    """
    frames: list[tuple[str, np.ndarray]] = []
    for path in resolve_image_paths(source, max_frames=max_frames):
        rgb = read_rgb(path)
        if resize_hw is not None:
            rgb = cv2.resize(
                rgb,
                (resize_hw[1], resize_hw[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        frames.append((path.name, rgb))
    return frames

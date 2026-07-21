"""High-level animation encoding for RGB frame sequences.

GIF output uses one shared palette for the full clip and skips Pillow's expensive
post-encoding optimization by default. MP4 output streams frames through the
existing PyAV-backed H.264 writer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from src.utils.video.writer import VideoWriter

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class GifEncodingOptions:
    """Speed/quality controls for GIF encoding.

    Args:
        colors: Number of colors in the shared palette.
        optimize: Run Pillow's additional GIF optimization pass.
        dither: Apply Floyd-Steinberg dithering while mapping frames to the palette.
        palette_sample_frames: Maximum number of representative frames used to
            build the shared palette.
        palette_max_size: Maximum width/height of each representative thumbnail.
    """

    colors: int = 256
    optimize: bool = False
    dither: bool = False
    palette_sample_frames: int = 16
    palette_max_size: int = 256


def save_rgb_animation(
    frames_rgb: Sequence[NDArray[np.uint8]],
    path: str | Path,
    *,
    fps: float,
    loop: int = 0,
    video_crf: int = 17,
    gif_options: GifEncodingOptions | None = None,
) -> None:
    """Encode RGB uint8 frames as GIF or MP4 based on ``path`` suffix.

    MP4 frames with odd dimensions are padded by at most one row/column using
    edge pixels because the shared H.264 writer uses ``yuv420p``.
    """
    frames = _validate_frames(frames_rgb)
    output_path = Path(path)
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if loop < 0:
        raise ValueError("loop must be non-negative.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        _save_gif(
            frames,
            output_path,
            fps=fps,
            loop=loop,
            options=gif_options or GifEncodingOptions(),
        )
        return
    if suffix == ".mp4":
        _save_mp4(frames, output_path, fps=fps, crf=video_crf)
        return
    raise ValueError(
        f"Unsupported animation output suffix {output_path.suffix!r}; "
        "expected .gif or .mp4."
    )


def _validate_frames(
    frames_rgb: Sequence[NDArray[np.uint8]],
) -> list[NDArray[np.uint8]]:
    if not frames_rgb:
        raise ValueError("At least one frame is required to save an animation.")

    validated: list[NDArray[np.uint8]] = []
    expected_shape: tuple[int, int, int] | None = None
    for index, frame in enumerate(frames_rgb):
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                f"frame {index} must have shape (H, W, 3), got {array.shape}."
            )
        if array.dtype != np.uint8:
            raise ValueError(f"frame {index} must be uint8, got {array.dtype}.")
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError(
                f"frame size changed from {expected_shape} to {array.shape} at index {index}."
            )
        validated.append(np.ascontiguousarray(array))
    return validated


def _save_mp4(
    frames: Sequence[NDArray[np.uint8]],
    path: Path,
    *,
    fps: float,
    crf: int,
) -> None:
    with VideoWriter(path, fps=fps, crf=crf) as writer:
        for frame in frames:
            writer.write_frame(_pad_to_even(frame))


def _pad_to_even(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    pad_height = frame.shape[0] % 2
    pad_width = frame.shape[1] % 2
    if pad_height == 0 and pad_width == 0:
        return frame
    return np.pad(
        frame,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode="edge",
    )


def _save_gif(
    frames: Sequence[NDArray[np.uint8]],
    path: Path,
    *,
    fps: float,
    loop: int,
    options: GifEncodingOptions,
) -> None:
    _validate_gif_options(options)
    palette = _build_shared_palette(frames, options)
    dither = (
        Image.Dither.FLOYDSTEINBERG if options.dither else Image.Dither.NONE
    )
    quantized = [
        Image.fromarray(frame).quantize(palette=palette, dither=dither)
        for frame in frames
    ]
    duration_ms = max(int(round(1000.0 / fps)), 1)
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=loop,
        disposal=2,
        optimize=options.optimize,
    )


def _validate_gif_options(options: GifEncodingOptions) -> None:
    if not 2 <= options.colors <= 256:
        raise ValueError("GIF colors must be in [2, 256].")
    if options.palette_sample_frames <= 0:
        raise ValueError("palette_sample_frames must be positive.")
    if options.palette_max_size <= 0:
        raise ValueError("palette_max_size must be positive.")


def _build_shared_palette(
    frames: Sequence[NDArray[np.uint8]],
    options: GifEncodingOptions,
) -> Image.Image:
    sample_count = min(len(frames), options.palette_sample_frames)
    indices = np.linspace(0, len(frames) - 1, sample_count, dtype=np.int64)
    samples: list[NDArray[np.uint8]] = []
    for index in indices:
        image = Image.fromarray(frames[int(index)])
        image.thumbnail(
            (options.palette_max_size, options.palette_max_size),
            Image.Resampling.BILINEAR,
        )
        samples.append(np.asarray(image, dtype=np.uint8))

    atlas = np.concatenate(samples, axis=0)
    return Image.fromarray(atlas).quantize(
        colors=options.colors,
        method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.NONE,
    )

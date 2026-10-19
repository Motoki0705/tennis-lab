"""Stream RGB presentation frames to H.264, sampled GIF and named stills."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.tasks.base.visualization.gif import save_gif
from src.utils.video.writer import VideoWriter


def _check_name(name: str) -> None:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(
            "output names must be nonempty filename stems, without separators"
        )


def render_presentation(
    draw_frame: Callable[[int], NDArray[np.uint8]],
    *,
    frame_count: int,
    fps: float,
    output_dir: Path,
    name: str,
    gif_width: int,
    gif_fps: float,
    stills: Mapping[str, int],
    manifest: Mapping[str, Any],
) -> None:
    """Write ``name.mp4``, ``name.gif``, ``stills/*.png`` and ``manifest.json``.

    An existing destination (including an empty directory) is always rejected.
    Frames must be RGB uint8 arrays of one consistent, positive, even size.
    GIF samples use the source frame at or immediately before each GIF time;
    its last frame may extend the duration by less than one GIF frame interval.
    Only GIF-sized samples are buffered; each full-sized frame is drawn once.
    The manifest is caller-owned JSON metadata, preserved without extra keys.
    A failed render leaves its partial directory for inspection and refuses a
    subsequent overwrite; no alternative encoder or frame conversion is used.
    """
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output_dir already exists: {output_dir}")
    for value, label in ((frame_count, "frame_count"), (gif_width, "gif_width")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    for rate in (fps, gif_fps):
        if isinstance(rate, bool) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("fps and gif_fps must be finite and positive")
    _check_name(name)
    for still_name, index in stills.items():
        _check_name(still_name)
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("still frame indices must be integers")
        if not 0 <= index < frame_count:
            raise ValueError(f"still frame index out of range: {index}")
    # Validate serialization before producing any output; NaN is not valid JSON.
    manifest_json = json.dumps(
        dict(manifest), ensure_ascii=False, indent=2, allow_nan=False
    )
    source_rate = Fraction(fps).limit_denominator(1_000_000)
    gif_rate = Fraction(gif_fps).limit_denominator(1_000_000)
    gif_count = math.ceil(frame_count * gif_rate / source_rate)
    gif_indices = [int(index * source_rate // gif_rate) for index in range(gif_count)]

    output_dir.mkdir(parents=True, exist_ok=False)
    still_dir = output_dir / "stills"
    still_dir.mkdir()
    still_names: dict[int, list[str]] = {}
    for still_name, index in stills.items():
        still_names.setdefault(index, []).append(still_name)
    gif_frames: list[NDArray[np.uint8]] = []
    shape: tuple[int, ...] | None = None
    gif_cursor = 0
    with VideoWriter(output_dir / f"{name}.mp4", fps=fps) as writer:
        for index in range(frame_count):
            frame = draw_frame(index)
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
                raise TypeError("draw_frame must return a uint8 numpy array")
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(
                    "draw_frame must return RGB frames with shape (H, W, 3)"
                )
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0 or height % 2 or width % 2:
                raise ValueError("H.264 frames must have positive, even dimensions")
            if shape is not None and frame.shape != shape:
                raise ValueError("draw_frame size changed during rendering")
            shape = frame.shape
            writer.write_frame(frame)
            image = Image.fromarray(frame)
            for still_name in still_names.get(index, ()):
                image.save(still_dir / f"{still_name}.png")
            if gif_cursor < gif_count and gif_indices[gif_cursor] == index:
                gif_height = max(1, round(height * gif_width / width))
                sample: NDArray[np.uint8] = np.array(
                    image.resize(
                        (gif_width, gif_height),
                        Image.Resampling.LANCZOS,
                    )
                )
                while gif_cursor < gif_count and gif_indices[gif_cursor] == index:
                    gif_frames.append(sample)
                    gif_cursor += 1
    save_gif(frames_rgb=gif_frames, path=output_dir / f"{name}.gif", fps=gif_fps)
    (output_dir / "manifest.json").write_text(manifest_json + "\n", encoding="utf-8")

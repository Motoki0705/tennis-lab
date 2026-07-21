"""Shared saving helpers for qualitative validation artifacts.

Task LightningModules build a sequence of RGB frames (raster renderers) or a
matplotlib animation (scene renderers) and delegate persistence to the helpers
here, which:

* write a versioned artifact to ``artifact_dir`` -- an animated ``.gif`` for
  multi-frame outputs, or a ``.png`` for single-frame outputs, and
* log the same content to TensorBoard -- ``add_video`` (animated) for clips,
  ``add_image`` for single frames.

The TensorBoard video path is implemented without ``moviepy`` (which is not a
project dependency): frames are encoded to a GIF with the shared
:func:`~src.utils.video.animation.save_rgb_animation` writer and the encoded
bytes are emitted as an animated image summary -- exactly how ``add_video``
stores a GIF internally.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from src.utils.video.animation import save_rgb_animation

logger = logging.getLogger(__name__)


def save_qualitative_clip(
    *,
    frames_rgb: Sequence[np.ndarray],
    artifact_dir: Path,
    name: str,
    tb_writer: Any | None,
    tag: str,
    global_step: int,
    fps: float = 10.0,
) -> Path:
    """Persist RGB frames as an artifact and log them to TensorBoard.

    A single frame is saved as ``<name>.png`` and logged with ``add_image``;
    multiple frames are saved as ``<name>.gif`` and logged with an animated
    video summary.

    Args:
        frames_rgb: Non-empty sequence of ``(H, W, 3)`` uint8 RGB frames.
        artifact_dir: Directory to write the artifact into.
        name: Artifact file stem (without extension).
        tb_writer: TensorBoard ``SummaryWriter`` (may be ``None``).
        tag: TensorBoard tag.
        global_step: Global step for the summary.
        fps: Playback frames per second for multi-frame clips.

    Returns:
        Path to the written artifact.
    """
    frames = [np.ascontiguousarray(_to_uint8_rgb(f)) for f in frames_rgb]
    if not frames:
        raise ValueError("At least one frame is required to save a qualitative clip.")

    artifact_dir.mkdir(parents=True, exist_ok=True)

    if len(frames) == 1:
        path = artifact_dir / f"{name}.png"
        Image.fromarray(frames[0]).save(path)
        _add_image(tb_writer, tag, frames[0], global_step)
        return path

    path = artifact_dir / f"{name}.gif"
    save_rgb_animation(frames, path, fps=fps)
    _add_gif_video(tb_writer, tag, path, global_step, fps)
    return path


def save_qualitative_animation(
    *,
    animation: Any,
    artifact_dir: Path,
    name: str,
    tb_writer: Any | None,
    tag: str,
    global_step: int,
    fps: float = 10.0,
) -> Path:
    """Persist a matplotlib ``FuncAnimation`` as a GIF and log it to TensorBoard.

    The animation is rendered with Pillow (no ``moviepy`` required); the encoded
    GIF is then logged as an animated video summary.

    Args:
        animation: A matplotlib ``FuncAnimation``.
        artifact_dir: Directory to write the artifact into.
        name: Artifact file stem (without extension).
        tb_writer: TensorBoard ``SummaryWriter`` (may be ``None``).
        tag: TensorBoard tag.
        global_step: Global step for the summary.
        fps: Playback frames per second.

    Returns:
        Path to the written ``.gif`` artifact.
    """
    from matplotlib import pyplot as plt
    from matplotlib.animation import PillowWriter

    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{name}.gif"
    animation.save(str(path), writer=PillowWriter(fps=int(round(fps))))
    plt.close(animation._fig)
    _add_gif_video(tb_writer, tag, path, global_step, fps)
    return path


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cast("np.ndarray", arr)


def _add_image(tb_writer: Any | None, tag: str, frame_rgb: np.ndarray, global_step: int) -> None:
    if tb_writer is None:
        return
    # SummaryWriter.add_image expects (C, H, W).
    tb_writer.add_image(tag, np.transpose(frame_rgb, (2, 0, 1)), global_step)


def _add_gif_video(
    tb_writer: Any | None,
    tag: str,
    gif_path: Path,
    global_step: int,
    fps: float,
) -> None:
    """Emit an already-encoded GIF as an animated TensorBoard image summary.

    Mirrors how ``SummaryWriter.add_video`` stores a clip (an ``image/gif``
    encoded ``Summary.Image``) but reuses our Pillow GIF instead of requiring
    ``moviepy``. Failures are logged and swallowed so training is never broken
    by a logging issue.
    """
    if tb_writer is None:
        return
    try:
        from tensorboard.compat.proto.summary_pb2 import Summary

        with Image.open(gif_path) as im:
            width, height = im.size
        encoded = gif_path.read_bytes()
        image = Summary.Image(
            height=height,
            width=width,
            colorspace=3,
            encoded_image_string=encoded,
        )
        summary = Summary(value=[Summary.Value(tag=tag, image=image)])
        tb_writer._get_file_writer().add_summary(summary, global_step)
    except Exception as exc:  # pragma: no cover - logging must never break training
        logger.warning("Failed to log animated qualitative summary '%s': %s", tag, exc)

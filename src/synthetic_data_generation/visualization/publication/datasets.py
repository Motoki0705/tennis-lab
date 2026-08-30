"""Deterministic endpoint-inclusive GIF rendering for canonical datasets."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.visualization.overlays import (
    new_ball_history,
    render_blcs_overlay,
    render_court_overlay,
    render_plcs_overlay,
)
from src.synthetic_data_generation.visualization.sources import (
    BLCSVisualizationSource,
    CourtVisualizationSource,
    PLCSVisualizationSource,
)

GIF_ENCODER = "pillow-gif-fixed-palette-v1"


@dataclass(frozen=True, slots=True)
class DatasetGifResult:
    """Exact source/media mapping returned by one dataset GIF renderer."""

    domain: str
    dataset_schema: str
    dataset_scene_id: str
    source_count: int
    source_width: int
    source_height: int
    source_fps: float | None
    width: int
    height: int
    duration_ms: int
    mapping: tuple[Mapping[str, object], ...]


def render_court_dataset_gif(
    source: CourtVisualizationSource,
    output: Path,
    *,
    trajectory_id: str,
    frame_indices: tuple[int, ...],
    size: tuple[int, int],
    duration_ms: int,
) -> DatasetGifResult:
    """Render one exact Court trajectory selection without dropping source records."""
    frames = (
        render_court_overlay(frame, trajectory_id=trajectory_id)
        for frame in source.frames()
    )
    mapping = _render_selected_gif(
        frames,
        source.frame_order,
        output,
        frame_indices=frame_indices,
        size=size,
        duration_ms=duration_ms,
    )
    return DatasetGifResult(
        domain="court",
        dataset_schema=source.dataset_schema,
        dataset_scene_id=source.dataset_scene_id,
        source_count=source.frame_count,
        source_width=source.width,
        source_height=source.height,
        source_fps=None,
        width=size[0],
        height=size[1],
        duration_ms=duration_ms,
        mapping=mapping,
    )


def render_blcs_dataset_gif(
    source: BLCSVisualizationSource,
    output: Path,
    *,
    logical_scene_id: str,
    camera_id: str,
    frame_indices: tuple[int, ...],
    size: tuple[int, int],
    duration_ms: int,
    history_frames: int,
) -> DatasetGifResult:
    """Render one complete BLCS view while preserving history through unselected frames."""
    history = new_ball_history(source.object_ids, history_frames=history_frames)
    frames = (
        render_blcs_overlay(
            frame,
            logical_scene_id=logical_scene_id,
            camera_id=camera_id,
            object_ids=source.object_ids,
            court_kp=source.court_kp,
            court_vis=source.court_vis,
            history=history,
            history_frames=history_frames,
        )
        for frame in source.frames()
    )
    mapping = _render_selected_gif(
        frames,
        source.frame_order,
        output,
        frame_indices=frame_indices,
        size=size,
        duration_ms=duration_ms,
    )
    return DatasetGifResult(
        domain="blcs",
        dataset_schema=source.dataset_schema,
        dataset_scene_id=source.dataset_scene_id,
        source_count=source.frame_count,
        source_width=source.width,
        source_height=source.height,
        source_fps=source.source_fps,
        width=size[0],
        height=size[1],
        duration_ms=duration_ms,
        mapping=tuple(
            {
                **value,
                "logical_scene_id": logical_scene_id,
                "camera_id": camera_id,
            }
            for value in mapping
        ),
    )


def render_plcs_dataset_gif(
    source: PLCSVisualizationSource,
    output: Path,
    *,
    logical_scene_id: str,
    camera_id: str,
    frame_indices: tuple[int, ...],
    size: tuple[int, int],
    duration_ms: int,
) -> DatasetGifResult:
    """Render one complete PLCS view in canonical logical-frame order."""
    frames = (
        render_plcs_overlay(
            frame,
            logical_scene_id=logical_scene_id,
            camera_id=camera_id,
            object_ids=source.object_ids,
        )
        for frame in source.frames()
    )
    mapping = _render_selected_gif(
        frames,
        source.frame_order,
        output,
        frame_indices=frame_indices,
        size=size,
        duration_ms=duration_ms,
    )
    return DatasetGifResult(
        domain="plcs",
        dataset_schema=source.dataset_schema,
        dataset_scene_id=source.dataset_scene_id,
        source_count=source.frame_count,
        source_width=source.width,
        source_height=source.height,
        source_fps=None,
        width=size[0],
        height=size[1],
        duration_ms=duration_ms,
        mapping=tuple(
            {
                **value,
                "logical_scene_id": logical_scene_id,
                "camera_id": camera_id,
            }
            for value in mapping
        ),
    )


def write_deterministic_gif(
    frames: tuple[NDArray[np.uint8], ...],
    output: Path,
    *,
    duration_ms: int,
) -> None:
    """Write an exact-size, fixed-loop Pillow GIF and mechanically reopen it."""
    if not frames:
        raise ValueError("A publication GIF requires at least one frame.")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Publication staging artifact already exists: {output}")
    expected_shape = frames[0].shape
    if (
        len(expected_shape) != 3
        or expected_shape[2] != 3
        or any(
            frame.shape != expected_shape or frame.dtype != np.uint8 for frame in frames
        )
    ):
        raise ValueError("Publication GIF frames must share one uint8 HxWx3 shape.")
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms <= 0
    ):
        raise ValueError("GIF duration_ms must be a positive integer.")
    images = tuple(
        Image.fromarray(frame, mode="RGB").quantize(
            colors=255,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        for frame in frames
    )
    images[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=list(images[1:]),
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
        comment=b"tennis-lab deterministic publication gif v1",
    )
    with Image.open(output) as reopened:
        if (
            reopened.format != "GIF"
            or reopened.size != (expected_shape[1], expected_shape[0])
            or reopened.n_frames != len(frames)
        ):
            raise ValueError("Published GIF media metadata differs after reopening.")
        for index in range(reopened.n_frames):
            reopened.seek(index)
            if reopened.info.get("duration") != duration_ms:
                raise ValueError("Published GIF frame timing differs after reopening.")
            reopened.convert("RGB").load()


def _render_selected_gif(
    rendered_frames: Iterator[NDArray[np.uint8]],
    frame_order: tuple[Mapping[str, object], ...],
    output: Path,
    *,
    frame_indices: tuple[int, ...],
    size: tuple[int, int],
    duration_ms: int,
) -> tuple[Mapping[str, object], ...]:
    source_count = len(frame_order)
    if source_count == 0:
        raise ValueError("Dataset publication source is empty.")
    if frame_indices[0] != 0 or frame_indices[-1] != source_count - 1:
        raise ValueError(
            "Publication frame_indices must include both source timeline endpoints."
        )
    if any(index >= source_count for index in frame_indices):
        raise ValueError("Publication frame index exceeds the exact source inventory.")
    selected_set = set(frame_indices)
    selected: list[NDArray[np.uint8]] = []
    observed_count = 0
    for source_index, frame in enumerate(rendered_frames):
        observed_count += 1
        if source_index not in selected_set:
            continue
        rgb = _rgb_uint8(frame)
        image = Image.fromarray(rgb, mode="RGB")
        if image.size != size:
            image = image.resize(size, resample=Image.Resampling.LANCZOS)
        selected.append(cast(NDArray[np.uint8], np.asarray(image, dtype=np.uint8)))
    if observed_count != source_count:
        raise ValueError(
            "Dataset renderer did not consume the exact canonical source inventory."
        )
    if len(selected) != len(frame_indices):
        raise ValueError("Dataset GIF selection silently lost configured frames.")
    write_deterministic_gif(tuple(selected), output, duration_ms=duration_ms)
    return tuple(
        {"source_index": index, **dict(frame_order[index])} for index in frame_indices
    )


def _rgb_uint8(value: NDArray[np.uint8]) -> NDArray[np.uint8]:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Dataset overlay must return uint8 HxWx3 RGB.")
    return cast(NDArray[np.uint8], array)


__all__ = [
    "DatasetGifResult",
    "GIF_ENCODER",
    "render_blcs_dataset_gif",
    "render_court_dataset_gif",
    "render_plcs_dataset_gif",
    "write_deterministic_gif",
]

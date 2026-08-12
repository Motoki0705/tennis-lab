"""Streaming MP4 publication for canonical synthetic-dataset views."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.visualization.contracts import (
    VISUALIZATION_METADATA_SCHEMA,
    DatasetVisualizationDomain,
    DatasetVisualizationRequest,
    DatasetVisualizationResult,
)
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
from src.utils.video.writer import VideoWriter


def visualize_dataset(
    request: DatasetVisualizationRequest,
) -> DatasetVisualizationResult:
    """Validate one exact selection, stream overlays, and publish MP4 + JSON."""
    if not isinstance(request, DatasetVisualizationRequest):
        raise TypeError("visualize_dataset requires DatasetVisualizationRequest.")
    frame_order: tuple[Mapping[str, object], ...]
    if request.domain is DatasetVisualizationDomain.COURT:
        assert request.trajectory_id is not None
        source = CourtVisualizationSource(
            request.dataset_root,
            trajectory_id=request.trajectory_id,
        )
        dataset_schema = source.dataset_schema
        dataset_scene_id = source.dataset_scene_id
        source_width, source_height = source.width, source.height

        frame_iterator = (
            render_court_overlay(frame, trajectory_id=request.trajectory_id)
            for frame in source.frames()
        )
        frame_order = source.frame_order
        selection: dict[str, object] = {
            "trajectory_id": request.trajectory_id,
            "logical_scene_id": None,
            "camera_id": None,
        }
        source_fps: float | None = None
    elif request.domain is DatasetVisualizationDomain.BLCS:
        assert request.logical_scene_id is not None
        assert request.camera_id is not None
        blcs_source = BLCSVisualizationSource(
            request.dataset_root,
            logical_scene_id=request.logical_scene_id,
            camera_id=request.camera_id,
        )
        dataset_schema = blcs_source.dataset_schema
        dataset_scene_id = blcs_source.dataset_scene_id
        source_width, source_height = blcs_source.width, blcs_source.height
        history = new_ball_history(
            blcs_source.object_ids,
            history_frames=request.history_frames,
        )

        frame_iterator = (
            render_blcs_overlay(
                frame,
                logical_scene_id=request.logical_scene_id,
                camera_id=request.camera_id,
                object_ids=blcs_source.object_ids,
                court_kp=blcs_source.court_kp,
                court_vis=blcs_source.court_vis,
                history=history,
                history_frames=request.history_frames,
            )
            for frame in blcs_source.frames()
        )
        frame_order = blcs_source.frame_order
        selection = {
            "trajectory_id": None,
            "logical_scene_id": request.logical_scene_id,
            "camera_id": request.camera_id,
        }
        source_fps = blcs_source.source_fps
    else:
        assert request.logical_scene_id is not None
        assert request.camera_id is not None
        plcs_source = PLCSVisualizationSource(
            request.dataset_root,
            logical_scene_id=request.logical_scene_id,
            camera_id=request.camera_id,
        )
        dataset_schema = plcs_source.dataset_schema
        dataset_scene_id = plcs_source.dataset_scene_id
        source_width, source_height = plcs_source.width, plcs_source.height

        frame_iterator = (
            render_plcs_overlay(
                frame,
                logical_scene_id=request.logical_scene_id,
                camera_id=request.camera_id,
                object_ids=plcs_source.object_ids,
            )
            for frame in plcs_source.frames()
        )
        frame_order = plcs_source.frame_order
        selection = {
            "trajectory_id": None,
            "logical_scene_id": request.logical_scene_id,
            "camera_id": request.camera_id,
        }
        source_fps = None
    width = source_width + source_width % 2
    height = source_height + source_height % 2
    right_padding = width - source_width
    bottom_padding = height - source_height
    if right_padding or bottom_padding:
        frame_iterator = (
            _pad_frame_for_yuv420(
                frame,
                source_width=source_width,
                source_height=source_height,
                output_width=width,
                output_height=height,
            )
            for frame in frame_iterator
        )
    if len(frame_order) == 0:
        raise ValueError("Visualization selection produced no frames.")
    output = request.output_video
    metadata_path = request.metadata_path
    output.parent.mkdir(parents=True, exist_ok=True)
    for published in (output, metadata_path):
        if published.exists() or published.is_symlink():
            raise FileExistsError(
                f"Visualization publication appeared after request validation: {published}"
            )
    temporary_video = _new_staging_file(output, suffix=".mp4")
    temporary_metadata: _OwnedStagingFile | None = None
    encoded_count = 0
    try:
        typed_frames = cast(
            Iterator[NDArray[np.uint8]],
            frame_iterator,
        )
        with VideoWriter(
            temporary_video.path,
            fps=request.fps,
            crf=request.crf,
        ) as writer:
            for value in typed_frames:
                writer.write_frame(value)
                encoded_count += 1
        if encoded_count != len(frame_order):
            raise ValueError(
                "Encoded frame count differs from the canonical selected inventory."
            )
        metadata: Mapping[str, object] = {
            "schema": VISUALIZATION_METADATA_SCHEMA,
            "domain": request.domain.value,
            "dataset_schema": dataset_schema,
            "dataset_scene_id": dataset_scene_id,
            "selection": selection,
            "frame_count": encoded_count,
            "source_frame_order": list(frame_order),
            "source_width": source_width,
            "source_height": source_height,
            "width": width,
            "height": height,
            "padding": {
                "right": right_padding,
                "bottom": bottom_padding,
            },
            "output_fps": request.fps,
            "source_fps": source_fps,
            "history_frames": request.history_frames,
            "video": {
                "file_name": output.name,
                "codec": "libx264",
                "pixel_format": "yuv420p",
                "crf": request.crf,
            },
        }
        temporary_metadata = _new_staging_file(output, suffix=".json")
        temporary_metadata.path.write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _publish_exclusively(temporary_metadata.path, metadata_path)
        try:
            _publish_exclusively(temporary_video.path, output)
        except Exception:
            _unlink_if_same_file(metadata_path, temporary_metadata.path)
            raise
    finally:
        temporary_video.cleanup()
        if temporary_metadata is not None:
            temporary_metadata.cleanup()
    return DatasetVisualizationResult(
        video_path=output,
        metadata_path=metadata_path,
        frame_count=encoded_count,
        width=width,
        height=height,
    )


@dataclass(frozen=True, slots=True)
class _OwnedStagingFile:
    """Invocation-private staging path plus its immutable inode identity."""

    path: Path
    device: int
    inode: int

    def cleanup(self) -> None:
        """Remove only the staging inode created by this invocation."""
        try:
            stat = self.path.lstat()
        except FileNotFoundError:
            return
        if (stat.st_dev, stat.st_ino) == (self.device, self.inode):
            self.path.unlink()


def _new_staging_file(output: Path, *, suffix: str) -> _OwnedStagingFile:
    """Create an exclusive invocation-unique sibling for atomic publication."""
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=f".staging{suffix}",
        dir=output.parent,
    )
    try:
        stat = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    return _OwnedStagingFile(
        path=Path(raw_path),
        device=stat.st_dev,
        inode=stat.st_ino,
    )


def _publish_exclusively(staged: Path, target: Path) -> None:
    """Atomically link one staged inode to an absent final name."""
    try:
        os.link(staged, target, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Visualization publication already exists: {target}"
        ) from error


def _unlink_if_same_file(target: Path, staged: Path) -> None:
    """Roll back only a final link still owned by the current staging inode."""
    try:
        target_stat = target.lstat()
        staged_stat = staged.lstat()
    except FileNotFoundError:
        return
    if (target_stat.st_dev, target_stat.st_ino) == (
        staged_stat.st_dev,
        staged_stat.st_ino,
    ):
        target.unlink()


def _pad_frame_for_yuv420(
    frame: NDArray[np.uint8],
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> NDArray[np.uint8]:
    """Pad only the right/bottom edge when canonical RGB dimensions are odd."""
    expected_shape = (source_height, source_width, 3)
    if frame.shape != expected_shape or frame.dtype != np.uint8:
        raise ValueError(
            "Rendered visualization frame differs from its declared source shape: "
            f"expected {expected_shape} uint8, got {frame.shape} {frame.dtype}."
        )
    padded: NDArray[np.uint8] = np.zeros(
        (output_height, output_width, 3), dtype=np.uint8
    )
    padded[:source_height, :source_width] = frame
    return padded


__all__ = ["visualize_dataset"]

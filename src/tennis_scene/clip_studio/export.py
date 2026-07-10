"""Export synchronized clips that satisfy the tennis_scene pipeline contract.

`TennisSceneOrchestrator` requires per-camera videos with identical fps,
frame count and resolution. The exporter cuts each project clip from the
unsynchronized sources by mapping every output frame's global time to the
nearest source frame (project convention ``local = global + offset_sec``),
re-encodes, and writes a ``clip.json`` manifest whose ``video_paths`` /
``camera_ids`` plug directly into ``run_pipeline.py``.

Planning (:func:`plan_clip_export`) is pure and fully validated; encoding
(:func:`export_clip`) streams each source once in index order, so long
videos are never decoded beyond the clip's span. Exported files are
re-probed and checked against the plan — a broken export raises instead of
producing an out-of-contract clip.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.tennis_scene.clip_studio.imaging import compute_letterbox, letterbox_frame
from src.tennis_scene.clip_studio.project import Clip, ClipStudioProject
from src.tennis_scene.clip_studio.timeline import (
    source_coverage_sec,
    source_frame_index,
)
from src.utils.io import save_json_atomic, utc_now_iso
from src.utils.video import (
    RandomAccessVideoReader,
    VideoInfo,
    VideoWriter,
    probe_video_info,
)

LOGGER = logging.getLogger(__name__)

MANIFEST_FILENAME = "clip.json"


@dataclass(frozen=True)
class ExportSettings:
    """Target format of exported clips.

    Attributes:
        output_dir: Directory receiving one subdirectory per clip.
        fps: Target frame rate. ``None`` requires all sources to share one
            fps (which is then used).
        width, height: Target resolution. Both ``None`` requires all sources
            to share one resolution; otherwise both must be set and sources
            are letterboxed into it.
        crf: x264 quality (17 is visually lossless).
        overwrite: Allow re-exporting into an existing clip directory.
    """

    output_dir: Path
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    crf: int = 17
    overwrite: bool = False


@dataclass(frozen=True)
class CameraExportPlan:
    """Frame mapping of one camera for one clip."""

    camera_id: str
    source_path: Path
    offset_sec: float
    source_info: VideoInfo
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class ClipExportPlan:
    """Validated, self-contained description of one clip export."""

    clip_name: str
    global_start_sec: float
    global_end_sec: float
    fps: float
    width: int
    height: int
    num_frames: int
    cameras: tuple[CameraExportPlan, ...]


@dataclass(frozen=True)
class ClipExportResult:
    """Artifacts written for one exported clip."""

    clip_dir: Path
    video_paths: list[Path]
    manifest_path: Path


def _resolve_target_fps(infos: Sequence[VideoInfo], settings: ExportSettings) -> float:
    if settings.fps is not None:
        if settings.fps <= 0:
            raise ValueError(f"fps must be positive, got {settings.fps}")
        return float(settings.fps)
    fps_values = sorted({round(info.fps, 6) for info in infos})
    if len(fps_values) != 1:
        raise ValueError(
            f"sources have mixed fps {fps_values}; set an explicit export fps"
        )
    return float(fps_values[0])


def _resolve_target_size(
    infos: Sequence[VideoInfo], settings: ExportSettings
) -> tuple[int, int]:
    if (settings.width is None) != (settings.height is None):
        raise ValueError("width and height must be set together (or both omitted)")
    if settings.width is not None and settings.height is not None:
        width, height = int(settings.width), int(settings.height)
    else:
        sizes = sorted({(info.width, info.height) for info in infos})
        if len(sizes) != 1:
            raise ValueError(
                f"sources have mixed resolutions {sizes}; "
                "set explicit export width/height"
            )
        width, height = sizes[0]
    if width <= 0 or height <= 0:
        raise ValueError(f"target size must be positive, got {width}x{height}")
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError(
            f"target size must be even for H.264 encoding, got {width}x{height}"
        )
    return width, height


def plan_clip_export(
    project: ClipStudioProject,
    infos: Sequence[VideoInfo],
    clip: Clip,
    settings: ExportSettings,
) -> ClipExportPlan:
    """Validate and compute the per-camera frame mapping for one clip.

    Raises:
        ValueError: On invalid project/clip, mixed formats without explicit
            targets, or when a camera does not cover the clip's global range.
    """
    errors = project.validate()
    if errors:
        raise ValueError(f"Invalid project: {errors}")
    if len(infos) != len(project.sources):
        raise ValueError(
            f"infos length {len(infos)} must match sources {len(project.sources)}"
        )

    fps = _resolve_target_fps(infos, settings)
    width, height = _resolve_target_size(infos, settings)

    num_frames = round(clip.duration_sec * fps)
    if num_frames < 1:
        raise ValueError(
            f"clip '{clip.name}' spans {clip.duration_sec:.3f}s which is shorter "
            f"than one frame at {fps} fps"
        )

    cameras: list[CameraExportPlan] = []
    for source, info in zip(project.sources, infos, strict=True):
        indices: list[int] = []
        for output_frame in range(num_frames):
            global_sec = clip.start_sec + output_frame / fps
            index = source_frame_index(
                global_sec,
                offset_sec=source.offset_sec,
                fps=info.fps,
                frame_count=info.frame_count,
            )
            if index is None:
                coverage = source_coverage_sec(
                    offset_sec=source.offset_sec,
                    duration_sec=info.frame_count / info.fps,
                )
                raise ValueError(
                    f"camera '{source.camera_id}' does not cover clip "
                    f"'{clip.name}' at global t={global_sec:.3f}s "
                    f"(coverage [{coverage[0]:.3f}, {coverage[1]:.3f}]s)"
                )
            indices.append(index)
        cameras.append(
            CameraExportPlan(
                camera_id=source.camera_id,
                source_path=source.path,
                offset_sec=source.offset_sec,
                source_info=info,
                frame_indices=tuple(indices),
            )
        )
    return ClipExportPlan(
        clip_name=clip.name,
        global_start_sec=clip.start_sec,
        global_end_sec=clip.end_sec,
        fps=fps,
        width=width,
        height=height,
        num_frames=num_frames,
        cameras=tuple(cameras),
    )


def _write_camera_video(
    camera: CameraExportPlan, plan: ClipExportPlan, video_path: Path, crf: int
) -> None:
    source_size = (camera.source_info.width, camera.source_info.height)
    needs_fit = source_size != (plan.width, plan.height)
    previous_index: int | None = None
    previous_frame: np.ndarray | None = None
    with (
        RandomAccessVideoReader(camera.source_path) as reader,
        VideoWriter(video_path, fps=plan.fps, crf=crf) as writer,
    ):
        for position, index in enumerate(camera.frame_indices):
            if index == previous_index and previous_frame is not None:
                frame_rgb = previous_frame
            else:
                frame_bgr = reader.read(index)
                frame_rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
                if needs_fit:
                    frame_rgb, _ = letterbox_frame(frame_rgb, plan.width, plan.height)
                previous_index = index
                previous_frame = frame_rgb
            writer.write_frame(frame_rgb)
            if (position + 1) % 500 == 0:
                LOGGER.info(
                    f"  {camera.camera_id}: {position + 1}/{plan.num_frames} frames"
                )


def _build_manifest(plan: ClipExportPlan) -> dict[str, Any]:
    cameras: list[dict[str, Any]] = []
    for camera in plan.cameras:
        source_size = (camera.source_info.width, camera.source_info.height)
        letterbox = (
            compute_letterbox(*source_size, plan.width, plan.height).to_dict()
            if source_size != (plan.width, plan.height)
            else None
        )
        cameras.append(
            {
                "camera_id": camera.camera_id,
                "video": f"{camera.camera_id}.mp4",
                "source_path": str(camera.source_path),
                "offset_sec": camera.offset_sec,
                "source_fps": camera.source_info.fps,
                "source_frame_start": camera.frame_indices[0],
                "source_frame_end": camera.frame_indices[-1],
                "letterbox": letterbox,
            }
        )
    return {
        "clip_name": plan.clip_name,
        "fps": plan.fps,
        "num_frames": plan.num_frames,
        "width": plan.width,
        "height": plan.height,
        "global_start_sec": plan.global_start_sec,
        "global_end_sec": plan.global_end_sec,
        "camera_ids": [camera.camera_id for camera in plan.cameras],
        "video_paths": [f"{camera.camera_id}.mp4" for camera in plan.cameras],
        "cameras": cameras,
        "sync_source": "clip_studio",
        "exported_at": utc_now_iso(),
    }


def _verify_exported_video(video_path: Path, plan: ClipExportPlan) -> None:
    info = probe_video_info(video_path)
    problems: list[str] = []
    if info.frame_count != plan.num_frames:
        problems.append(f"frame_count {info.frame_count} != {plan.num_frames}")
    if (info.width, info.height) != (plan.width, plan.height):
        problems.append(
            f"resolution {info.width}x{info.height} != {plan.width}x{plan.height}"
        )
    if abs(info.fps - plan.fps) > 0.01:
        problems.append(f"fps {info.fps} != {plan.fps}")
    if problems:
        raise RuntimeError(
            f"Exported video {video_path} violates the pipeline contract: "
            f"{'; '.join(problems)}"
        )


def export_clip(plan: ClipExportPlan, settings: ExportSettings) -> ClipExportResult:
    """Encode one planned clip and write its manifest.

    Raises:
        ValueError: If the clip directory already contains files and
            ``settings.overwrite`` is false.
        RuntimeError: If a written video fails the post-export contract check.
    """
    clip_dir = Path(settings.output_dir) / plan.clip_name
    if clip_dir.exists() and any(clip_dir.iterdir()) and not settings.overwrite:
        raise ValueError(
            f"clip directory {clip_dir} is not empty; set overwrite=true to replace"
        )
    clip_dir.mkdir(parents=True, exist_ok=True)

    video_paths: list[Path] = []
    for camera in plan.cameras:
        video_path = clip_dir / f"{camera.camera_id}.mp4"
        LOGGER.info(
            f"Exporting {plan.clip_name}/{camera.camera_id}: "
            f"{plan.num_frames} frames from {camera.source_path}"
        )
        _write_camera_video(camera, plan, video_path, settings.crf)
        _verify_exported_video(video_path, plan)
        video_paths.append(video_path)

    manifest_path = save_json_atomic(_build_manifest(plan), clip_dir / MANIFEST_FILENAME)
    LOGGER.info(f"Exported clip '{plan.clip_name}' to {clip_dir}")
    return ClipExportResult(
        clip_dir=clip_dir, video_paths=video_paths, manifest_path=manifest_path
    )


def export_clips(
    project: ClipStudioProject,
    settings: ExportSettings,
    *,
    infos: Sequence[VideoInfo] | None = None,
    clip_names: Sequence[str] | None = None,
) -> list[ClipExportResult]:
    """Plan and export several clips of a project.

    Args:
        project: Validated clip studio project.
        settings: Target format and destination.
        infos: Pre-probed per-source metadata (probed from disk when omitted).
        clip_names: Subset of clips to export (all clips when omitted).
    """
    if infos is None:
        infos = [probe_video_info(source.path) for source in project.sources]
    if clip_names is None:
        clips = list(project.clips)
    else:
        clips = [project.clips[project.clip_index_by_name(name)] for name in clip_names]
    if not clips:
        raise ValueError("no clips to export")

    results: list[ClipExportResult] = []
    for clip in clips:
        plan = plan_clip_export(project, infos, clip, settings)
        results.append(export_clip(plan, settings))
    return results


__all__ = [
    "CameraExportPlan",
    "ClipExportPlan",
    "ClipExportResult",
    "ExportSettings",
    "MANIFEST_FILENAME",
    "export_clip",
    "export_clips",
    "plan_clip_export",
]

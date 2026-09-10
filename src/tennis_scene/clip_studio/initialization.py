"""Create initial sync once, before saving a new project; resume verbatim."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from src.tennis_scene.clip_studio.project import ClipSource, ClipStudioProject
from src.tennis_scene.configuration import ClipStudioRuntimeConfig
from src.utils.video.metadata import CreationTimeError, read_creation_time

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StartupNotice:
    message: str
    warning: bool = False


def initialize_recording_offsets(project: ClipStudioProject) -> StartupNotice:
    """Initialize a fresh project's sources together, or leave all at zero."""
    if project.clips or any(source.offset_sec != 0 for source in project.sources):
        raise ValueError("Recording-time initialization requires a fresh project")
    if not project.sources:
        raise ValueError("Recording-time initialization requires sources")
    timestamps: list[datetime] = []
    errors: list[str] = []
    for source in project.sources:
        try:
            timestamps.append(read_creation_time(source.path))
        except CreationTimeError as error:
            errors.append(f"{source.camera_id} ({source.path.name}): {error}")
    if errors:
        return StartupNotice(
            "録画開始時刻による同期の初期推定を行いませんでした。"
            "全カメラの時差を従来どおり0秒で開始します。"
            "音声同期または手動調整を利用してください。 " + " / ".join(errors),
            warning=True,
        )
    latest = max(timestamps)
    for source, timestamp in zip(project.sources, timestamps, strict=True):
        source.offset_sec = (latest - timestamp).total_seconds()
    references = ", ".join(
        source.camera_id for source in project.sources if source.offset_sec == 0
    )
    return StartupNotice(
        f"録画開始時刻から同期の初期値を設定しました（共通時刻0秒: {references} の先頭）。"
        "映像を確認し、必要に応じて音声同期または手動で調整してください。"
    )


def load_or_create_project(
    runtime: ClipStudioRuntimeConfig,
) -> tuple[ClipStudioProject, StartupNotice | None]:
    """Existing JSON is authoritative, even with zero offsets or no clips."""
    path = runtime.export.project_path
    resolver = runtime.export.resolver
    if path.is_file():
        if runtime.video_paths is not None or runtime.camera_ids is not None:
            raise ValueError(
                "Existing projects forbid video_paths and camera_ids; remove both keys."
            )
        return ClipStudioProject.load(path, resolver), None
    if runtime.video_paths is None or runtime.camera_ids is None:
        raise ValueError("New projects require video_paths and camera_ids.")
    missing = [video for video in runtime.video_paths if not video.is_file()]
    if missing:
        raise FileNotFoundError(f"video not found: {missing[0]}")
    project = ClipStudioProject(
        recording_id=runtime.recording_id,
        sources=[
            ClipSource(path=video, camera_id=camera_id)
            for video, camera_id in zip(
                runtime.video_paths, runtime.camera_ids, strict=True
            )
        ],
    )
    notice = initialize_recording_offsets(project)
    project.save(path, resolver)
    LOGGER.log(logging.WARNING if notice.warning else logging.INFO, notice.message)
    return project, notice

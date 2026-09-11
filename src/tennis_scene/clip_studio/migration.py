"""One-way migration from recording-based storage to dataset/video/clip storage."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.io import load_json, save_json_atomic

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class LegacyVideo:
    recording_id: str
    video_id: str
    raw_directory: Path


@dataclass(frozen=True, slots=True)
class VideoClipMigrationPlan:
    data_root: Path
    dataset_id: str
    raw_dataset_directory: Path
    processed_directory: Path
    videos: tuple[LegacyVideo, ...]


def _require_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _video_id_for_recording(dataset_id: str, recording_id: str) -> str:
    if recording_id == dataset_id:
        index = 0
    else:
        prefix = f"{dataset_id}_"
        if not recording_id.startswith(prefix):
            raise ValueError(
                f"recording_id {recording_id!r} does not belong to {dataset_id!r}"
            )
        suffix = recording_id[len(prefix) :]
        if not suffix.isdigit() or int(suffix) < 2:
            raise ValueError(
                f"legacy recording suffix must be an integer >= 2, got {recording_id!r}"
            )
        index = int(suffix) - 1
    return f"video_{index:03d}"


def plan_video_clip_migration(
    data_root: Path, dataset_id: str
) -> VideoClipMigrationPlan:
    """Validate every legacy source and destination before any rename occurs."""
    root = data_root.resolve(strict=False)
    if not root.is_absolute() or root == Path(root.anchor):
        raise ValueError("data_root must be a specific absolute directory")
    if _ID_PATTERN.fullmatch(dataset_id) is None:
        raise ValueError(f"invalid dataset_id: {dataset_id!r}")
    multiview = root / "tennis_multivew"
    raw_parent = multiview / "raw"
    processed = multiview / "processed" / dataset_id
    base = raw_parent / dataset_id
    if not base.is_dir():
        raise FileNotFoundError(f"legacy raw dataset directory not found: {base}")
    if any(base.glob("video_*")):
        raise ValueError(f"dataset already contains video_* directories: {base}")

    legacy_raw = [base]
    for candidate in raw_parent.glob(f"{dataset_id}_*"):
        if candidate.is_dir():
            _video_id_for_recording(dataset_id, candidate.name)
            legacy_raw.append(candidate)
    videos = tuple(
        LegacyVideo(
            recording_id=directory.name,
            video_id=_video_id_for_recording(dataset_id, directory.name),
            raw_directory=directory,
        )
        for directory in sorted(
            legacy_raw,
            key=lambda item: int(_video_id_for_recording(dataset_id, item.name)[6:]),
        )
    )
    expected_ids = [f"video_{index:03d}" for index in range(len(videos))]
    actual_ids = [video.video_id for video in videos]
    if actual_ids != expected_ids:
        raise ValueError(
            f"legacy raw recordings must be contiguous, got video IDs {actual_ids}"
        )

    legacy_projects = processed / "projects"
    legacy_dataset = processed / "dataset" / "dataset.json"
    legacy_clips = processed / "dataset" / "clips"
    for required in (legacy_projects, legacy_dataset, legacy_clips):
        if not required.exists():
            raise FileNotFoundError(f"legacy processed path not found: {required}")
    if (processed / "projects.json").exists() or (
        processed / "dataset/videos"
    ).exists():
        raise ValueError(f"new-layout destination already exists under {processed}")

    known_recordings = {video.recording_id for video in videos}
    project_recordings = {
        path.parent.name for path in legacy_projects.glob("*/project.json")
    }
    clip_recordings = {path.name for path in legacy_clips.iterdir() if path.is_dir()}
    unknown = (project_recordings | clip_recordings) - known_recordings
    if unknown:
        raise ValueError(
            f"processed data references unknown recordings: {sorted(unknown)}"
        )
    return VideoClipMigrationPlan(
        data_root=root,
        dataset_id=dataset_id,
        raw_dataset_directory=base,
        processed_directory=processed,
        videos=videos,
    )


def _migrated_project(
    project_path: Path,
    *,
    dataset_id: str,
    video_id: str,
) -> dict[str, Any]:
    payload = _require_mapping(load_json(project_path), name=str(project_path))
    expected = {"version", "recording_id", "sources", "clips"}
    if set(payload) != expected or payload["version"] != 2:
        raise ValueError(f"unsupported legacy project contract: {project_path}")
    sources = payload["sources"]
    if not isinstance(sources, list):
        raise ValueError(f"project sources must be a list: {project_path}")
    migrated_sources = []
    for source in sources:
        block = _require_mapping(source, name=f"source in {project_path}")
        camera_id = str(block["camera_id"])
        migrated_sources.append(
            {
                "path": (
                    Path("tennis_multivew/raw")
                    / dataset_id
                    / video_id
                    / f"{camera_id}.mp4"
                ).as_posix(),
                "camera_id": camera_id,
                "offset_sec": float(block["offset_sec"]),
            }
        )
    return {"sources": migrated_sources, "clips": payload["clips"]}


def _migrate_clip_manifest(
    path: Path,
    *,
    data_root: Path,
    dataset_id: str,
    video_id: str,
) -> dict[str, Any]:
    payload = _require_mapping(load_json(path), name=str(path))
    if payload.get("version") != 1:
        raise ValueError(f"unsupported legacy clip manifest: {path}")
    clip_name = str(payload["clip_name"])
    cameras = payload.get("cameras")
    if not isinstance(cameras, list):
        raise ValueError(f"clip cameras must be a list: {path}")
    for camera in cameras:
        block = _require_mapping(camera, name=f"camera in {path}")
        camera_id = str(block["camera_id"])
        block["source_path"] = str(
            data_root
            / "tennis_multivew/raw"
            / dataset_id
            / video_id
            / f"{camera_id}.mp4"
        )
    payload["version"] = 2
    payload["dataset_id"] = dataset_id
    payload["clip_id"] = f"{video_id}/{clip_name}"
    payload["video_id"] = video_id
    del payload["recording_id"]
    return payload


def apply_video_clip_migration(plan: VideoClipMigrationPlan) -> None:
    """Apply a fully validated same-filesystem migration using directory renames."""
    processed = plan.processed_directory
    legacy_projects = processed / "projects"
    dataset_directory = processed / "dataset"
    legacy_clips = dataset_directory / "clips"

    project_entries: dict[str, dict[str, Any]] = {}
    for video in plan.videos:
        project_path = legacy_projects / video.recording_id / "project.json"
        if project_path.is_file():
            project_entries[video.video_id] = _migrated_project(
                project_path,
                dataset_id=plan.dataset_id,
                video_id=video.video_id,
            )

    dataset_payload = _require_mapping(
        load_json(dataset_directory / "dataset.json"), name="dataset manifest"
    )
    if dataset_payload.get("version") != 1:
        raise ValueError("legacy dataset manifest must have version 1")
    migrated_records = []
    for raw_record in dataset_payload["clips"]:
        record = _require_mapping(raw_record, name="dataset clip record")
        recording_id = str(record["recording_id"])
        video_id = _video_id_for_recording(plan.dataset_id, recording_id)
        clip_name = str(record["clip_name"])
        migrated_records.append(
            {
                **record,
                "clip_id": f"{video_id}/{clip_name}",
                "video_id": video_id,
                "path": f"videos/{video_id}/clips/{clip_name}",
            }
        )
        del migrated_records[-1]["recording_id"]
    migrated_dataset = {
        **dataset_payload,
        "version": 2,
        "dataset_id": plan.dataset_id,
        "clips": migrated_records,
    }

    migrated_clip_manifests: dict[tuple[str, str], dict[str, Any]] = {}
    for video in plan.videos:
        old_clips = legacy_clips / video.recording_id
        if not old_clips.exists():
            continue
        for clip_manifest in old_clips.glob("*/clip.json"):
            migrated_clip_manifests[(video.video_id, clip_manifest.parent.name)] = (
                _migrate_clip_manifest(
                    clip_manifest,
                    data_root=plan.data_root,
                    dataset_id=plan.dataset_id,
                    video_id=video.video_id,
                )
            )

    temporary = plan.raw_dataset_directory.parent / (
        f".{plan.dataset_id}.video-clip-layout-migration"
    )
    if temporary.exists():
        raise ValueError(f"stale migration directory exists: {temporary}")
    plan.raw_dataset_directory.rename(temporary)
    plan.raw_dataset_directory.mkdir()
    try:
        temporary.rename(plan.raw_dataset_directory / "video_000")
        for video in plan.videos[1:]:
            video.raw_directory.rename(plan.raw_dataset_directory / video.video_id)
    except Exception:
        for video in reversed(plan.videos[1:]):
            migrated = plan.raw_dataset_directory / video.video_id
            if migrated.exists() and not video.raw_directory.exists():
                migrated.rename(video.raw_directory)
        migrated_base = plan.raw_dataset_directory / "video_000"
        if migrated_base.exists() and not temporary.exists():
            migrated_base.rename(temporary)
        if plan.raw_dataset_directory.exists() and not any(
            plan.raw_dataset_directory.iterdir()
        ):
            plan.raw_dataset_directory.rmdir()
        if temporary.exists() and not plan.raw_dataset_directory.exists():
            temporary.rename(plan.raw_dataset_directory)
        raise

    videos_directory = dataset_directory / "videos"
    videos_directory.mkdir()
    for video in plan.videos:
        old_clips = legacy_clips / video.recording_id
        if not old_clips.exists():
            continue
        new_video = videos_directory / video.video_id
        new_video.mkdir()
        old_clips.rename(new_video / "clips")
        for clip_manifest in (new_video / "clips").glob("*/clip.json"):
            save_json_atomic(
                migrated_clip_manifests[(video.video_id, clip_manifest.parent.name)],
                clip_manifest,
            )
    legacy_clips.rmdir()
    save_json_atomic(migrated_dataset, dataset_directory / "dataset.json")
    save_json_atomic(
        {
            "version": 1,
            "dataset_id": plan.dataset_id,
            "projects": project_entries,
        },
        processed / "projects.json",
    )
    shutil.rmtree(legacy_projects)


__all__ = [
    "LegacyVideo",
    "VideoClipMigrationPlan",
    "apply_video_clip_migration",
    "plan_video_clip_migration",
]

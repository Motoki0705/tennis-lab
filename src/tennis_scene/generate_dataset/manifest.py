"""Canonical manifests for the incrementally growing real-video dataset.

This module owns both the root ``dataset.json`` and per-clip ``clip.json``
contracts.  Downstream tasks must consume these models instead of defining a
parallel on-disk schema.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.io import load_json, save_json_atomic, utc_now_iso

DATASET_MANIFEST_FILENAME = "dataset.json"
DATASET_SCHEMA_VERSION = 1
CLIP_MANIFEST_FILENAME = "clip.json"
CLIP_SCHEMA_VERSION = 1

_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class DatasetManifestError(RuntimeError):
    """A structured dataset manifest violates the canonical contract."""


class UnsupportedDatasetVersionError(DatasetManifestError):
    """A dataset or clip manifest declares an unsupported version."""


def validate_id_component(value: str, *, field_name: str) -> str:
    """Validate one recording, clip, or camera identifier component."""
    if not isinstance(value, str) or not value:
        raise DatasetManifestError(
            f"{field_name} must be a non-empty string, got {value!r}."
        )
    if not _ID_PATTERN.match(value) or value in {".", ".."}:
        raise DatasetManifestError(
            f"{field_name}={value!r} must match [A-Za-z0-9._-]+ and not be '.' or '..'."
        )
    return value


def split_clip_id(clip_id: str) -> tuple[str, str]:
    """Split and validate the canonical ``<recording_id>/<clip_name>`` id."""
    parts = clip_id.split("/")
    if len(parts) != 2:
        raise DatasetManifestError(
            f"clip_id must be '<recording_id>/<clip_name>', got {clip_id!r}."
        )
    return (
        validate_id_component(parts[0], field_name="recording_id"),
        validate_id_component(parts[1], field_name="clip_name"),
    )


def file_sha256(path: str | Path) -> str:
    """Return the raw hexadecimal SHA-256 used by annotation markers."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetClipRecord:
    """Stable index entry for one exported synchronized clip."""

    clip_id: str
    recording_id: str
    clip_name: str
    path: str
    num_cameras: int
    num_frames: int
    fps: float
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "recording_id": self.recording_id,
            "clip_name": self.clip_name,
            "path": self.path,
            "num_cameras": self.num_cameras,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetClipRecord:
        clip_id = str(data["clip_id"])
        recording_id, clip_name = split_clip_id(clip_id)
        if str(data["recording_id"]) != recording_id:
            raise DatasetManifestError(
                f"recording_id disagrees with clip_id={clip_id!r}."
            )
        if str(data["clip_name"]) != clip_name:
            raise DatasetManifestError(f"clip_name disagrees with clip_id={clip_id!r}.")
        path = str(data["path"])
        expected_path = f"clips/{recording_id}/{clip_name}"
        if path != expected_path:
            raise DatasetManifestError(
                f"clip path {path!r} must be {expected_path!r}."
            )
        return cls(
            clip_id=clip_id,
            recording_id=recording_id,
            clip_name=clip_name,
            path=path,
            num_cameras=int(data["num_cameras"]),
            num_frames=int(data["num_frames"]),
            fps=float(data["fps"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


@dataclass
class DatasetManifest:
    """Dataset inventory; clips can be registered across many sessions."""

    clips: dict[str, DatasetClipRecord] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": DATASET_SCHEMA_VERSION,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "clips": [self.clips[key].to_dict() for key in sorted(self.clips)],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        version = data.get("version")
        if version != DATASET_SCHEMA_VERSION:
            raise UnsupportedDatasetVersionError(
                f"Unsupported dataset version {version!r}; "
                f"expected {DATASET_SCHEMA_VERSION}"
            )
        records = [DatasetClipRecord.from_dict(item) for item in data["clips"]]
        clips = {record.clip_id: record for record in records}
        if len(clips) != len(records):
            raise DatasetManifestError("dataset manifest contains duplicate clip_id values")
        return cls(
            clips=clips,
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
        )

    def save(self, dataset_dir: str | Path) -> Path:
        destination = Path(dataset_dir) / DATASET_MANIFEST_FILENAME
        save_json_atomic(self.to_dict(), destination)
        return destination


def load_dataset_manifest(dataset_dir: str | Path) -> DatasetManifest:
    """Load the required root manifest for a structured dataset."""
    path = Path(dataset_dir) / DATASET_MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"dataset manifest not found: {path}")
    return DatasetManifest.from_dict(load_json(path))


@dataclass(frozen=True)
class ClipManifest:
    """Parsed canonical ``clip.json`` with safe media resolution helpers."""

    clip_dir: Path
    clip_id: str
    recording_id: str
    clip_name: str
    fps: float
    num_frames: int
    width: int
    height: int
    camera_ids: tuple[str, ...]
    video_paths: tuple[str, ...]
    cameras: tuple[dict[str, Any], ...]

    @classmethod
    def load(cls, clip_dir: str | Path) -> ClipManifest:
        root = Path(clip_dir)
        path = root / CLIP_MANIFEST_FILENAME
        if not path.is_file():
            raise DatasetManifestError(f"clip manifest not found: {path}")
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise DatasetManifestError(f"{path} must contain a JSON object.")
        version = payload.get("version")
        if version != CLIP_SCHEMA_VERSION:
            raise UnsupportedDatasetVersionError(
                f"{path} declares version={version!r}; supported: {CLIP_SCHEMA_VERSION}."
            )

        clip_id = payload.get("clip_id")
        if not isinstance(clip_id, str):
            raise DatasetManifestError(f"{path}: missing string clip_id.")
        recording_id, clip_name = split_clip_id(clip_id)
        if payload.get("recording_id") != recording_id or payload.get("clip_name") != clip_name:
            raise DatasetManifestError(
                f"{path}: recording_id/clip_name disagree with clip_id={clip_id!r}."
            )

        fps = payload.get("fps")
        num_frames = payload.get("num_frames")
        width = payload.get("width")
        height = payload.get("height")
        if not isinstance(fps, (int, float)) or fps <= 0:
            raise DatasetManifestError(f"{path}: fps must be positive.")
        if not isinstance(num_frames, int) or num_frames <= 0:
            raise DatasetManifestError(f"{path}: num_frames must be a positive int.")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise DatasetManifestError(f"{path}: width/height must be positive ints.")

        raw_camera_ids = payload.get("camera_ids")
        raw_video_paths = payload.get("video_paths")
        raw_cameras = payload.get("cameras")
        if not isinstance(raw_camera_ids, list) or not raw_camera_ids:
            raise DatasetManifestError(f"{path}: camera_ids must be a non-empty list.")
        if not isinstance(raw_video_paths, list) or len(raw_video_paths) != len(raw_camera_ids):
            raise DatasetManifestError(
                f"{path}: video_paths must align one-to-one with camera_ids."
            )
        if not isinstance(raw_cameras, list):
            raise DatasetManifestError(f"{path}: cameras must be a list.")
        if raw_cameras and len(raw_cameras) != len(raw_camera_ids):
            raise DatasetManifestError(
                f"{path}: non-empty cameras must align with camera_ids."
            )

        camera_ids = tuple(
            validate_id_component(str(value), field_name="camera_id")
            for value in raw_camera_ids
        )
        if len(set(camera_ids)) != len(camera_ids):
            raise DatasetManifestError(f"{path}: camera_ids contains duplicates.")
        video_paths = tuple(str(value) for value in raw_video_paths)
        for relative in video_paths:
            media_path = Path(relative)
            if media_path.is_absolute() or ".." in media_path.parts:
                raise DatasetManifestError(
                    f"{path}: video path {relative!r} escapes the clip directory."
                )

        cameras: list[dict[str, Any]] = []
        for expected_id, block in zip(camera_ids, raw_cameras, strict=True):
            if not isinstance(block, dict) or str(block.get("camera_id")) != expected_id:
                raise DatasetManifestError(
                    f"{path}: camera metadata must follow camera_ids order."
                )
            if block.get("calibrated"):
                raise DatasetManifestError(
                    f"{path}: camera {expected_id!r} declares calibrated=true, but the "
                    "canonical calibrated-camera contract is not defined."
                )
            cameras.append(dict(block))

        return cls(
            clip_dir=root,
            clip_id=clip_id,
            recording_id=recording_id,
            clip_name=clip_name,
            fps=float(fps),
            num_frames=num_frames,
            width=width,
            height=height,
            camera_ids=camera_ids,
            video_paths=video_paths,
            cameras=tuple(cameras),
        )

    @property
    def manifest_path(self) -> Path:
        return self.clip_dir / CLIP_MANIFEST_FILENAME

    def media_path(self, camera_id: str, *, must_exist: bool = True) -> Path:
        """Resolve the video corresponding to ``camera_id``."""
        index = self.camera_index(camera_id)
        path = self.clip_dir / self.video_paths[index]
        if must_exist and not path.is_file():
            raise DatasetManifestError(f"{self.clip_id}: media file missing: {path}")
        return path

    def camera_index(self, camera_id: str) -> int:
        try:
            return self.camera_ids.index(camera_id)
        except ValueError:
            raise DatasetManifestError(
                f"{self.clip_id}: unknown camera_id {camera_id!r}; "
                f"known: {list(self.camera_ids)}."
            ) from None

    def digest(self) -> str:
        """Return the canonical raw SHA-256 digest of ``clip.json``."""
        return file_sha256(self.manifest_path)


def _record_from_clip_manifest(
    dataset_dir: Path, clip_manifest_path: Path
) -> DatasetClipRecord:
    clip_dir = clip_manifest_path.parent.resolve()
    try:
        relative_clip_dir = clip_dir.relative_to(dataset_dir.resolve())
    except ValueError as error:
        raise DatasetManifestError(
            f"clip directory {clip_dir} must be inside dataset {dataset_dir.resolve()}"
        ) from error
    clip_manifest = ClipManifest.load(clip_dir)
    return DatasetClipRecord(
        clip_id=clip_manifest.clip_id,
        recording_id=clip_manifest.recording_id,
        clip_name=clip_manifest.clip_name,
        path=str(relative_clip_dir),
        num_cameras=len(clip_manifest.camera_ids),
        num_frames=clip_manifest.num_frames,
        fps=clip_manifest.fps,
        width=clip_manifest.width,
        height=clip_manifest.height,
    )


def register_exported_clip(
    dataset_dir: str | Path,
    clip_manifest_path: str | Path,
    *,
    allow_replace: bool = False,
) -> DatasetManifest:
    """Atomically add or verify one completed clip in ``dataset.json``."""
    root = Path(dataset_dir)
    manifest_path = root / DATASET_MANIFEST_FILENAME
    dataset = (
        DatasetManifest.from_dict(load_json(manifest_path))
        if manifest_path.exists()
        else DatasetManifest()
    )
    record = _record_from_clip_manifest(root, Path(clip_manifest_path))
    existing = dataset.clips.get(record.clip_id)
    if existing is not None and existing != record and not allow_replace:
        raise DatasetManifestError(
            f"clip_id collision for {record.clip_id!r}: "
            f"existing={existing.to_dict()}, new={record.to_dict()}"
        )
    dataset.clips[record.clip_id] = record
    dataset.updated_at = utc_now_iso()
    dataset.save(root)
    return dataset


__all__ = [
    "CLIP_MANIFEST_FILENAME",
    "CLIP_SCHEMA_VERSION",
    "DATASET_MANIFEST_FILENAME",
    "DATASET_SCHEMA_VERSION",
    "ClipManifest",
    "DatasetClipRecord",
    "DatasetManifestError",
    "DatasetManifest",
    "UnsupportedDatasetVersionError",
    "file_sha256",
    "load_dataset_manifest",
    "register_exported_clip",
    "split_clip_id",
    "validate_id_component",
]

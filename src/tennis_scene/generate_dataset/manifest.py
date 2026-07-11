"""Versioned manifest for an incrementally growing real-video dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.io import load_json, save_json_atomic, utc_now_iso

DATASET_MANIFEST_FILENAME = "dataset.json"
DATASET_SCHEMA_VERSION = 1


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
        return cls(
            clip_id=str(data["clip_id"]),
            recording_id=str(data["recording_id"]),
            clip_name=str(data["clip_name"]),
            path=str(data["path"]),
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
            raise ValueError(
                f"Unsupported dataset version {version!r}; "
                f"expected {DATASET_SCHEMA_VERSION}"
            )
        records = [DatasetClipRecord.from_dict(item) for item in data["clips"]]
        clips = {record.clip_id: record for record in records}
        if len(clips) != len(records):
            raise ValueError("dataset manifest contains duplicate clip_id values")
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


def _record_from_clip_manifest(
    dataset_dir: Path, clip_manifest_path: Path
) -> DatasetClipRecord:
    clip_manifest = load_json(clip_manifest_path)
    clip_dir = clip_manifest_path.parent.resolve()
    try:
        relative_clip_dir = clip_dir.relative_to(dataset_dir.resolve())
    except ValueError as error:
        raise ValueError(
            f"clip directory {clip_dir} must be inside dataset {dataset_dir.resolve()}"
        ) from error

    camera_ids = clip_manifest["camera_ids"]
    return DatasetClipRecord(
        clip_id=str(clip_manifest["clip_id"]),
        recording_id=str(clip_manifest["recording_id"]),
        clip_name=str(clip_manifest["clip_name"]),
        path=str(relative_clip_dir),
        num_cameras=len(camera_ids),
        num_frames=int(clip_manifest["num_frames"]),
        fps=float(clip_manifest["fps"]),
        width=int(clip_manifest["width"]),
        height=int(clip_manifest["height"]),
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
        raise ValueError(
            f"clip_id collision for {record.clip_id!r}: "
            f"existing={existing.to_dict()}, new={record.to_dict()}"
        )
    dataset.clips[record.clip_id] = record
    dataset.updated_at = utc_now_iso()
    dataset.save(root)
    return dataset


__all__ = [
    "DATASET_MANIFEST_FILENAME",
    "DATASET_SCHEMA_VERSION",
    "DatasetClipRecord",
    "DatasetManifest",
    "load_dataset_manifest",
    "register_exported_clip",
]

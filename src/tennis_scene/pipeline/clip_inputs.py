"""Shared media binding, structured-clip selection and input integrity checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.source import build_clip_source
from src.utils.checksum import dual_sha256

if TYPE_CHECKING:
    from src.tennis_scene.generate_dataset.manifest import DatasetClipRecord


@dataclass(frozen=True)
class InputSnapshot:
    files: tuple[tuple[Path, str], ...]

    def assert_unchanged(self) -> None:
        if any(dual_sha256(path) != digest for path, digest in self.files):
            raise ValueError("Clip inputs changed during reconstruction")


@dataclass(frozen=True)
class SourceInputs:
    source: ClipSource
    snapshot: InputSnapshot


def prepare_clip_source(paths: Sequence[Path], camera_ids: Sequence[str], *, max_frames: int | None = None,
                        clip_id: str | None = None) -> SourceInputs:
    """Validate actual media frame counts, FPS and synchronization once."""
    source = build_clip_source(paths, camera_ids, max_frames=max_frames, clip_id=clip_id)
    return SourceInputs(source, InputSnapshot(tuple((v.path, v.sha256) for v in source.videos)))


@dataclass(frozen=True)
class DatasetClipInputs:
    record: DatasetClipRecord
    manifest_path: Path
    video_paths: tuple[Path, ...]
    camera_ids: tuple[str, ...]
    manifest_sha256: str
    media_sha256: Mapping[str, str]
    snapshot: InputSnapshot

    @property
    def clip_dir(self) -> Path:
        return self.manifest_path.parent


@dataclass(frozen=True)
class PreparedDatasetClip:
    inputs: DatasetClipInputs
    media: SourceInputs


def prepare_video_inputs(dataset_dir: Path, video_id: str,
                         clip_ids: Sequence[str] | None = None) -> tuple[str, tuple[PreparedDatasetClip, ...]]:
    """Freeze and validate the whole video selection before creating outputs."""
    from src.tennis_scene.generate_dataset.manifest import (
        ClipManifest,
        load_dataset_manifest,
        validate_id_component,
    )

    if not dataset_dir.is_absolute() or dataset_dir.resolve(strict=False) != dataset_dir:
        raise ValueError("dataset_dir must be an absolute normalized Path from PathResolver.")
    validate_id_component(video_id, field_name="video_id")
    dataset = load_dataset_manifest(dataset_dir)
    selected = (tuple(key for key in sorted(dataset.clips) if dataset.clips[key].video_id == video_id)
                if clip_ids is None else tuple(clip_ids))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Video selection must contain nonempty, unique clip IDs")
    unknown = set(selected) - dataset.clips.keys()
    if unknown:
        raise ValueError(f"Unknown clip IDs: {sorted(unknown)}")
    if any(dataset.clips[key].video_id != video_id for key in selected):
        raise ValueError("Video selection contains clips from another video")
    prepared = []
    for key in selected:
        record = dataset.clips[key]
        bound = resolve_dataset_clip(dataset_dir, record)
        manifest = ClipManifest.load(bound.clip_dir)
        if manifest.dataset_id != dataset.dataset_id:
            raise ValueError(f"Clip belongs to another dataset: {key}")
        if len(set(path.resolve() for path in bound.video_paths)) != len(bound.camera_ids):
            raise ValueError(f"Camera media must be one-to-one: {key}")
        media = prepare_clip_source(bound.video_paths, bound.camera_ids, clip_id=key)
        source = media.source
        if ((source.num_frames, *source.size) != (record.num_frames, record.width, record.height)
                or abs(source.fps - record.fps) > 1e-6):
            raise ValueError(f"Actual media timeline disagrees with dataset record: {key}")
        bound.snapshot.assert_unchanged()
        prepared.append(PreparedDatasetClip(bound, media))
    return dataset.dataset_id, tuple(prepared)


def select_dataset_clips(dataset_dir: Path, clip_ids: Sequence[str] | None = None) -> tuple[DatasetClipRecord, ...]:
    """Resolve an explicit dataset selection without constructing any models."""
    from src.tennis_scene.generate_dataset.manifest import load_dataset_manifest

    if not dataset_dir.is_absolute() or dataset_dir.resolve(strict=False) != dataset_dir:
        raise ValueError("dataset_dir must be an absolute normalized Path from PathResolver.")
    dataset = load_dataset_manifest(dataset_dir)
    selected = sorted(dataset.clips) if clip_ids is None else list(clip_ids)
    unknown = [clip_id for clip_id in selected if clip_id not in dataset.clips]
    if unknown:
        raise KeyError(f"clip_id not found in dataset: {unknown[0]!r}")
    return tuple(dataset.clips[clip_id] for clip_id in selected)


def resolve_dataset_clip(dataset_dir: Path, record: DatasetClipRecord) -> DatasetClipInputs:
    """Bind a manifest and existing media, and pin their hashes before execution.

    The record and clip manifest must describe the same timeline. Actual media
    synchronization is checked by prepare_clip_source at the pipeline boundary;
    this step also works for callers publishing an already reconstructed scene.
    """
    from src.tennis_scene.generate_dataset.manifest import ClipManifest

    clip_dir = dataset_dir / record.path
    manifest_path = clip_dir / "clip.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"clip manifest not found: {manifest_path}")
    manifest_hash = dual_sha256(manifest_path)
    manifest = ClipManifest.load(clip_dir)
    if manifest.clip_id != record.clip_id:
        raise ValueError(f"clip_id mismatch: dataset has {record.clip_id!r}, clip manifest has {manifest.clip_id!r}")
    if ((len(manifest.camera_ids), manifest.num_frames, manifest.width, manifest.height)
            != (record.num_cameras, record.num_frames, record.width, record.height)
            or abs(manifest.fps - record.fps) > 1e-6):
        raise ValueError(f"Clip manifest timeline disagrees with dataset record: {record.clip_id}")
    paths = tuple(manifest.media_path(camera) for camera in manifest.camera_ids)
    media_hashes = {camera: dual_sha256(path) for camera, path in zip(manifest.camera_ids, paths, strict=True)}
    snapshot = InputSnapshot(((manifest_path, manifest_hash),
                             *((path, media_hashes[camera]) for camera, path in zip(manifest.camera_ids, paths, strict=True))))
    snapshot.assert_unchanged()
    return DatasetClipInputs(record, manifest_path, paths, manifest.camera_ids, manifest_hash, media_hashes, snapshot)

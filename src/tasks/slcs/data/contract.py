"""Reader for the issue #634 structured real-clip dataset contract.

Layout (relative to ``dataset_root``)::

    dataset.json
    clips/<recording_id>/<clip_name>/
        clip.json
        media/<camera_id>.mp4
        annotations/tennis_scene/
            scene.npz
            scene.metadata.json
            annotation.json          # completion marker (required)
            pipeline_config.yaml     # optional provenance snapshot
        annotations/dino_v3/         # see src.tasks.slcs.data.dino_tokens
            annotation.json
            <camera_id>.npz

Contract rules enforced here (no silent fallback):

- ``format_version`` must match a supported version, otherwise
  :class:`UnsupportedFormatVersionError` is raised.
- ``clip_id`` must be exactly ``<recording_id>/<clip_name>`` with both parts
  restricted to ``[A-Za-z0-9._-]`` (no path traversal).
- An annotation directory without its ``annotation.json`` marker is treated as
  incomplete and raises :class:`IncompleteAnnotationError` when loaded.
- ``scene.npz`` array shapes are validated against ``clip.json``
  (frame count, camera count, image size) and against the shape spec recorded
  in the completion marker.
- Camera calibration is a separate, future contract: a camera block declaring
  ``calibrated: true`` is rejected instead of being partially interpreted.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.tennis_scene.io import SceneResult
from src.utils.io import load_json

DATASET_INDEX_NAME = "dataset.json"
CLIP_MANIFEST_NAME = "clip.json"
CLIPS_DIR_NAME = "clips"
MEDIA_DIR_NAME = "media"
ANNOTATIONS_DIR_NAME = "annotations"
TENNIS_SCENE_DIR_NAME = "tennis_scene"
ANNOTATION_MARKER_NAME = "annotation.json"
SCENE_NPZ_NAME = "scene.npz"

DATASET_FORMAT_VERSION = 1
CLIP_FORMAT_VERSION = 1
TENNIS_SCENE_ANNOTATION_VERSION = 1
TENNIS_SCENE_ANNOTATION_KIND = "tennis_scene"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

# SceneResult arrays SLCS training cannot run without.
REQUIRED_SCENE_ARRAYS: tuple[str, ...] = (
    "court_kp",
    "court_vis",
    "player_position",
    "player_yaw",
    "ball_uv",
    "ball_vis",
    "ball_3d",
    "human_kp_2d",
    "human_kp_vis",
)


class DatasetContractError(RuntimeError):
    """A dataset/clip/annotation file violates the issue #634 contract."""


class UnsupportedFormatVersionError(DatasetContractError):
    """A manifest declares a format version this reader does not support."""


class IncompleteAnnotationError(DatasetContractError):
    """An annotation directory has no completion marker (generation unfinished)."""


def validate_id_component(value: str, *, field_name: str) -> str:
    """Validate a single id component (``recording_id`` or ``clip_name``)."""
    if not isinstance(value, str) or not value:
        raise DatasetContractError(f"{field_name} must be a non-empty string, got {value!r}.")
    if not _ID_PATTERN.match(value):
        raise DatasetContractError(
            f"{field_name}={value!r} contains characters outside [A-Za-z0-9._-]."
        )
    if value in {".", ".."}:
        raise DatasetContractError(f"{field_name}={value!r} is a path traversal component.")
    return value


def split_clip_id(clip_id: str) -> tuple[str, str]:
    """Split ``<recording_id>/<clip_name>`` and validate both components."""
    parts = clip_id.split("/")
    if len(parts) != 2:
        raise DatasetContractError(
            f"clip_id must be '<recording_id>/<clip_name>', got {clip_id!r}."
        )
    recording_id = validate_id_component(parts[0], field_name="recording_id")
    clip_name = validate_id_component(parts[1], field_name="clip_name")
    return recording_id, clip_name


def _require_version(payload: dict[str, Any], *, expected: int, source: Path) -> None:
    version = payload.get("format_version")
    if version != expected:
        raise UnsupportedFormatVersionError(
            f"{source} declares format_version={version!r}; this reader supports {expected}."
        )


def file_sha256(path: Path) -> str:
    """Return ``sha256:<hex>`` digest of a file's bytes."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class ClipRef:
    """One entry of ``dataset.json``."""

    clip_id: str
    recording_id: str
    clip_name: str
    path: str  # relative to dataset_root, POSIX form

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: Path) -> ClipRef:
        clip_id = data.get("clip_id")
        if not isinstance(clip_id, str):
            raise DatasetContractError(f"{source}: clip entry without string clip_id: {data!r}")
        recording_id, clip_name = split_clip_id(clip_id)
        declared_recording = data.get("recording_id", recording_id)
        declared_name = data.get("clip_name", clip_name)
        if declared_recording != recording_id or declared_name != clip_name:
            raise DatasetContractError(
                f"{source}: clip entry fields disagree with clip_id={clip_id!r}: "
                f"recording_id={declared_recording!r}, clip_name={declared_name!r}."
            )
        path = data.get("path", f"{CLIPS_DIR_NAME}/{recording_id}/{clip_name}")
        expected_path = f"{CLIPS_DIR_NAME}/{recording_id}/{clip_name}"
        if path != expected_path:
            raise DatasetContractError(
                f"{source}: clip path {path!r} must be {expected_path!r} (derived from clip_id)."
            )
        return cls(
            clip_id=clip_id,
            recording_id=recording_id,
            clip_name=clip_name,
            path=path,
        )


@dataclass(frozen=True)
class DatasetIndex:
    """Parsed ``dataset.json``: the appendable index of all clips."""

    root: Path
    clips: tuple[ClipRef, ...]

    @classmethod
    def load(cls, dataset_root: str | Path) -> DatasetIndex:
        root = Path(dataset_root)
        index_path = root / DATASET_INDEX_NAME
        if not index_path.is_file():
            raise DatasetContractError(f"dataset index not found: {index_path}")
        payload = load_json(index_path)
        if not isinstance(payload, dict):
            raise DatasetContractError(f"{index_path} must contain a JSON object.")
        _require_version(payload, expected=DATASET_FORMAT_VERSION, source=index_path)
        raw_clips = payload.get("clips")
        if not isinstance(raw_clips, list):
            raise DatasetContractError(f"{index_path} must contain a 'clips' list.")
        clips = tuple(ClipRef.from_dict(entry, source=index_path) for entry in raw_clips)
        seen: set[str] = set()
        for ref in clips:
            if ref.clip_id in seen:
                raise DatasetContractError(f"{index_path}: duplicate clip_id {ref.clip_id!r}.")
            seen.add(ref.clip_id)
        return cls(root=root, clips=clips)

    def clip_dir(self, ref: ClipRef) -> Path:
        return self.root / ref.path

    def recording_ids(self) -> tuple[str, ...]:
        """Unique recording ids in index order."""
        out: list[str] = []
        for ref in self.clips:
            if ref.recording_id not in out:
                out.append(ref.recording_id)
        return tuple(out)


@dataclass(frozen=True)
class ClipManifest:
    """Parsed ``clip.json``: the immutable per-clip manifest."""

    clip_dir: Path
    clip_id: str
    recording_id: str
    clip_name: str
    fps: float
    num_frames: int
    width: int
    height: int
    camera_ids: tuple[str, ...]
    media: dict[str, str] = field(default_factory=dict)  # camera_id -> relative path
    source: dict[str, Any] = field(default_factory=dict)  # opaque provenance

    @classmethod
    def load(cls, clip_dir: str | Path) -> ClipManifest:
        clip_dir = Path(clip_dir)
        manifest_path = clip_dir / CLIP_MANIFEST_NAME
        if not manifest_path.is_file():
            raise DatasetContractError(f"clip manifest not found: {manifest_path}")
        payload = load_json(manifest_path)
        if not isinstance(payload, dict):
            raise DatasetContractError(f"{manifest_path} must contain a JSON object.")
        _require_version(payload, expected=CLIP_FORMAT_VERSION, source=manifest_path)

        clip_id = payload.get("clip_id")
        if not isinstance(clip_id, str):
            raise DatasetContractError(f"{manifest_path}: missing string clip_id.")
        recording_id, clip_name = split_clip_id(clip_id)

        fps = payload.get("fps")
        num_frames = payload.get("num_frames")
        width = payload.get("width")
        height = payload.get("height")
        if not isinstance(fps, (int, float)) or fps <= 0:
            raise DatasetContractError(f"{manifest_path}: fps must be a positive number.")
        if not isinstance(num_frames, int) or num_frames <= 0:
            raise DatasetContractError(f"{manifest_path}: num_frames must be a positive int.")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise DatasetContractError(f"{manifest_path}: width/height must be positive ints.")

        camera_ids_raw = payload.get("camera_ids")
        if not isinstance(camera_ids_raw, list) or not camera_ids_raw:
            raise DatasetContractError(f"{manifest_path}: camera_ids must be a non-empty list.")
        camera_ids = tuple(
            validate_id_component(str(cam), field_name="camera_id") for cam in camera_ids_raw
        )
        if len(set(camera_ids)) != len(camera_ids):
            raise DatasetContractError(f"{manifest_path}: camera_ids contains duplicates.")

        media_raw = payload.get("media")
        if not isinstance(media_raw, dict):
            raise DatasetContractError(f"{manifest_path}: media must be a mapping.")
        media = {str(k): str(v) for k, v in media_raw.items()}
        if set(media) != set(camera_ids):
            raise DatasetContractError(
                f"{manifest_path}: media keys {sorted(media)} must equal "
                f"camera_ids {sorted(camera_ids)}."
            )
        for cam, rel in media.items():
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise DatasetContractError(
                    f"{manifest_path}: media path {rel!r} for camera {cam!r} escapes the clip dir."
                )

        cameras_raw = payload.get("cameras", {})
        if not isinstance(cameras_raw, dict):
            raise DatasetContractError(f"{manifest_path}: cameras must be a mapping when present.")
        for cam, block in cameras_raw.items():
            if isinstance(block, dict) and block.get("calibrated"):
                raise DatasetContractError(
                    f"{manifest_path}: camera {cam!r} declares calibrated=true, but the "
                    "calibrated-camera contract is not supported by this reader yet. "
                    "Refusing to guess intrinsics/extrinsics."
                )

        source = payload.get("source", {})
        if not isinstance(source, dict):
            raise DatasetContractError(f"{manifest_path}: source must be a mapping when present.")

        return cls(
            clip_dir=clip_dir,
            clip_id=clip_id,
            recording_id=recording_id,
            clip_name=clip_name,
            fps=float(fps),
            num_frames=num_frames,
            width=width,
            height=height,
            camera_ids=camera_ids,
            media=media,
            source=source,
        )

    @property
    def manifest_path(self) -> Path:
        return self.clip_dir / CLIP_MANIFEST_NAME

    def media_path(self, camera_id: str, *, must_exist: bool = True) -> Path:
        """Resolve the media file for a camera, optionally requiring existence."""
        if camera_id not in self.media:
            raise DatasetContractError(
                f"{self.clip_id}: unknown camera_id {camera_id!r}; known: {list(self.camera_ids)}."
            )
        path = self.clip_dir / self.media[camera_id]
        if must_exist and not path.is_file():
            raise DatasetContractError(f"{self.clip_id}: media file missing: {path}")
        return path

    def camera_index(self, camera_id: str) -> int:
        """Index of ``camera_id`` in the manifest camera order (= scene camera axis)."""
        try:
            return self.camera_ids.index(camera_id)
        except ValueError:
            raise DatasetContractError(
                f"{self.clip_id}: unknown camera_id {camera_id!r}; known: {list(self.camera_ids)}."
            ) from None

    def digest(self) -> str:
        """Digest of the manifest file bytes (provenance anchor for annotations)."""
        return file_sha256(self.manifest_path)


def tennis_scene_dir(clip_dir: Path) -> Path:
    return clip_dir / ANNOTATIONS_DIR_NAME / TENNIS_SCENE_DIR_NAME


def has_tennis_scene_annotation(clip_dir: Path) -> bool:
    """Whether a *complete* tennis_scene annotation exists for the clip."""
    return (tennis_scene_dir(clip_dir) / ANNOTATION_MARKER_NAME).is_file()


def _validate_marker(marker: dict[str, Any], *, source: Path) -> dict[str, Any]:
    _require_version(marker, expected=TENNIS_SCENE_ANNOTATION_VERSION, source=source)
    kind = marker.get("kind")
    if kind != TENNIS_SCENE_ANNOTATION_KIND:
        raise DatasetContractError(
            f"{source}: kind={kind!r}, expected {TENNIS_SCENE_ANNOTATION_KIND!r}."
        )
    arrays = marker.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        raise DatasetContractError(f"{source}: marker must record an 'arrays' shape spec.")
    return arrays


def _validate_scene_against_manifest(scene: SceneResult, manifest: ClipManifest) -> None:
    if scene.num_frames != manifest.num_frames:
        raise DatasetContractError(
            f"{manifest.clip_id}: scene num_frames={scene.num_frames} != "
            f"manifest num_frames={manifest.num_frames}."
        )
    if abs(scene.fps - manifest.fps) > 1e-3:
        raise DatasetContractError(
            f"{manifest.clip_id}: scene fps={scene.fps} != manifest fps={manifest.fps}."
        )
    if (scene.width, scene.height) != (manifest.width, manifest.height):
        raise DatasetContractError(
            f"{manifest.clip_id}: scene size {(scene.width, scene.height)} != "
            f"manifest size {(manifest.width, manifest.height)}."
        )
    n_cams = len(manifest.camera_ids)
    if scene.court_kp.shape[0] != n_cams:
        raise DatasetContractError(
            f"{manifest.clip_id}: scene camera axis N={scene.court_kp.shape[0]} != "
            f"manifest camera count {n_cams}."
        )


def _validate_scene_arrays(
    scene: SceneResult, arrays_spec: dict[str, Any], *, clip_id: str
) -> None:
    for name in REQUIRED_SCENE_ARRAYS:
        value = getattr(scene, name, None)
        if value is None:
            raise DatasetContractError(
                f"{clip_id}: scene.npz is missing required array {name!r} "
                f"(required for SLCS: {REQUIRED_SCENE_ARRAYS})."
            )
        if name not in arrays_spec:
            raise DatasetContractError(
                f"{clip_id}: completion marker does not record array {name!r}."
            )
        spec = arrays_spec[name]
        expected_shape = tuple(spec.get("shape", ()))
        actual_shape = tuple(np.asarray(value).shape)
        if expected_shape != actual_shape:
            raise DatasetContractError(
                f"{clip_id}: array {name!r} shape {actual_shape} != "
                f"marker-recorded shape {expected_shape}."
            )


def load_tennis_scene_annotation(
    manifest: ClipManifest,
    *,
    verify_manifest_digest: bool = True,
) -> SceneResult:
    """Load and validate the tennis_scene pseudo-annotation of a clip.

    Raises:
        IncompleteAnnotationError: no completion marker (generation unfinished).
        DatasetContractError: shape/metadata mismatch with the clip manifest.
    """
    ann_dir = tennis_scene_dir(manifest.clip_dir)
    marker_path = ann_dir / ANNOTATION_MARKER_NAME
    if not marker_path.is_file():
        raise IncompleteAnnotationError(
            f"{manifest.clip_id}: tennis_scene annotation has no completion marker "
            f"({marker_path}); treating as incomplete."
        )
    marker = load_json(marker_path)
    if not isinstance(marker, dict):
        raise DatasetContractError(f"{marker_path} must contain a JSON object.")
    arrays_spec = _validate_marker(marker, source=marker_path)

    if verify_manifest_digest:
        recorded = marker.get("input_manifest_digest")
        actual = manifest.digest()
        if recorded != actual:
            raise DatasetContractError(
                f"{manifest.clip_id}: annotation was generated from a different clip.json "
                f"(marker digest {recorded!r} != current {actual!r})."
            )

    scene_path = ann_dir / SCENE_NPZ_NAME
    if not scene_path.is_file():
        raise DatasetContractError(f"{manifest.clip_id}: scene archive missing: {scene_path}")
    scene = SceneResult.load(scene_path)
    _validate_scene_against_manifest(scene, manifest)
    _validate_scene_arrays(scene, arrays_spec, clip_id=manifest.clip_id)
    return scene


__all__ = [
    "ANNOTATION_MARKER_NAME",
    "CLIP_FORMAT_VERSION",
    "CLIP_MANIFEST_NAME",
    "CLIPS_DIR_NAME",
    "DATASET_FORMAT_VERSION",
    "DATASET_INDEX_NAME",
    "MEDIA_DIR_NAME",
    "REQUIRED_SCENE_ARRAYS",
    "SCENE_NPZ_NAME",
    "TENNIS_SCENE_ANNOTATION_KIND",
    "TENNIS_SCENE_ANNOTATION_VERSION",
    "ClipManifest",
    "ClipRef",
    "DatasetContractError",
    "DatasetIndex",
    "IncompleteAnnotationError",
    "UnsupportedFormatVersionError",
    "file_sha256",
    "has_tennis_scene_annotation",
    "load_tennis_scene_annotation",
    "split_clip_id",
    "tennis_scene_dir",
    "validate_id_component",
]

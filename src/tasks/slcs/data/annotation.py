"""Strict SLCS reader for canonical tennis-scene pseudo annotations.

Dataset and clip manifests remain owned by
``src.tennis_scene.generate_dataset.manifest``.  This module owns only the
additional completion, archive, and array constraints required before a clip
can enter the SLCS data pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    DatasetClipRecord,
    DatasetManifestError,
    UnsupportedDatasetVersionError,
    load_dataset_manifest,
)
from src.tennis_scene.generate_dataset.pseudo_annotation import (
    ANNOTATION_SCHEMA_VERSION,
)
from src.tennis_scene.schema import SceneResult
from src.utils.io import load_json

SLCS_ANNOTATION_FILENAME = "annotation.json"
SLCS_SCENE_ARCHIVE_FILENAME = "scene.npz"
_SLCS_ANNOTATION_RELATIVE_DIR = Path("annotations") / "tennis_scene"

# SceneResult arrays without which SLCS training has no defined input/target.
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


class IncompleteAnnotationError(DatasetManifestError):
    """A requested annotation has no completion marker."""


@dataclass(frozen=True, slots=True)
class SLCSDataIndex:
    """Canonical manifest records together with the resolved dataset root."""

    root: Path
    clips: tuple[DatasetClipRecord, ...]

    @classmethod
    def load(cls, dataset_root: str | Path) -> SLCSDataIndex:
        root = Path(dataset_root)
        manifest = load_dataset_manifest(root)
        return cls(root=root, clips=tuple(manifest.clips.values()))

    def clip_dir(self, record: DatasetClipRecord) -> Path:
        """Resolve a canonical role-relative record path under this dataset."""
        return self.root / str(record.path)

    def recording_ids(self) -> tuple[str, ...]:
        """Return unique recording identifiers in manifest order."""
        return tuple(dict.fromkeys(record.recording_id for record in self.clips))


def slcs_annotation_dir(clip_dir: Path) -> Path:
    """Return the SLCS-consumed tennis-scene annotation directory."""
    return clip_dir / _SLCS_ANNOTATION_RELATIVE_DIR


def has_slcs_annotation(clip_dir: Path) -> bool:
    """Return whether the clip has a completed SLCS-consumable annotation."""
    return (slcs_annotation_dir(clip_dir) / SLCS_ANNOTATION_FILENAME).is_file()


def _validate_marker(marker: dict[str, Any], *, source: Path) -> dict[str, Any]:
    version = marker.get("version")
    if version != ANNOTATION_SCHEMA_VERSION:
        raise UnsupportedDatasetVersionError(
            f"{source} declares version={version!r}; supported: "
            f"{ANNOTATION_SCHEMA_VERSION}."
        )
    arrays = marker.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        raise DatasetManifestError(
            f"{source}: marker must record an 'arrays' shape spec."
        )
    return arrays


def _validate_scene_against_manifest(
    scene: SceneResult, manifest: ClipManifest
) -> None:
    if scene.num_frames != manifest.num_frames:
        raise DatasetManifestError(
            f"{manifest.clip_id}: scene num_frames={scene.num_frames} != "
            f"manifest num_frames={manifest.num_frames}."
        )
    if abs(scene.fps - manifest.fps) > 1e-3:
        raise DatasetManifestError(
            f"{manifest.clip_id}: scene fps={scene.fps} != "
            f"manifest fps={manifest.fps}."
        )
    if (scene.width, scene.height) != (manifest.width, manifest.height):
        raise DatasetManifestError(
            f"{manifest.clip_id}: scene size {(scene.width, scene.height)} != "
            f"manifest size {(manifest.width, manifest.height)}."
        )
    camera_count = len(manifest.camera_ids)
    if scene.court_kp.shape[0] != camera_count:
        raise DatasetManifestError(
            f"{manifest.clip_id}: scene camera axis N={scene.court_kp.shape[0]} != "
            f"manifest camera count {camera_count}."
        )


def _validate_scene_arrays(
    scene: SceneResult, arrays_spec: dict[str, Any], *, clip_id: str
) -> None:
    for name in REQUIRED_SCENE_ARRAYS:
        value = getattr(scene, name, None)
        if value is None:
            raise DatasetManifestError(
                f"{clip_id}: scene archive is missing required array {name!r} "
                f"(required for SLCS: {REQUIRED_SCENE_ARRAYS})."
            )
        if name not in arrays_spec:
            raise DatasetManifestError(
                f"{clip_id}: completion marker does not record array {name!r}."
            )
        spec = arrays_spec[name]
        if not isinstance(spec, dict):
            raise DatasetManifestError(
                f"{clip_id}: marker spec for array {name!r} must be an object."
            )
        expected_shape = tuple(spec.get("shape", ()))
        actual_shape = tuple(np.asarray(value).shape)
        if expected_shape != actual_shape:
            raise DatasetManifestError(
                f"{clip_id}: array {name!r} shape {actual_shape} != "
                f"marker-recorded shape {expected_shape}."
            )


def load_slcs_annotation(
    manifest: ClipManifest,
    *,
    verify_manifest_digest: bool = True,
) -> SceneResult:
    """Load a completed annotation and enforce every SLCS data constraint."""
    annotation_dir = slcs_annotation_dir(manifest.clip_dir)
    marker_path = annotation_dir / SLCS_ANNOTATION_FILENAME
    if not marker_path.is_file():
        raise IncompleteAnnotationError(
            f"{manifest.clip_id}: tennis_scene annotation has no completion marker "
            f"({marker_path}); treating as incomplete."
        )
    marker = load_json(marker_path)
    if not isinstance(marker, dict):
        raise DatasetManifestError(f"{marker_path} must contain a JSON object.")
    arrays_spec = _validate_marker(marker, source=marker_path)

    if verify_manifest_digest:
        recorded = marker.get("clip_manifest_sha256")
        actual = manifest.digest()
        if recorded != actual:
            raise DatasetManifestError(
                f"{manifest.clip_id}: annotation was generated from a different "
                f"clip.json (marker digest {recorded!r} != current {actual!r})."
            )

    scene_path = annotation_dir / SLCS_SCENE_ARCHIVE_FILENAME
    if not scene_path.is_file():
        raise DatasetManifestError(
            f"{manifest.clip_id}: scene archive missing: {scene_path}"
        )
    scene = load_scene_result(scene_path)
    _validate_scene_against_manifest(scene, manifest)
    _validate_scene_arrays(scene, arrays_spec, clip_id=manifest.clip_id)
    return scene


__all__ = [
    "IncompleteAnnotationError",
    "REQUIRED_SCENE_ARRAYS",
    "SLCS_ANNOTATION_FILENAME",
    "SLCS_SCENE_ARCHIVE_FILENAME",
    "SLCSDataIndex",
    "has_slcs_annotation",
    "load_slcs_annotation",
    "slcs_annotation_dir",
]

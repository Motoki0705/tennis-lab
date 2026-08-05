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

- Canonical ``version`` fields must match the versions owned by
  :mod:`src.tennis_scene.generate_dataset`, otherwise
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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np

from src.tennis_scene.generate_dataset.manifest import (
    CLIP_MANIFEST_FILENAME,
    DATASET_MANIFEST_FILENAME,
    ClipManifest,
    DatasetManifestError,
    UnsupportedDatasetVersionError,
    file_sha256,
    load_dataset_manifest,
    split_clip_id,
    validate_id_component,
)
from src.tennis_scene.generate_dataset.pseudo_annotation import (
    ANNOTATION_SCHEMA_VERSION,
)
from src.tennis_scene.io import SceneResult
from src.utils.io import load_json

DATASET_INDEX_NAME = DATASET_MANIFEST_FILENAME
CLIP_MANIFEST_NAME = CLIP_MANIFEST_FILENAME
CLIPS_DIR_NAME = "clips"
MEDIA_DIR_NAME = "media"
ANNOTATIONS_DIR_NAME = "annotations"
TENNIS_SCENE_DIR_NAME = "tennis_scene"
ANNOTATION_MARKER_NAME = "annotation.json"
SCENE_NPZ_NAME = "scene.npz"

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


DatasetContractError: TypeAlias = DatasetManifestError
UnsupportedFormatVersionError: TypeAlias = UnsupportedDatasetVersionError


class IncompleteAnnotationError(DatasetContractError):
    """An annotation directory has no completion marker (generation unfinished)."""


class ClipRef(Protocol):
    """Read-only dataset-index record fields consumed by SLCS."""

    @property
    def clip_id(self) -> str:
        """Canonical ``<recording_id>/<clip_name>`` identifier."""
        ...

    @property
    def recording_id(self) -> str:
        """Recording group used to prevent split leakage."""
        ...

    @property
    def path(self) -> str:
        """Canonical role-relative clip directory."""
        ...


@dataclass(frozen=True)
class DatasetIndex:
    """Parsed ``dataset.json``: the appendable index of all clips."""

    root: Path
    clips: tuple[ClipRef, ...]

    @classmethod
    def load(cls, dataset_root: str | Path) -> DatasetIndex:
        root = Path(dataset_root)
        manifest = load_dataset_manifest(root)
        return cls(root=root, clips=tuple(manifest.clips.values()))

    def clip_dir(self, ref: ClipRef) -> Path:
        return self.root / ref.path

    def recording_ids(self) -> tuple[str, ...]:
        """Unique recording ids in index order."""
        out: list[str] = []
        for ref in self.clips:
            if ref.recording_id not in out:
                out.append(ref.recording_id)
        return tuple(out)


def tennis_scene_dir(clip_dir: Path) -> Path:
    return clip_dir / ANNOTATIONS_DIR_NAME / TENNIS_SCENE_DIR_NAME


def has_tennis_scene_annotation(clip_dir: Path) -> bool:
    """Whether a *complete* tennis_scene annotation exists for the clip."""
    return (tennis_scene_dir(clip_dir) / ANNOTATION_MARKER_NAME).is_file()


def _validate_marker(marker: dict[str, Any], *, source: Path) -> dict[str, Any]:
    version = marker.get("version")
    if version != ANNOTATION_SCHEMA_VERSION:
        raise UnsupportedFormatVersionError(
            f"{source} declares version={version!r}; supported: {ANNOTATION_SCHEMA_VERSION}."
        )
    arrays = marker.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        raise DatasetContractError(
            f"{source}: marker must record an 'arrays' shape spec."
        )
    return arrays


def _validate_scene_against_manifest(
    scene: SceneResult, manifest: ClipManifest
) -> None:
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
        recorded = marker.get("clip_manifest_sha256")
        actual = manifest.digest()
        if recorded != actual:
            raise DatasetContractError(
                f"{manifest.clip_id}: annotation was generated from a different clip.json "
                f"(marker digest {recorded!r} != current {actual!r})."
            )

    scene_path = ann_dir / SCENE_NPZ_NAME
    if not scene_path.is_file():
        raise DatasetContractError(
            f"{manifest.clip_id}: scene archive missing: {scene_path}"
        )
    scene = SceneResult.load(scene_path)
    _validate_scene_against_manifest(scene, manifest)
    _validate_scene_arrays(scene, arrays_spec, clip_id=manifest.clip_id)
    return scene


__all__ = [
    "ANNOTATION_MARKER_NAME",
    "CLIP_MANIFEST_NAME",
    "CLIPS_DIR_NAME",
    "DATASET_INDEX_NAME",
    "MEDIA_DIR_NAME",
    "REQUIRED_SCENE_ARRAYS",
    "SCENE_NPZ_NAME",
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

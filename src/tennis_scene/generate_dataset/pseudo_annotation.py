"""Generate auditable pseudo annotations for structured video clips."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    DatasetClipRecord,
    load_dataset_manifest,
)
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path
from src.tennis_scene.schema import (
    SCENE_MASK_FIELDS,
    SCENE_REASON_FIELDS,
    SceneResult,
    validate_scene_result_arrays,
)
from src.utils.checksum import dual_sha256
from src.utils.io import save_json_atomic, utc_now_iso

ANNOTATION_SCHEMA_VERSION = 1
ANNOTATION_RELATIVE_DIR = Path("annotations") / "tennis_scene"
DECLARED_PIPELINE_CONTRACT = "declared_components_v1"

# (video paths, camera IDs, clip directory) -> scene; the clip directory owns the component store.
SceneRunner = Callable[[Sequence[Path], Sequence[str], Path], SceneResult]


@dataclass(frozen=True)
class AnnotationGenerationResult:
    """Outcome for one requested clip."""

    clip_id: str
    status: Literal["generated", "skipped", "failed"]
    annotation_path: Path | None = None
    error: str | None = None


def _sha256_file(path: Path) -> str:
    return cast(str, dual_sha256(path))


def scene_array_manifest(result: SceneResult) -> dict[str, dict[str, object]]:
    """Shape/dtype of every present scene array, recorded in ``annotation.json``."""
    arrays: dict[str, np.ndarray | None] = {
        "court_kp": result.court_kp,
        "court_vis": result.court_vis,
        "player_position": result.player_position,
        "player_yaw": result.player_yaw,
        "smpl_body_pose": result.smpl_body_pose,
        "smpl_global_orient": result.smpl_global_orient,
        "smpl_betas": result.smpl_betas,
        "smpl_vertices_local": result.smpl_vertices_local,
        "gvhmr_aligned_player_position": result.gvhmr_aligned_player_position,
        "gvhmr_aligned_player_yaw": result.gvhmr_aligned_player_yaw,
        "gvhmr_aligned_smpl_global_orient": (result.gvhmr_aligned_smpl_global_orient),
        "gvhmr_aligned_smpl_vertices_local": (result.gvhmr_aligned_smpl_vertices_local),
        "ball_uv": result.ball_uv,
        "ball_vis": result.ball_vis,
        "ball_3d": result.ball_3d,
        "human_kp_2d": result.human_kp_2d,
        "human_kp_vis": result.human_kp_vis,
        "player_track_ids": result.player_track_ids,
        "player_kp_3d": result.player_kp_3d,
    }
    arrays.update({name: getattr(result, name) for name in (*SCENE_MASK_FIELDS, *SCENE_REASON_FIELDS)})
    return {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in arrays.items()
        if array is not None
    }


def _validate_result(result: SceneResult, record: DatasetClipRecord) -> None:
    validate_scene_result_arrays(result)
    problems: list[str] = []
    expected_n = record.num_cameras
    expected_t = record.num_frames
    if result.num_frames != expected_t:
        problems.append(f"num_frames {result.num_frames} != {expected_t}")
    if abs(result.fps - record.fps) > 0.01:
        problems.append(f"fps {result.fps} != {record.fps}")
    if (result.width, result.height) != (record.width, record.height):
        problems.append(
            f"resolution {result.width}x{result.height} != "
            f"{record.width}x{record.height}"
        )

    required_shapes: dict[str, tuple[int | None, ...]] = {
        "court_kp": (expected_n, expected_t, None, 2),
        "court_vis": (expected_n, expected_t, None),
        "player_position": (None, expected_t, 3),
        "player_yaw": (None, expected_t),
        "human_kp_2d": (None, expected_n, expected_t, 17, 2),
        "human_kp_vis": (None, expected_n, expected_t, 17),
    }
    if result.schema_version != 2 or result.metadata.get("pipeline_contract") != DECLARED_PIPELINE_CONTRACT:
        problems.append(f"only {DECLARED_PIPELINE_CONTRACT} scene v2 results are published, got "
                        f"schema v{result.schema_version} / {result.metadata.get('pipeline_contract')!r}")
    # A v2 scene always carries ball arrays; disabled features are all-invalid masks.
    required_shapes.update({"ball_uv": (expected_n, expected_t, 2), "ball_vis": (expected_n, expected_t), "ball_3d": (expected_t, 3)})
    for name, expected_shape in required_shapes.items():
        value = getattr(result, name)
        if value is None:
            problems.append(f"required pseudo-label array {name!r} is missing")
            continue
        actual_shape = tuple(int(size) for size in value.shape)
        if len(actual_shape) != len(expected_shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(actual_shape, expected_shape, strict=False)
        ):
            problems.append(f"{name} shape {actual_shape} != {expected_shape}")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            problems.append(f"{name} contains non-finite values")

    if result.player_position.shape[0] != result.player_yaw.shape[0]:
        problems.append("player_position and player_yaw player counts differ")
    if result.human_kp_2d is not None and (
        result.human_kp_2d.shape[0] != result.player_position.shape[0]
    ):
        problems.append("human_kp_2d and player_position player counts differ")
    if problems:
        raise ValueError(
            f"pseudo annotation for {record.clip_id!r} violates the dataset "
            f"contract: {'; '.join(problems)}"
        )


def _resolve_clip_inputs(
    dataset_dir: Path, record: DatasetClipRecord
) -> tuple[Path, list[Path], list[str]]:
    clip_dir = dataset_dir / record.path
    clip_manifest_path = clip_dir / "clip.json"
    if not clip_manifest_path.exists():
        raise FileNotFoundError(f"clip manifest not found: {clip_manifest_path}")
    clip_manifest = ClipManifest.load(clip_dir)
    if clip_manifest.clip_id != record.clip_id:
        raise ValueError(
            f"clip_id mismatch: dataset has {record.clip_id!r}, "
            f"clip manifest has {clip_manifest.clip_id!r}"
        )
    camera_ids = list(clip_manifest.camera_ids)
    video_paths = [clip_manifest.media_path(camera_id) for camera_id in camera_ids]
    missing = [path for path in video_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"clip video not found: {missing[0]}")
    return clip_manifest_path, video_paths, camera_ids


def _publish_annotation(
    *, dataset_dir: Path, record: DatasetClipRecord, result: SceneResult,
    clip_manifest_path: Path, pipeline_config_yaml: str,
    publication_identity: Mapping[str, object],
) -> Path:
    """Publish a marker pointing at the clip store's immutable scene export.

    The component store and its exports are never replaced or modified. The
    pipeline configuration is stored content-addressed next to the marker.
    """
    record_path: object = record.path
    if type(record_path) is not str:
        raise TypeError("Dataset clip path must be a string")
    destination = dataset_dir / record_path / ANNOTATION_RELATIVE_DIR
    scene_path = indexed_scene_path(destination / "scene.json")
    config_bytes = pipeline_config_yaml.encode("utf-8")
    config_path = destination / "configs" / f"{hashlib.sha256(config_bytes).hexdigest()}.yaml"
    if not config_path.exists():
        config_path.parent.mkdir(exist_ok=True)
        temporary = config_path.with_suffix(".yaml.tmp")
        temporary.write_bytes(config_bytes)
        temporary.replace(config_path)
    elif config_path.read_bytes() != config_bytes:
        raise ValueError(f"Content-addressed pipeline config was modified: {config_path}")
    _, media_paths, camera_ids = _resolve_clip_inputs(dataset_dir, record)
    annotation = {
        "version": ANNOTATION_SCHEMA_VERSION, "clip_id": record.clip_id, "generator": "src.tennis_scene",
        "generated_at": utc_now_iso(), "scene_result": str(scene_path.relative_to(destination)),
        "scene_index": "scene.json", "pipeline_config": str(config_path.relative_to(destination)),
        "clip_manifest_sha256": _sha256_file(clip_manifest_path), "arrays": scene_array_manifest(result),
        "scene_schema_version": result.schema_version, "result_status": result.metadata["status"],
        "validity_statistics": result.metadata["validity_statistics"],
        "publication_identity_sha256": document_digest(publication_identity),
        "media_sha256": {camera: _sha256_file(path) for camera, path in zip(camera_ids, media_paths, strict=True)},
    }
    marker = destination / "annotation.json"
    save_json_atomic(annotation, marker)
    failure_marker = destination.parent / "tennis_scene.failure.json"
    if failure_marker.exists():
        failure_marker.unlink()
    return marker


def generate_pseudo_annotations(
    dataset_dir: Path,
    runner: SceneRunner,
    *,
    pipeline_config_yaml: str,
    publication_identity: Mapping[str, object],
    clip_ids: Sequence[str] | None = None,
    overwrite: bool = False,
    continue_on_error: bool = True,
) -> list[AnnotationGenerationResult]:
    """Generate missing pseudo annotations while preserving per-clip outcomes.

    ``runner`` must leave the scene export in the clip store
    (``<clip>/annotations/tennis_scene/scene.json``); this function only
    publishes the completion marker. An existing marker is reused only when
    its publication identity, clip manifest and media hashes all match.
    """
    if (
        not dataset_dir.is_absolute()
        or dataset_dir.resolve(strict=False) != dataset_dir
    ):
        raise ValueError(
            "dataset_dir must be an absolute normalized Path from PathResolver."
        )
    root = dataset_dir
    dataset = load_dataset_manifest(root)
    selected_ids = sorted(dataset.clips) if clip_ids is None else list(clip_ids)
    unknown = [clip_id for clip_id in selected_ids if clip_id not in dataset.clips]
    if unknown:
        raise KeyError(f"clip_id not found in dataset: {unknown[0]!r}")

    outcomes: list[AnnotationGenerationResult] = []
    for clip_id in selected_ids:
        record = dataset.clips[clip_id]
        destination = root / record.path / ANNOTATION_RELATIVE_DIR / "annotation.json"
        try:
            clip_manifest_path, video_paths, camera_ids = _resolve_clip_inputs(
                root, record
            )
            manifest_before = _sha256_file(clip_manifest_path)
            media_before = {camera: _sha256_file(path) for camera, path in zip(camera_ids, video_paths, strict=True)}
            if destination.exists() and not overwrite:
                marker = json.loads(destination.read_text())
                expected_media = {camera: _sha256_file(path) for camera, path in zip(camera_ids, video_paths, strict=True)}
                if (marker.get("publication_identity_sha256") != document_digest(publication_identity)
                    or marker.get("clip_manifest_sha256") != _sha256_file(clip_manifest_path)
                    or marker.get("media_sha256") != expected_media):
                    raise ValueError("Stale annotation inputs/model/settings; use overwrite=true")
                outcomes.append(AnnotationGenerationResult(clip_id, "skipped", destination))
                continue
            result = runner(video_paths, camera_ids, clip_manifest_path.parent)
            if _sha256_file(clip_manifest_path) != manifest_before or any(_sha256_file(path) != media_before[camera] for camera, path in zip(camera_ids, video_paths, strict=True)):
                raise ValueError("Clip inputs changed during reconstruction")
            _validate_result(result, record)
            annotation_path = _publish_annotation(
                dataset_dir=root,
                record=record,
                result=result,
                clip_manifest_path=clip_manifest_path,
                pipeline_config_yaml=pipeline_config_yaml,
                publication_identity=publication_identity,
            )
            outcomes.append(
                AnnotationGenerationResult(
                    clip_id=clip_id,
                    status="generated",
                    annotation_path=annotation_path,
                )
            )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            failure_path = (
                root / record.path / "annotations" / "tennis_scene.failure.json"
            )
            save_json_atomic(
                {
                    "clip_id": clip_id,
                    "failed_at": utc_now_iso(),
                    "error": message,
                },
                failure_path,
            )
            outcomes.append(
                AnnotationGenerationResult(
                    clip_id=clip_id, status="failed", error=message
                )
            )
            if not continue_on_error:
                raise
    return outcomes


__all__ = [
    "ANNOTATION_RELATIVE_DIR",
    "ANNOTATION_SCHEMA_VERSION",
    "AnnotationGenerationResult",
    "SceneRunner",
    "generate_pseudo_annotations",
    "scene_array_manifest",
]

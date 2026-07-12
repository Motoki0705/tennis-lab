"""Generate auditable pseudo annotations for structured video clips."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    DatasetClipRecord,
    load_dataset_manifest,
)
from src.tennis_scene.io import SceneResult
from src.utils.io import save_json_atomic, utc_now_iso

ANNOTATION_SCHEMA_VERSION = 1
ANNOTATION_RELATIVE_DIR = Path("annotations") / "tennis_scene"

SceneRunner = Callable[[Sequence[Path], Sequence[str]], SceneResult]


@dataclass(frozen=True)
class AnnotationGenerationResult:
    """Outcome for one requested clip."""

    clip_id: str
    status: Literal["generated", "skipped", "failed"]
    annotation_path: Path | None = None
    error: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_manifest(result: SceneResult) -> dict[str, dict[str, object]]:
    arrays: dict[str, np.ndarray | None] = {
        "court_kp": result.court_kp,
        "court_vis": result.court_vis,
        "player_position": result.player_position,
        "player_yaw": result.player_yaw,
        "smpl_body_pose": result.smpl_body_pose,
        "smpl_global_orient": result.smpl_global_orient,
        "smpl_betas": result.smpl_betas,
        "smpl_vertices_local": result.smpl_vertices_local,
        "ball_uv": result.ball_uv,
        "ball_vis": result.ball_vis,
        "ball_3d": result.ball_3d,
        "human_kp_2d": result.human_kp_2d,
        "human_kp_vis": result.human_kp_vis,
        "player_track_ids": result.player_track_ids,
        "player_kp_3d": result.player_kp_3d,
    }
    return {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in arrays.items()
        if array is not None
    }


def _validate_result(result: SceneResult, record: DatasetClipRecord) -> None:
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
        "ball_uv": (expected_n, expected_t, 2),
        "ball_vis": (expected_n, expected_t),
        "ball_3d": (expected_t, 3),
    }
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
    *,
    dataset_dir: Path,
    record: DatasetClipRecord,
    result: SceneResult,
    clip_manifest_path: Path,
    pipeline_config_yaml: str,
    overwrite: bool,
) -> Path:
    clip_dir = dataset_dir / record.path
    destination = clip_dir / ANNOTATION_RELATIVE_DIR
    completion_marker = destination / "annotation.json"
    if completion_marker.exists() and not overwrite:
        return completion_marker
    if destination.exists() and not completion_marker.exists() and not overwrite:
        raise ValueError(
            f"incomplete annotation directory exists at {destination}; "
            "inspect it or set overwrite=true"
        )

    staging = destination.parent / ".tennis_scene.tmp"
    backup = destination.parent / ".tennis_scene.backup"
    if staging.exists() or backup.exists():
        raise ValueError(
            f"stale annotation transaction exists under {destination.parent}; "
            "inspect or remove it before retrying"
        )
    staging.mkdir(parents=True)

    result.metadata = {
        **result.metadata,
        "dataset_clip_id": record.clip_id,
        "clip_manifest": str(clip_manifest_path.relative_to(dataset_dir)),
    }
    scene_path = staging / "scene.npz"
    result.save(scene_path)
    (staging / "pipeline_config.yaml").write_text(
        pipeline_config_yaml, encoding="utf-8"
    )
    annotation = {
        "version": ANNOTATION_SCHEMA_VERSION,
        "clip_id": record.clip_id,
        "generator": "src.tennis_scene",
        "generated_at": utc_now_iso(),
        "scene_result": "scene.npz",
        "pipeline_config": "pipeline_config.yaml",
        "clip_manifest_sha256": _sha256_file(clip_manifest_path),
        "arrays": _shape_manifest(result),
    }
    save_json_atomic(annotation, staging / "annotation.json")

    if destination.exists():
        destination.replace(backup)
    staging.replace(destination)
    if backup.exists():
        shutil.rmtree(backup)
    failure_marker = destination.parent / "tennis_scene.failure.json"
    if failure_marker.exists():
        failure_marker.unlink()
    return destination / "annotation.json"


def generate_pseudo_annotations(
    dataset_dir: str | Path,
    runner: SceneRunner,
    *,
    pipeline_config_yaml: str,
    clip_ids: Sequence[str] | None = None,
    overwrite: bool = False,
    continue_on_error: bool = True,
) -> list[AnnotationGenerationResult]:
    """Generate missing pseudo annotations while preserving per-clip outcomes."""
    root = Path(dataset_dir).resolve()
    dataset = load_dataset_manifest(root)
    selected_ids = sorted(dataset.clips) if clip_ids is None else list(clip_ids)
    unknown = [clip_id for clip_id in selected_ids if clip_id not in dataset.clips]
    if unknown:
        raise KeyError(f"clip_id not found in dataset: {unknown[0]!r}")

    outcomes: list[AnnotationGenerationResult] = []
    for clip_id in selected_ids:
        record = dataset.clips[clip_id]
        destination = root / record.path / ANNOTATION_RELATIVE_DIR / "annotation.json"
        if destination.exists() and not overwrite:
            outcomes.append(
                AnnotationGenerationResult(
                    clip_id=clip_id,
                    status="skipped",
                    annotation_path=destination,
                )
            )
            continue
        try:
            clip_manifest_path, video_paths, camera_ids = _resolve_clip_inputs(
                root, record
            )
            result = runner(video_paths, camera_ids)
            _validate_result(result, record)
            annotation_path = _publish_annotation(
                dataset_dir=root,
                record=record,
                result=result,
                clip_manifest_path=clip_manifest_path,
                pipeline_config_yaml=pipeline_config_yaml,
                overwrite=overwrite,
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
            failure_path = root / record.path / "annotations" / "tennis_scene.failure.json"
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
]

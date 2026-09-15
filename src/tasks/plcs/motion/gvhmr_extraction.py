"""Manifest-driven GVHMR extraction of foreground tennis-player motions."""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.submodules.configuration import GvhmrDemoConfig
from src.submodules.models import (
    DinoPersonTracker,
    GvhmrMeshRecovery,
    GvhmrRequest,
    Hmr2FeatureExtractor,
    ImageFeatureRequest,
    Pose2DRequest,
    SmplCoco17Reconstructor,
    TrackRequest,
    ViTPosePose2D,
)
from src.tasks.plcs.motion.artifact import load_motion_clip, save_motion_clip
from src.tasks.plcs.motion.contracts import Coco17MotionClip
from src.tasks.plcs.motion.sources import GvhmrCoco17Adapter
from src.tennis_scene.generate_dataset import load_dataset_manifest
from src.tennis_scene.generate_dataset.manifest import ClipManifest, file_sha256
from src.utils.io import load_json, save_json_atomic, utc_now_iso
from src.utils.schema.player import COCO17_BONE_LENGTH_EDGES, COCO17_SKELETON
from src.utils.video.reader import probe_video_info

SELECTION_SCHEMA_VERSION = "plcs_gvhmr_selection_v1"
EXTRACTION_PIPELINE_VERSION = "plcs_gvhmr_dino_roi_stitch_v1"
COLLECTION_SCHEMA_VERSION = "plcs_gvhmr_collection_v2"
RECORD_SCHEMA_VERSION = "plcs_gvhmr_record_v2"
RAW_SCHEMA_VERSION = "plcs_gvhmr_global_smpl_v2"


def require_extraction_space(output_root: Path, frame_count: int) -> None:
    """Reserve 1 GiB plus conservative per-clip space, including atomic writes."""
    required = 1024**3 + frame_count * 16_384 + 8 * 1024**2
    available = shutil.disk_usage(output_root).free
    if available < required:
        raise OSError(
            f"Insufficient GVHMR output storage at {output_root}: "
            f"{available} bytes free, {required} required including reserve. "
            "Completed clips are retained; free space and resume extraction."
        )


_RAW_KEYS = {
    "metadata_json",
    "body_pose",
    "betas",
    "global_orient",
    "transl",
    "K_fullimg",
    "keypoints_2d_px",
    "boxes_xys_px",
    "observed_mask",
}


def normalize_vitpose_confidence(
    raw_scores: NDArray[np.float32],
) -> tuple[NDArray[np.float32], int]:
    """Bound ViTPose heatmap maxima to the common confidence interval.

    Released ViTPose weights occasionally produce heatmap maxima just above
    one. The unmodified values remain in the raw sidecar; the common contract
    explicitly clips only that upper numerical overshoot.
    """
    scores = np.asarray(raw_scores)
    if scores.dtype != np.dtype(np.float32):
        raise TypeError("ViTPose confidence scores must use float32.")
    if not np.isfinite(scores).all() or bool((scores < 0.0).any()):
        raise ValueError("ViTPose confidence scores must be finite and non-negative.")
    clipped_count = int(np.count_nonzero(scores > 1.0))
    normalized = np.minimum(scores, np.float32(1.0)).astype(np.float32, copy=True)
    return normalized, clipped_count


def _exact_mapping(
    value: object,
    *,
    fields: set[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    if set(value) != fields:
        raise ValueError(
            f"{name} fields must be exactly {sorted(fields)}, got {sorted(value)}."
        )
    return cast(Mapping[str, object], value)


def _trimmed_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True, slots=True)
class GvhmrCameraSelection:
    """One camera and the normalized image region assigned to its near player."""

    camera_id: str
    player_role: str
    footpoint_polygon_normalized: tuple[tuple[float, float], ...]

    @classmethod
    def from_mapping(cls, value: object) -> GvhmrCameraSelection:
        mapping = _exact_mapping(
            value,
            fields={
                "camera_id",
                "player_role",
                "footpoint_polygon_normalized",
            },
            name="GVHMR camera selection",
        )
        raw_polygon = mapping["footpoint_polygon_normalized"]
        if not isinstance(raw_polygon, Sequence) or isinstance(raw_polygon, str):
            raise TypeError("footpoint_polygon_normalized must be a point sequence.")
        points: list[tuple[float, float]] = []
        for index, raw_point in enumerate(raw_polygon):
            if (
                not isinstance(raw_point, Sequence)
                or isinstance(raw_point, str)
                or len(raw_point) != 2
            ):
                raise ValueError(
                    f"footpoint_polygon_normalized[{index}] must contain x and y."
                )
            if any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in raw_point
            ):
                raise TypeError("Normalized polygon coordinates must be numeric.")
            point = tuple(float(item) for item in raw_point)
            if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in point):
                raise ValueError("Normalized polygon coordinates must be within [0,1].")
            points.append(cast(tuple[float, float], point))
        if len(points) < 3:
            raise ValueError(
                "A camera footpoint polygon requires at least three points."
            )
        return cls(
            camera_id=_trimmed_string(mapping["camera_id"], name="camera_id"),
            player_role=_trimmed_string(mapping["player_role"], name="player_role"),
            footpoint_polygon_normalized=tuple(points),
        )

    def polygon_px(
        self,
        *,
        width: int,
        height: int,
    ) -> tuple[tuple[float, float], ...]:
        return tuple(
            (x * width, y * height) for x, y in self.footpoint_polygon_normalized
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "player_role": self.player_role,
            "footpoint_polygon_normalized": [
                [x, y] for x, y in self.footpoint_polygon_normalized
            ],
        }


@dataclass(frozen=True, slots=True)
class GvhmrSelectionConfig:
    """Strict dataset-specific foreground-player selection contract."""

    dataset_id: str
    cameras: tuple[GvhmrCameraSelection, ...]

    @classmethod
    def load(cls, path: str | Path) -> GvhmrSelectionConfig:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"GVHMR selection config not found: {source}")
        raw = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
        mapping = _exact_mapping(
            raw,
            fields={"schema_version", "dataset_id", "cameras"},
            name="GVHMR selection config",
        )
        if mapping["schema_version"] != SELECTION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported GVHMR selection schema: {mapping['schema_version']!r}."
            )
        raw_cameras = mapping["cameras"]
        if not isinstance(raw_cameras, Sequence) or isinstance(raw_cameras, str):
            raise TypeError("GVHMR selection cameras must be a sequence.")
        cameras = tuple(
            GvhmrCameraSelection.from_mapping(value) for value in raw_cameras
        )
        if not cameras:
            raise ValueError("GVHMR selection requires at least one camera.")
        camera_ids = tuple(camera.camera_id for camera in cameras)
        player_roles = tuple(camera.player_role for camera in cameras)
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("GVHMR selection camera_id values must be unique.")
        if len(set(player_roles)) != len(player_roles):
            raise ValueError("GVHMR selection player_role values must be unique.")
        return cls(
            dataset_id=_trimmed_string(mapping["dataset_id"], name="dataset_id"),
            cameras=cameras,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "cameras": [camera.to_dict() for camera in self.cameras],
        }

    def select_cameras(
        self,
        camera_ids: tuple[str, ...] | None,
    ) -> tuple[GvhmrCameraSelection, ...]:
        """Resolve an optional CLI subset while preserving configured order."""
        if camera_ids is None:
            return self.cameras
        if not camera_ids or len(set(camera_ids)) != len(camera_ids):
            raise ValueError("Requested camera IDs must be non-empty and unique.")
        configured = {camera.camera_id for camera in self.cameras}
        missing = sorted(set(camera_ids) - configured)
        if missing:
            raise ValueError(f"Unknown requested camera IDs: {missing}.")
        requested = set(camera_ids)
        return tuple(camera for camera in self.cameras if camera.camera_id in requested)


def load_model_runtime(
    path: str | Path,
    *,
    repository_root: str | Path,
    checkpoint_root: str | Path | None = None,
    dino_checkpoint: str | Path | None = None,
) -> GvhmrDemoConfig:
    """Reuse the canonical model settings with DINO's separate checkpoint root."""
    config_path = Path(path)
    root = Path(repository_root).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"GVHMR model config not found: {config_path}")
    if not root.is_dir():
        raise FileNotFoundError(f"GVHMR asset repository not found: {root}")
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    if not isinstance(raw, dict):
        raise TypeError("GVHMR model config must contain a mapping.")
    assets = raw.get("assets")
    if not isinstance(assets, dict):
        raise TypeError("GVHMR model config assets must contain a mapping.")
    checkpoints = assets.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise TypeError("GVHMR model config checkpoints must contain a mapping.")
    dino_relative = checkpoints.get("dino")
    if not isinstance(dino_relative, str) or not dino_relative.strip():
        raise TypeError("GVHMR model config DINO checkpoint must be a path string.")
    raw.pop("defaults", None)
    raw.pop("hydra", None)
    if checkpoint_root is not None:
        paths = raw.get("paths")
        if not isinstance(paths, dict):
            raise TypeError("GVHMR model config paths must contain a mapping.")
        resolved_checkpoints = Path(checkpoint_root).resolve()
        if not resolved_checkpoints.is_dir():
            raise FileNotFoundError(
                f"GVHMR checkpoint root not found: {resolved_checkpoints}"
            )
        paths["checkpoint_root"] = str(resolved_checkpoints)
    runtime = GvhmrDemoConfig.from_mapping(
        cast("Mapping[str, object]", raw),
        repository_root=root,
    )
    resolved_dino = (
        Path(dino_checkpoint).resolve()
        if dino_checkpoint
        else (root / "ckpt" / dino_relative).resolve()
    )
    if not resolved_dino.is_file():
        raise FileNotFoundError(f"DINO checkpoint not found: {resolved_dino}")
    return replace(
        runtime,
        assets=replace(runtime.assets, dino_checkpoint=resolved_dino),
    )


@dataclass(frozen=True, slots=True)
class ExtractionOutputPaths:
    """All durable artifacts produced for one camera/player motion."""

    common_motion: Path
    raw_global_smpl: Path
    record: Path
    selection_preview: Path

    @classmethod
    def for_motion(
        cls,
        output_root: Path,
        *,
        clip: ClipManifest,
        camera_id: str,
    ) -> ExtractionOutputPaths:
        directory = output_root / clip.video_id / clip.clip_name
        return cls(
            common_motion=directory / f"{camera_id}.motion.npz",
            raw_global_smpl=directory / f"{camera_id}.gvhmr.npz",
            record=directory / f"{camera_id}.json",
            selection_preview=directory / f"{camera_id}.selection.jpg",
        )


@dataclass(slots=True)
class _Models:
    tracker: DinoPersonTracker
    pose: ViTPosePose2D
    features: Hmr2FeatureExtractor
    gvhmr: GvhmrMeshRecovery
    joints: SmplCoco17Reconstructor

    @classmethod
    def build(cls, runtime: GvhmrDemoConfig) -> _Models:
        device = runtime.runtime.device
        return cls(
            tracker=DinoPersonTracker(
                checkpoint=runtime.assets.dino_checkpoint,
                repository=runtime.assets.dino_repository,
                device=device,
                confidence=runtime.runtime.dino_detector.confidence,
                short_side=runtime.runtime.dino_detector.short_side,
                max_long_side=runtime.runtime.dino_detector.max_long_side,
            ),
            pose=ViTPosePose2D(
                checkpoint=runtime.assets.vitpose_checkpoint,
                device=device,
                flip_test=runtime.runtime.vitpose.flip_test,
                batch_size=runtime.runtime.vitpose.batch_size,
                head_config=runtime.runtime.vitpose.head,
            ),
            features=Hmr2FeatureExtractor(
                checkpoint=runtime.assets.hmr2_checkpoint,
                device=device,
                batch_size=runtime.runtime.hmr2.batch_size,
                mean_params_path=runtime.assets.bundled.hmr2_mean_params,
            ),
            gvhmr=GvhmrMeshRecovery(
                checkpoint=runtime.assets.gvhmr_checkpoint,
                body_models_dir=runtime.assets.body_models_dir,
                device=device,
                bundled_assets=runtime.assets.bundled,
            ),
            joints=SmplCoco17Reconstructor(
                body_models_dir=runtime.assets.body_models_dir,
                device=device,
                bundled_assets=runtime.assets.bundled,
            ),
        )

    def unload(self) -> None:
        self.joints.unload()
        for model in (self.gvhmr, self.features, self.pose, self.tracker):
            model.unload()


class GvhmrMotionExtractor:
    """Extract two foreground-view motions per manifest clip into PLCS format."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        output_root: str | Path,
        selection: GvhmrSelectionConfig,
        selection_digest: str,
        model_runtime: GvhmrDemoConfig,
        max_frames: int | None,
        overwrite: bool,
        write_preview: bool,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.output_root = Path(output_root).resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Input dataset root not found: {self.dataset_root}"
            )
        if (
            self.output_root == self.dataset_root
            or self.dataset_root in self.output_root.parents
        ):
            raise ValueError("GVHMR output_root must not be inside the input dataset.")
        if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
            raise ValueError("max_frames must be a positive integer or None.")
        if type(overwrite) is not bool or type(write_preview) is not bool:
            raise TypeError("overwrite and write_preview must be bool values.")
        self.selection = selection
        self.selection_digest = selection_digest
        self.model_runtime = model_runtime
        self.max_frames = max_frames
        self.overwrite = overwrite
        self.write_preview = write_preview
        self._records: dict[str, dict[str, object]] = {}

    def run(
        self,
        *,
        clip_ids: tuple[str, ...] | None = None,
        camera_ids: tuple[str, ...] | None = None,
        max_clips: int | None = None,
    ) -> dict[str, int]:
        """Run extraction in stable clip/camera order and publish progress atomically."""
        dataset = load_dataset_manifest(self.dataset_root)
        if dataset.dataset_id != self.selection.dataset_id:
            raise ValueError(
                f"Selection dataset_id={self.selection.dataset_id!r} disagrees with "
                f"manifest dataset_id={dataset.dataset_id!r}."
            )
        selected_ids = tuple(sorted(dataset.clips))
        selected_cameras = self.selection.select_cameras(camera_ids)
        if clip_ids is not None:
            missing = sorted(set(clip_ids) - set(dataset.clips))
            if missing:
                raise ValueError(f"Unknown requested clip IDs: {missing}.")
            selected_ids = tuple(
                clip_id for clip_id in selected_ids if clip_id in clip_ids
            )
        if max_clips is not None:
            if type(max_clips) is not int or max_clips <= 0:
                raise ValueError("max_clips must be a positive integer or None.")
            selected_ids = selected_ids[:max_clips]
        if not selected_ids:
            raise ValueError("No clips were selected for GVHMR extraction.")

        self.output_root.mkdir(parents=True, exist_ok=True)
        self._load_collection_manifest()
        models = _Models.build(self.model_runtime)
        # Validate the licensed body model/checkpoint before spending time tracking.
        models.gvhmr.load()
        adapter = GvhmrCoco17Adapter(models.joints)
        generated = 0
        skipped = 0
        try:
            for clip_id in selected_ids:
                record = dataset.clips[clip_id]
                clip = ClipManifest.load(self.dataset_root / record.path)
                self._validate_clip_record(clip, expected=record.to_dict())
                for camera in selected_cameras:
                    paths = ExtractionOutputPaths.for_motion(
                        self.output_root,
                        clip=clip,
                        camera_id=camera.camera_id,
                    )
                    expected_frames = min(
                        clip.num_frames,
                        self.max_frames or clip.num_frames,
                    )
                    source_id = self._source_id(clip, camera)
                    if self._can_resume(
                        paths,
                        source_id=source_id,
                        expected_frames=expected_frames,
                    ):
                        skipped += 1
                        continue
                    require_extraction_space(self.output_root, expected_frames)
                    result = self._extract_one(
                        models=models,
                        adapter=adapter,
                        clip=clip,
                        camera=camera,
                        paths=paths,
                    )
                    self._records[source_id] = result
                    self._save_collection_manifest()
                    generated += 1
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            models.unload()
        return {"generated": generated, "skipped": skipped}

    def _extract_one(
        self,
        *,
        models: _Models,
        adapter: GvhmrCoco17Adapter,
        clip: ClipManifest,
        camera: GvhmrCameraSelection,
        paths: ExtractionOutputPaths,
    ) -> dict[str, object]:
        if camera.camera_id not in clip.camera_ids:
            raise ValueError(
                f"{clip.clip_id} does not contain configured camera {camera.camera_id!r}."
            )
        video_path = clip.media_path(camera.camera_id)
        info = probe_video_info(video_path)
        if (info.width, info.height, info.frame_count) != (
            clip.width,
            clip.height,
            clip.num_frames,
        ):
            raise ValueError(
                f"{clip.clip_id}/{camera.camera_id} video metadata disagrees with clip.json."
            )
        if abs(info.fps - clip.fps) > 1e-3:
            raise ValueError(
                f"{clip.clip_id}/{camera.camera_id} FPS disagrees with clip.json."
            )
        tracks = models.tracker.predict(
            TrackRequest(
                video_path=video_path,
                num_tracks=1,
                interactive=False,
                footpoint_polygon_px=camera.polygon_px(
                    width=info.width,
                    height=info.height,
                ),
                stitch_tracklets_in_roi=True,
                max_frames=self.max_frames,
            )
        )
        if len(tracks.track_ids) != 1:
            raise RuntimeError("Foreground tracking must select exactly one track.")
        track_id = tracks.track_ids[0]
        frame_count = tracks.num_frames
        boxes = tracks.bbx_xys(
            track_id,
            base_enlarge=self.model_runtime.runtime.tracking.bbox_enlarge,
        )
        observed = tracks.observed_mask(track_id)
        pose = models.pose.predict(Pose2DRequest(video_path=video_path, bbx_xys=boxes))
        features = models.features.predict(
            ImageFeatureRequest(video_path=video_path, bbx_xys=boxes)
        )
        recovered = models.gvhmr.predict(
            GvhmrRequest(
                kp2d=pose.keypoints,
                bbx_xys=boxes,
                f_imgseq=features.features,
                width=info.width,
                height=info.height,
                static_cam=self.model_runtime.runtime.static_cam,
            )
        )
        source_id = self._source_id(clip, camera)
        raw_confidence = np.asarray(
            pose.keypoints[..., 2].detach().cpu().numpy(),
            dtype=np.float32,
        )
        confidence, clipped_confidence_count = normalize_vitpose_confidence(
            raw_confidence
        )
        canonical = adapter.convert(
            recovered.smpl_params_global,
            source_id=source_id,
            source_path=video_path,
            fps=clip.fps,
            joint_confidence=confidence,
            frame_valid=np.asarray(observed.cpu().numpy(), dtype=np.bool_),
            provenance={
                "dataset_id": clip.dataset_id,
                "clip_id": clip.clip_id,
                "clip_manifest_sha256": clip.digest(),
                "camera_id": camera.camera_id,
                "player_role": camera.player_role,
                "track_id": track_id,
                "selection_config_sha256": self.selection_digest,
                "extraction_pipeline_version": EXTRACTION_PIPELINE_VERSION,
                "footpoint_polygon_normalized": [
                    list(point) for point in camera.footpoint_polygon_normalized
                ],
                "detector": "dino_4scale_swin_l_coco_person",
                "tracker": "dino_botsort_largest_bbox_per_frame_stitched_in_roi",
                "static_camera_prior": self.model_runtime.runtime.static_cam,
                "joint_confidence_transform": "clip_vitpose_heatmap_max_to_unit_v1",
                "joint_confidence_values_clipped": clipped_confidence_count,
                "joint_confidence_raw_max": float(raw_confidence.max()),
                "truncated_to_frames": (
                    frame_count if frame_count < clip.num_frames else None
                ),
            },
        )
        require_extraction_space(self.output_root, canonical.frame_count)
        save_motion_clip(canonical, paths.common_motion)
        self._save_raw_global_smpl(
            paths.raw_global_smpl,
            clip=canonical,
            parameters=recovered.smpl_params_global,
            K_fullimg=recovered.K_fullimg,
            keypoints=pose.keypoints,
            boxes=boxes,
            observed=observed,
        )
        if self.write_preview:
            self._write_selection_preview(
                paths.selection_preview,
                video_path=video_path,
                camera=camera,
                boxes=boxes,
                keypoints=pose.keypoints,
            )
        quality = self._quality_metrics(canonical)
        quality.update(
            {
                "joint_confidence_raw_max": float(raw_confidence.max()),
                "joint_confidence_values_clipped": clipped_confidence_count,
            }
        )
        result: dict[str, object] = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "source_id": source_id,
            "dataset_id": clip.dataset_id,
            "clip_id": clip.clip_id,
            "camera_id": camera.camera_id,
            "player_role": camera.player_role,
            "track_id": track_id,
            "frame_count": canonical.frame_count,
            "native_fps": canonical.fps,
            "selection_config_sha256": self.selection_digest,
            "extraction_pipeline_version": EXTRACTION_PIPELINE_VERSION,
            "common_motion": str(paths.common_motion.relative_to(self.output_root)),
            "raw_global_smpl": str(paths.raw_global_smpl.relative_to(self.output_root)),
            "selection_preview": (
                str(paths.selection_preview.relative_to(self.output_root))
                if self.write_preview
                else None
            ),
            "quality": quality,
            "completed_at": utc_now_iso(),
        }
        save_json_atomic(result, paths.record)
        return result

    def _can_resume(
        self,
        paths: ExtractionOutputPaths,
        *,
        source_id: str,
        expected_frames: int,
    ) -> bool:
        required = {paths.common_motion, paths.raw_global_smpl, paths.record}
        if self.write_preview:
            required.add(paths.selection_preview)
        existing = {path for path in required if path.exists()}
        if self.overwrite:
            return False
        if not existing:
            return False
        if existing != required:
            raise RuntimeError(
                f"Incomplete GVHMR artifact set exists for {source_id}: "
                f"{sorted(str(path) for path in existing)}. Use --overwrite after review."
            )
        clip = load_motion_clip(paths.common_motion)
        record = load_json(paths.record)
        if not isinstance(record, dict):
            raise ValueError(f"GVHMR record must be an object: {paths.record}")
        if (
            clip.source_id != source_id
            or clip.frame_count != expected_frames
            or record.get("source_id") != source_id
            or record.get("frame_count") != expected_frames
            or record.get("selection_config_sha256") != self.selection_digest
            or record.get("schema_version") != RECORD_SCHEMA_VERSION
            or record.get("extraction_pipeline_version") != EXTRACTION_PIPELINE_VERSION
        ):
            raise RuntimeError(
                f"Existing GVHMR artifacts are incompatible for {source_id}; "
                "use --overwrite after review."
            )
        with np.load(paths.raw_global_smpl, allow_pickle=False) as archive:
            if set(archive.files) != _RAW_KEYS:
                raise ValueError(
                    f"Invalid raw GVHMR artifact keys: {paths.raw_global_smpl}"
                )
            if archive["body_pose"].shape != (expected_frames, 63):
                raise ValueError(
                    f"Invalid raw GVHMR frame count: {paths.raw_global_smpl}"
                )
        self._records[source_id] = cast(dict[str, object], record)
        self._save_collection_manifest()
        return True

    def _load_collection_manifest(self) -> None:
        path = self.output_root / "manifest.json"
        if not path.exists():
            return
        raw = load_json(path)
        mapping = _exact_mapping(
            raw,
            fields={
                "schema_version",
                "dataset_id",
                "selection_config_sha256",
                "extraction_pipeline_version",
                "selection",
                "updated_at",
                "records",
            },
            name="GVHMR collection manifest",
        )
        if (
            mapping["schema_version"] != COLLECTION_SCHEMA_VERSION
            or mapping["dataset_id"] != self.selection.dataset_id
            or mapping["selection_config_sha256"] != self.selection_digest
            or mapping["extraction_pipeline_version"] != EXTRACTION_PIPELINE_VERSION
            or mapping["selection"] != self.selection.to_dict()
        ):
            raise RuntimeError(f"Existing collection manifest is incompatible: {path}.")
        records = mapping["records"]
        if not isinstance(records, Sequence) or isinstance(records, str):
            raise TypeError("GVHMR collection records must be a sequence.")
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("GVHMR collection record entries must be objects.")
            source_id = _trimmed_string(record.get("source_id"), name="source_id")
            if source_id in self._records:
                raise ValueError(f"Duplicate GVHMR collection source_id: {source_id}")
            self._records[source_id] = record

    def _save_collection_manifest(self) -> None:
        save_json_atomic(
            {
                "schema_version": COLLECTION_SCHEMA_VERSION,
                "dataset_id": self.selection.dataset_id,
                "selection_config_sha256": self.selection_digest,
                "extraction_pipeline_version": EXTRACTION_PIPELINE_VERSION,
                "selection": self.selection.to_dict(),
                "updated_at": utc_now_iso(),
                "records": [self._records[key] for key in sorted(self._records)],
            },
            self.output_root / "manifest.json",
        )

    @staticmethod
    def _validate_clip_record(
        clip: ClipManifest,
        *,
        expected: Mapping[str, object],
    ) -> None:
        checks = {
            "clip_id": clip.clip_id,
            "video_id": clip.video_id,
            "clip_name": clip.clip_name,
            "num_cameras": len(clip.camera_ids),
            "num_frames": clip.num_frames,
            "fps": clip.fps,
            "width": clip.width,
            "height": clip.height,
        }
        if any(expected[key] != value for key, value in checks.items()):
            raise ValueError(f"Dataset and clip manifests disagree for {clip.clip_id}.")

    @staticmethod
    def _source_id(
        clip: ClipManifest,
        camera: GvhmrCameraSelection,
    ) -> str:
        return f"{clip.clip_id}:{camera.camera_id}:{camera.player_role}"

    @staticmethod
    def _quality_metrics(clip: Coco17MotionClip) -> dict[str, object]:
        velocity = (
            np.linalg.norm(np.diff(clip.root_translation_m, axis=0), axis=1) * clip.fps
            if clip.frame_count > 1
            else np.zeros(1, dtype=np.float32)
        )
        bone_lengths = np.stack(
            [
                np.linalg.norm(
                    clip.joints_3d_m[:, left] - clip.joints_3d_m[:, right], axis=1
                )
                for left, right in COCO17_BONE_LENGTH_EDGES
            ],
            axis=1,
        )
        bone_means = np.maximum(bone_lengths.mean(axis=0), 1e-8)
        bone_cv = bone_lengths.std(axis=0) / bone_means
        return {
            "observed_frame_ratio": float(clip.frame_valid.mean()),
            "joint_confidence_mean": float(clip.joint_confidence.mean()),
            "joint_confidence_p10": float(np.percentile(clip.joint_confidence, 10)),
            "root_displacement_m": float(
                np.linalg.norm(clip.root_translation_m[-1] - clip.root_translation_m[0])
            ),
            "root_speed_mps_p50": float(np.percentile(velocity, 50)),
            "root_speed_mps_p95": float(np.percentile(velocity, 95)),
            "root_height_m_range": [
                float(clip.root_translation_m[:, 2].min()),
                float(clip.root_translation_m[:, 2].max()),
            ],
            "bone_length_cv_median": float(np.median(bone_cv)),
            "bone_length_cv_max": float(bone_cv.max()),
        }

    @staticmethod
    def _save_raw_global_smpl(
        path: Path,
        *,
        clip: Coco17MotionClip,
        parameters: Mapping[str, torch.Tensor],
        K_fullimg: torch.Tensor,
        keypoints: torch.Tensor,
        boxes: torch.Tensor,
        observed: torch.Tensor,
    ) -> None:
        metadata = json.dumps(
            {
                "schema_version": RAW_SCHEMA_VERSION,
                "extraction_pipeline_version": EXTRACTION_PIPELINE_VERSION,
                "source_id": clip.source_id,
                "coordinate_system": "gvhmr_aligned_y_up_m",
                "frame_count": clip.frame_count,
                "native_fps": clip.fps,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        arrays: dict[str, NDArray[Any]] = {
            name: np.asarray(
                parameters[name].detach().float().cpu().numpy(), dtype=np.float32
            )
            for name in ("body_pose", "betas", "global_orient", "transl")
        }
        arrays.update(
            {
                "metadata_json": np.asarray(metadata),
                "K_fullimg": np.asarray(
                    K_fullimg.detach().float().cpu().numpy(), dtype=np.float32
                ),
                "keypoints_2d_px": np.asarray(
                    keypoints.detach().float().cpu().numpy(), dtype=np.float32
                ),
                "boxes_xys_px": np.asarray(
                    boxes.detach().float().cpu().numpy(), dtype=np.float32
                ),
                "observed_mask": np.asarray(observed.cpu().numpy(), dtype=np.bool_),
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(
                    handle,
                    metadata_json=arrays["metadata_json"],
                    body_pose=arrays["body_pose"],
                    betas=arrays["betas"],
                    global_orient=arrays["global_orient"],
                    transl=arrays["transl"],
                    K_fullimg=arrays["K_fullimg"],
                    keypoints_2d_px=arrays["keypoints_2d_px"],
                    boxes_xys_px=arrays["boxes_xys_px"],
                    observed_mask=arrays["observed_mask"],
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @staticmethod
    def _write_selection_preview(
        path: Path,
        *,
        video_path: Path,
        camera: GvhmrCameraSelection,
        boxes: torch.Tensor,
        keypoints: torch.Tensor,
    ) -> None:
        import cv2

        frame_count = int(boxes.shape[0])
        indices = tuple(sorted({0, frame_count // 2, frame_count - 1}))
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open preview video: {video_path}")
        panels: list[NDArray[np.uint8]] = []
        polygon = np.asarray(
            camera.polygon_px(
                width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
            dtype=np.int32,
        )
        points = keypoints.detach().cpu().numpy()
        boxes_np = boxes.detach().cpu().numpy()
        try:
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Could not read preview frame {frame_index}: {video_path}"
                    )
                cv2.polylines(frame, [polygon], True, (255, 160, 0), 4)
                center_x, center_y, size = boxes_np[frame_index]
                half = size / 2.0
                cv2.rectangle(
                    frame,
                    (int(center_x - half), int(center_y - half)),
                    (int(center_x + half), int(center_y + half)),
                    (0, 255, 255),
                    4,
                )
                pose = points[frame_index]
                for left, right in COCO17_SKELETON:
                    if pose[left, 2] > 0.2 and pose[right, 2] > 0.2:
                        cv2.line(
                            frame,
                            tuple(np.rint(pose[left, :2]).astype(int)),
                            tuple(np.rint(pose[right, :2]).astype(int)),
                            (0, 255, 0),
                            3,
                        )
                cv2.putText(
                    frame,
                    f"{camera.camera_id} {camera.player_role} frame={frame_index}",
                    (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                panels.append(cast(NDArray[np.uint8], cv2.resize(frame, (640, 360))))
        finally:
            capture.release()
        preview = np.concatenate(panels, axis=1)
        ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError(f"Could not encode selection preview: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(path)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 used to bind artifacts to a selection config."""
    digest = file_sha256(Path(path))
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("file_sha256 returned an invalid digest.")
    return digest


__all__ = [
    "COLLECTION_SCHEMA_VERSION",
    "GvhmrCameraSelection",
    "GvhmrMotionExtractor",
    "GvhmrSelectionConfig",
    "RECORD_SCHEMA_VERSION",
    "SELECTION_SCHEMA_VERSION",
    "load_model_runtime",
    "normalize_vitpose_confidence",
    "sha256_file",
]

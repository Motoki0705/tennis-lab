"""Read-only SLCS clip catalog and mixed player/ball review payloads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.base.visualization.review.court import court_edges, court_keypoints
from src.tasks.base.visualization.review.payload import ScenePayload, pack_entity_frames
from src.tasks.slcs.configuration import SLCS_DATA_SCHEMA, SLCSDataRuntimeConfig
from src.tasks.slcs.data.annotation import (
    SLCSDataIndex,
    load_slcs_annotation,
    slcs_annotation_dir,
)
from src.tasks.slcs.data.dataset import SLCSDataConfig, load_clip_arrays
from src.tasks.slcs.visualization.review.inspection import ClipInspection
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    DatasetClipRecord,
    validate_id_component,
)
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def default_data_config(dataset_root: Path) -> SLCSDataConfig:
    """Read the training data defaults from their sole configuration source."""
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=dataset_root,
        checkpoint_root=dataset_root,
        artifact_root=dataset_root,
        output_root=dataset_root,
        cache_root=dataset_root,
        external_asset_root=dataset_root,
    )
    resolver = PathResolver(roots)
    path = resolver.resolve(PathRole.PROJECT, "src/tasks/slcs/configs")
    with initialize_config_dir(config_dir=str(path), version_base="1.3"):
        config = compose(config_name="data/default")
    raw = OmegaConf.to_container(config.data, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a data configuration object.")
    validated = SLCS_DATA_SCHEMA.validate(cast(dict[str, object], raw))
    return SLCSDataRuntimeConfig.from_mapping(dict(validated), resolver).pipeline


class SLCSDatasetReviewService:
    """Browse canonical video/clip manifests without DINO, splits or inference.

    The shared UI calls video IDs ``form`` and clip names ``scene``. Only a
    selected clip is decoded; the catalog never opens scene archives.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        video_ids: Sequence[str] | None = None,
        config: SLCSDataConfig | None = None,
    ) -> None:
        self.root = Path(dataset_root).expanduser().resolve(strict=True)
        self._video_ids = None if video_ids is None else tuple(dict.fromkeys(video_ids))
        self._index()
        self.config = default_data_config(self.root) if config is None else config
        self._verified_publication = lru_cache(maxsize=2)(self._verify_publication)
        self._cached_inspection = lru_cache(maxsize=2)(self._load_inspection)
        self._cached_image = lru_cache(maxsize=8)(self._load_image)
        self._cached_payload = lru_cache(maxsize=2)(self._load_payload)

    def _index(self) -> SLCSDataIndex:
        self._contained(self.root / "dataset.json")
        index = SLCSDataIndex.load(self.root)
        if self._video_ids is not None:
            for video_id in self._video_ids:
                validate_id_component(video_id, field_name="video_id")
                if video_id not in index.video_ids():
                    raise ValueError(f"Unknown video ID: {video_id!r}.")
            index = SLCSDataIndex(
                root=index.root,
                clips=tuple(c for c in index.clips if c.video_id in self._video_ids),
            )
        return index

    def _contained(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"Dataset path escapes the dataset root: {path}.")
        return resolved

    def catalog(self) -> dict[str, Any]:
        index = self._index()
        return {
            "task": "slcs",
            "root": str(self.root),
            "entity": "scene",
            "skeleton": None,
            "forms": [
                {
                    "name": video_id,
                    "path": f"videos/{video_id}/clips",
                    "mode": "pseudo-labels",
                    "scene_count": sum(c.video_id == video_id for c in index.clips),
                }
                for video_id in index.video_ids()
            ],
        }

    def scenes(self, form: str) -> dict[str, Any]:
        validate_id_component(form, field_name="video_id")
        index = self._index()
        if form not in index.video_ids():
            raise ValueError(f"Unknown video ID: {form!r}.")
        return {
            "form": form,
            "scenes": [c.clip_name for c in index.clips if c.video_id == form],
        }

    def _record(self, form: str, scene_id: str) -> DatasetClipRecord:
        validate_id_component(form, field_name="video_id")
        validate_id_component(scene_id, field_name="clip_name")
        for record in self._index().clips:
            if (record.video_id, record.clip_name) == (form, scene_id):
                return record
        raise ValueError(f"Unknown clip: {form}/{scene_id}.")

    def _revision(self, record: DatasetClipRecord) -> str:
        clip_dir = self._contained(self.root / record.path)
        annotation = slcs_annotation_dir(clip_dir)
        marker = json.loads((annotation / "annotation.json").read_text())
        relative = marker.get("scene_result")
        if not isinstance(relative, str) or not relative:
            raise ValueError("Annotation marker requires an explicit scene_result.")
        scene_path = self._contained(annotation / relative)
        index_path = self._contained(annotation / "scene.json")
        index = json.loads(index_path.read_text())
        manifest = ClipManifest.load(clip_dir)
        dependencies = [
            annotation / ref["path"] for ref in index.get("artifacts", {}).values()
        ]
        media = [
            manifest.media_path(camera, must_exist=False)
            for camera in manifest.camera_ids
        ]
        digest = hashlib.sha256()
        for path in (
            self.root / "dataset.json",
            clip_dir / "clip.json",
            annotation / "annotation.json",
            scene_path,
            scene_path.with_suffix(".metadata.json"),
            index_path,
            *dependencies,
            *media,
            self.root / "splits.json",
            clip_dir / "annotations/dino_v3/annotation.json",
        ):
            path = self._contained(path)
            if path.is_file():
                stat = path.stat()
                identity = f"{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}"
            else:
                identity = "missing"
            digest.update(f"{path.relative_to(self.root)}:{identity}".encode())
        revision = digest.hexdigest()[:20]
        # Full checksums and component lineage are still verified. Only an
        # unchanged publication reuses verification, avoiding hashing a large
        # SMPL export for every RGB playback request.
        self._verified_publication(record, revision)
        return revision

    def _verify_publication(self, record: DatasetClipRecord, revision: str) -> None:
        from src.tennis_scene.pipeline.storage.scene_index import annotation_scene_path

        annotation = slcs_annotation_dir(self._contained(self.root / record.path))
        marker = json.loads((annotation / "annotation.json").read_text())
        self._contained(annotation_scene_path(annotation, marker))

    def _load_inspection(
        self, record: DatasetClipRecord, revision: str
    ) -> ClipInspection:
        manifest = ClipManifest.load(self._contained(self.root / record.path))
        if manifest.clip_id != record.clip_id:
            raise ValueError("Dataset index and clip manifest disagree on clip_id.")
        clip = load_clip_arrays(manifest, config=self.config)
        scene = load_slcs_annotation(manifest)
        inspection = ClipInspection(clip, scene, self.config)
        return inspection

    def inspection_frame(
        self, form: str, scene_id: str, camera: str, frame: int, revision: str
    ) -> dict[str, Any]:
        record = self._record(form, scene_id)
        if self._revision(record) != revision:
            raise RuntimeError("Clip changed on disk. Reload the clip.")
        result = self._cached_inspection(record, revision).frame(camera, frame)
        if self._revision(record) != revision:
            raise RuntimeError("Clip changed on disk. Reload the clip.")
        return {**result, "revision": revision}

    def image(
        self, form: str, scene_id: str, camera: str, frame: int, revision: str
    ) -> bytes:
        self.inspection_frame(form, scene_id, camera, frame, revision)
        record = self._record(form, scene_id)
        result = self._cached_image(record, revision, camera, frame)
        if self._revision(record) != revision:
            raise RuntimeError("Clip changed on disk. Reload the clip.")
        return result

    def _load_image(
        self, record: DatasetClipRecord, revision: str, camera: str, frame: int
    ) -> bytes:
        manifest = self._cached_inspection(record, revision).clip.manifest
        path = self._contained(manifest.media_path(camera, must_exist=False))
        if not path.is_file():
            raise FileNotFoundError("Saved clip media is missing.")
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
        try:
            if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, frame):
                raise RuntimeError("Cannot seek the saved clip media.")
            ok, bgr = cap.read()
            if not ok or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
                raise RuntimeError("Cannot decode the requested clip frame.")
            if bgr.shape[:2] != (manifest.height, manifest.width):
                raise ValueError("Decoded media dimensions disagree with clip.json.")
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("Cannot encode the source frame.")
            return encoded.tobytes()
        finally:
            cap.release()

    def payload(
        self, form: str, scene_id: str, revision: str | None = None
    ) -> ScenePayload:
        record = self._record(form, scene_id)
        current = self._revision(record)
        if revision is not None and revision != current:
            raise RuntimeError("Clip changed on disk. Reload the clip.")
        result = self._cached_payload(record, current)
        if self._revision(record) != current:
            self._cached_payload.cache_clear()
            raise RuntimeError("Clip changed on disk. Reload the clip.")
        return result

    def scene(
        self, form: str, scene_id: str, revision: str | None = None
    ) -> dict[str, Any]:
        document: dict[str, Any] = self.payload(form, scene_id, revision).document
        return document

    def buffer(self, form: str, scene_id: str, revision: str | None = None) -> bytes:
        binary: bytes = self.payload(form, scene_id, revision).binary
        return binary

    def _load_payload(self, record: DatasetClipRecord, revision: str) -> ScenePayload:
        inspection = self._cached_inspection(record, revision)
        clip = inspection.clip
        manifest = clip.manifest
        review = inspection.summary()
        features = manifest.clip_dir / "annotations/dino_v3/annotation.json"
        split = self.root / "splits.json"
        review["preparation_note"] = (
            f"DINO完成marker {'あり' if features.is_file() else 'なし'} / "
            f"split file {'あり' if split.is_file() else 'なし'}。"
            "レビューには不要。内容検証・学習準備完了の判定はしていません。"
        )
        if not np.isfinite(clip.fps) or clip.fps <= 0:
            raise ValueError("Clip FPS must be finite and positive.")
        for name, shape in (
            ("player_position_norm", (self.config.num_players, clip.num_frames, 3)),
            ("ball_position_norm", (clip.num_frames, 3)),
        ):
            if getattr(clip, name).shape != shape:
                raise ValueError(f"{record.clip_id}: {name} must have shape {shape}.")
        scale = np.asarray(COURT_COORD_SCALE_XYZ, dtype=np.float32)
        players = clip.player_position_norm * scale
        ball = clip.ball_position_norm * scale
        rotation = clip.player_rotation.copy()
        # Missing labels must be absent, with finite transport values. The
        # presence mask also prevents trails from bridging those frame gaps.
        players[~clip.player_label_valid] = 0
        rotation[~clip.player_label_valid] = 0
        ball[~clip.ball_label_valid] = 0
        player_buffer = pack_entity_frames(
            players[:, :, None, :],
            orientation=rotation,
            presence=clip.player_label_valid,
        )
        ball_buffer = pack_entity_frames(
            ball[None, :, None, :], presence=clip.ball_label_valid[None, :]
        )
        entities = [
            {
                "kind": "player",
                "slots": int(players.shape[0]),
                "colors": ["#3a6fb0", "#d1623f"],
                "orientation": True,
            },
            {"kind": "ball", "slots": 1, "colors": ["#e07828"], "orientation": False},
        ]
        for entity in entities:
            entity.update(
                joint_count=1,
                frames=clip.num_frames,
                presence=True,
                joint_names=None,
                skeleton=None,
            )
        quality = self.config.quality
        document = {
            "task": "slcs",
            "form": record.video_id,
            "dataset": f"slcs/{manifest.dataset_id}/{record.video_id}",
            "mode": "pseudo-labels",
            "scene_id": record.clip_name,
            "clip_id": record.clip_id,
            "revision": revision,
            "fps": clip.fps,
            "frame_count": clip.num_frames,
            "units": "m",
            "coordinate_frame": review["coordinate_frame"],
            "court": {
                "keypoints": court_keypoints(None).tolist(),
                "edges": [list(edge) for edge in court_edges()],
            },
            "cameras": [],
            "source_camera_ids": list(manifest.camera_ids),
            "entities": entities,
            "description": (
                "疑似ラベル / 選手はルート位置と向き（手前→奥の順） / "
                "品質マスク外は非表示 / 実測GTなし / " + review["calibration_note"]
            ),
            "inspection": review,
            "quality": {
                "min_player_confidence": quality.min_player_confidence,
                "min_ball_cameras": quality.min_ball_cameras,
                "player_valid_frames": clip.player_label_valid.sum(axis=1).tolist(),
                "ball_valid_frames": int(clip.ball_label_valid.sum()),
            },
        }
        return ScenePayload(
            document=document,
            binary=player_buffer + bytes(-len(player_buffer) % 4) + ball_buffer,
        )

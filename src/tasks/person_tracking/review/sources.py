"""Explicit readers for a fixed BallPose campaign and current v5 component stores."""

from __future__ import annotations

import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, cast

import av
import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import SHARDS_DIR, BallFrameStore, shard_name
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tennis_scene.chat_annotation.player_pose.dataset import PlayerPoseStore
from src.tennis_scene.chat_annotation.player_pose.reviews import validate_and_remap
from src.tennis_scene.pipeline.storage.codec import unpack_value
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
    read_component_descriptor,
)
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathResolver, PathRole

from .model import ReviewSequence


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def contained_file(root: Path, fragment: str) -> Path:
    relative = Path(fragment)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Artifact path escapes its declared directory: {fragment}")
    return path


@dataclass(frozen=True)
class CheckedFile:
    path: Path
    sha256: str
    identity: tuple[int, int, int, int, int]

    @classmethod
    def open(cls, path: Path, expected: str | None = None) -> CheckedFile:
        before = cls._identity(path)
        actual = dual_sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"File checksum differs from the saved provenance: {path}")
        after = cls._identity(path)
        if before != after:
            raise ValueError(
                f"Review source changed during checksum verification: {path}"
            )
        item = cls(path, actual, after)
        return item

    @staticmethod
    def _identity(path: Path) -> tuple[int, int, int, int, int]:
        stat = path.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def unchanged(self) -> None:
        if self._identity(self.path) != self.identity:
            raise ValueError(
                f"Review source changed while the viewer was open: {self.path}"
            )

    def record(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


class StoredJpegFrames:
    def __init__(
        self, store: BallFrameStore, index: int, checked: tuple[CheckedFile, ...]
    ) -> None:
        self.store, self.clip, self.checked = store, store.clips[index], checked

    def read(self, frame: int) -> NDArray[np.uint8]:
        for item in self.checked:
            item.unchanged()
        image = self.store.read_bgr(self.store.row_of(self.clip, frame))
        for item in self.checked:
            item.unchanged()
        return image


class VideoFrames:
    """Validate exact RGB hash, dimensions and decoded PTS once; seek by real PTS."""

    def __init__(self, source: dict[str, Any], resolver: PathResolver) -> None:
        self.path = resolver.validate(PathRole.DATA, Path(source["path"]))
        self.file = CheckedFile.open(self.path, str(source["sha256"]))
        pts = []
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "SLICE"
            stream.codec_context.thread_count = 2
            if stream.time_base is None or stream.average_rate is None:
                raise ValueError("Source video has no actual time base / frame rate")
            self.time_base = str(stream.time_base)
            if not math.isclose(
                float(stream.average_rate), float(source["fps"]), rel_tol=1e-5
            ):
                raise ValueError(
                    "Decoded source frame rate differs from its declaration"
                )
            for frame in container.decode(stream):
                if frame.pts is None or str(frame.time_base) != self.time_base:
                    raise ValueError("Source frames must retain actual consistent PTS")
                if (frame.width, frame.height) != (source["width"], source["height"]):
                    raise ValueError(
                        "Decoded source dimensions differ from the tracking source"
                    )
                pts.append(frame.pts)
        self.file.unchanged()
        if len(pts) != source["num_frames"]:
            raise ValueError(
                "Decoded source frame count differs from the tracking source"
            )
        self.pts = np.asarray(pts, np.int64)
        if (np.diff(self.pts) <= 0).any():
            raise ValueError("Decoded source PTS must be strictly increasing")
        self._frames: OrderedDict[int, NDArray[np.uint8]] = OrderedDict()
        self._lock = Lock()

    def read(self, frame: int) -> NDArray[np.uint8]:
        self.file.unchanged()
        if not 0 <= frame < len(self.pts):
            raise IndexError(frame)
        with self._lock:
            if frame not in self._frames:
                self._frames[frame] = self._decode(frame)
                if len(self._frames) > 8:
                    self._frames.popitem(last=False)
            self._frames.move_to_end(frame)
            self.file.unchanged()
            return self._frames[frame]

    def _decode(self, frame: int) -> NDArray[np.uint8]:
        expected = int(self.pts[frame])
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "SLICE"
            stream.codec_context.thread_count = 2
            container.seek(expected, stream=stream, backward=True)
            for decoded in container.decode(stream):
                if decoded.pts == expected:
                    image = cast(NDArray[np.uint8], decoded.to_ndarray(format="bgr24"))
                    self.file.unchanged()
                    return image
                if decoded.pts is not None and decoded.pts > expected:
                    break
        raise ValueError(
            f"Could not decode exact source PTS {expected} for frame {frame}"
        )


class BallPoseCampaign:
    def __init__(self, campaign: Path, dataset: Path, resolver: PathResolver) -> None:
        self.directory = campaign
        identity = read_json(campaign / "identity.json")
        self.config_file = CheckedFile.open(
            campaign / "config.json", identity["config_sha256"]
        )
        self.plan_file = CheckedFile.open(
            campaign / "plan.json", identity["plan_sha256"]
        )
        self.config, self.plan = (
            read_json(self.config_file.path),
            read_json(self.plan_file.path),
        )
        if self.plan["schema"] != "ball_store_player_pose_plan.v1":
            raise ValueError("Unsupported BallPose campaign plan schema")
        store_path = resolver.validate(PathRole.ARTIFACT, Path(self.config["store"]))
        self.store_files = tuple(
            CheckedFile.open(store_path / name, expected)
            for name, expected in self.config["store_hashes"].items()
        )
        self.store = BallFrameStore(store_path)
        self.manifest_file = CheckedFile.open(dataset / "manifest.json")
        manifest = read_json(self.manifest_file.path)
        manifest_store = resolver.validate(
            PathRole.ARTIFACT, Path(manifest["ball_store"]["directory"])
        )
        if manifest_store != store_path:
            raise ValueError("Pose dataset is bound to a different RGB snapshot")
        self.pose_store = PlayerPoseStore(dataset)
        if (
            Path(self.pose_store.manifest["campaign"]).resolve() != campaign
            or self.pose_store.ball_store.directory.resolve() != store_path
        ):
            raise ValueError(
                "Pose dataset is bound to a different campaign or RGB snapshot"
            )
        if len(self.plan["clips"]) != len(self.store.clips):
            raise ValueError("Campaign plan and snapshot clip axes differ")
        self.records: dict[str, dict[str, Any]] = {}
        for index, (planned, clip) in enumerate(
            zip(self.plan["clips"], self.store.clips, strict=True)
        ):
            if (
                planned["index"],
                planned["clip_id"],
                planned["frame_count"],
                planned["width"],
                planned["height"],
            ) != (index, clip.clip_id, clip.frame_count, clip.width, clip.height):
                raise ValueError("Campaign plan and RGB snapshot identity differ")
            path = campaign / "clips" / f"clip-{index:05d}"
            completed = (path / "generation.json").is_file()
            if completed and not (path / "tracks.npz").is_file():
                raise ValueError(
                    f"Completed generation is missing raw tracking: {clip.clip_id}"
                )
            status = self.pose_store.clips[clip.clip_id]["pose_status"]
            missing_reason = (
                "球presence条件でskip / raw未生成"
                if status == "skipped"
                else "raw tracking未生成 / 停止中"
            )
            key = f"ball:{index}"
            self.records[key] = {
                "key": key,
                "clip_id": clip.clip_id,
                "camera_id": clip.camera_id or "single_view",
                "source": clip.source,
                "split": clip.split,
                "family": "BallPose raw / ID区間",
                "available": completed,
                "frame_count": clip.frame_count,
                "status": status,
                "reason": None if completed else missing_reason,
                "index": index,
            }

    def catalog(self) -> dict[str, Any]:
        self.manifest_file.unchanged()
        statuses = Counter(
            entry["pose_status"] for entry in self.pose_store.manifest["clips"]
        )
        return {
            "name": "BallPose 固定snapshot",
            "clips": len(self.records),
            "available": sum(record["available"] for record in self.records.values()),
            "status_counts": dict(statuses),
            "description": "公開Ball storeとは別版。rawは選手選別前の全人物。approvedは匿名選手ID区間の採用で、bbox/pose精度の保証ではありません。",
            "records": list(self.records.values()),
        }

    def load(self, key: str) -> ReviewSequence:
        record = self.records[key]
        if not record["available"]:
            raise ValueError(str(record["reason"]))
        self.config_file.unchanged()
        self.plan_file.unchanged()
        self.manifest_file.unchanged()
        index = int(record["index"])
        clip = self.store.clips[index]
        root = self.directory / "clips" / f"clip-{index:05d}"
        generation_file = CheckedFile.open(root / "generation.json")
        generation = read_json(generation_file.path)
        if (
            generation["status"] != "complete"
            or generation["clip_id"] != clip.clip_id
            or generation["frame_count"] != clip.frame_count
        ):
            raise ValueError("Raw generation identity differs from the RGB clip")
        checked = tuple(
            CheckedFile.open(contained_file(root, name), expected)
            for name, expected in generation["files"].items()
        )
        raw_file = next(item for item in checked if item.path.name == "tracks.npz")
        raw_input = read_json(root / "input.json")
        if raw_input["clip_id"] != clip.clip_id:
            raise ValueError("Raw tracking input belongs to a different clip")
        shard = CheckedFile.open(
            self.store.directory / SHARDS_DIR / shard_name(index),
            raw_input["shard_sha256"],
        )
        with np.load(raw_file.path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        rows = self.store.clip_rows(clip)
        if not np.array_equal(
            arrays["frame_index"], self.store.frames["frame_index"][rows]
        ) or not np.array_equal(arrays["pts"], self.store.frames["pts"][rows]):
            raise ValueError(
                "Raw tracking frame/PTS axes differ from the fixed RGB snapshot"
            )
        observed = arrays["detection_rows"] >= 0
        entry = self.pose_store.clips[clip.clip_id]
        intervals: tuple[dict[str, Any], ...] = ()
        selected = None
        review_record = None
        if entry["pose_status"] == "approved":
            if entry["raw_tracks_sha256"] != raw_file.sha256:
                raise ValueError(
                    "Approved ID intervals belong to a different raw tracking archive"
                )
            adopted = self.pose_store.read_clip(clip.clip_id)
            assert adopted is not None
            review_file = CheckedFile.open(
                contained_file(self.pose_store.directory, entry["review_file"]),
                entry["review_sha256"],
            )
            review = read_json(review_file.path)
            expected_sheets = [
                f"frames-{start:06d}-{min(start + 20, clip.frame_count):06d}.jpg"
                for start in range(0, clip.frame_count, 20)
            ]
            remapped = validate_and_remap(
                arrays,
                review,
                clip_id=clip.clip_id,
                raw_hash=raw_file.sha256,
                required_sheets=expected_sheets,
            )
            if review["status"] != "approved" or any(
                not np.array_equal(remapped[name], adopted[name]) for name in adopted
            ):
                raise ValueError(
                    "Approved pose artifact differs from its saved raw ID interval assignments"
                )
            intervals = tuple(review["segments"])
            selected = np.zeros_like(observed)
            lookup = {int(track): row for row, track in enumerate(arrays["track_ids"])}
            for interval in intervals:
                if interval["role"] == "player":
                    row, start, stop = (
                        lookup[interval["raw_track_id"]],
                        interval["start_frame"],
                        interval["stop_frame"],
                    )
                    selected[row, start:stop] = observed[row, start:stop]
            review_record = review_file.record()
            checked += (review_file,)
        metadata = {
            "family": record["family"],
            "source": clip.source,
            "split": clip.split,
            "tracking_profile": generation["tracking_profile"],
            "status": entry["pose_status"],
            "selection_kind": "gpt_interval" if selected is not None else "unavailable",
            "selection_note": "GPTによる匿名選手ID区間。bbox/pose全体の精度保証ではありません。"
            if selected is not None
            else "ID区間は未採用。未レビューを非選手に変換しません。",
            "synthetic_note": "このraw archiveには補間boxは保存されていません。rawに存在しないboxは補間箱として描きません。",
            "label_note": "各観測のID/役割区間。未検出人物の被覆やbbox/pose精度の独立GTではありません。",
            "reference": None,
            "saved_link_candidates": generation["link_candidates"],
            "provenance": {
                "raw": raw_file.record(),
                "generation": generation_file.record(),
                "rgb_shard": shard.record(),
                "plan": self.plan_file.record(),
                "manifest": self.manifest_file.record(),
                "review": review_record,
                "snapshot": str(self.store.directory),
            },
            "schema": "BallPose raw tracks.npz",
        }
        return ReviewSequence(
            key,
            clip.clip_id,
            str(record["camera_id"]),
            clip.width,
            clip.height,
            arrays["frame_index"],
            arrays["pts"],
            clip.time_base,
            arrays["track_ids"],
            arrays["boxes"],
            observed,
            arrays["detection_rows"],
            tuple(tuple(ids) for ids in generation["source_track_ids"]),
            None,
            None,
            selected,
            None,
            intervals,
            None,
            metadata,
            StoredJpegFrames(
                self.store,
                index,
                (
                    *self.store_files,
                    self.manifest_file,
                    generation_file,
                    shard,
                    *checked,
                ),
            ),
        )


class ComponentStore:
    def __init__(
        self,
        directory: Path,
        resolver: PathResolver,
        references: dict[str, tuple[ClipLabels, CheckedFile]],
    ) -> None:
        self.directory, self.resolver, self.references = directory, resolver, references
        self.index_file = CheckedFile.open(directory / "scene.json")
        self.document = read_json(self.index_file.path)
        if self.document["schema"] != "tennis_scene_index_v1":
            raise ValueError("Unsupported component scene index")
        self.source = self.document["source"]
        self.source_videos = {
            video["camera_id"]: video for video in self.source["videos"]
        }
        self.records = {}
        for node, reference in self.document["artifacts"].items():
            if not node.startswith("person_tracking/"):
                continue
            camera = node.partition("/")[2]
            if camera not in self.source_videos:
                raise ValueError("Tracking camera has no declared source RGB")
            supported = (reference["schema"], reference["version"]) == (
                "person_tracks",
                5,
            )
            key = f"component:{self.index_file.sha256[:16]}:{camera}"
            self.records[key] = {
                "key": key,
                "clip_id": self.source["clip_id"],
                "camera_id": camera,
                "source": "meiji"
                if self.source["clip_id"].startswith("video_")
                else "component_store",
                "split": "固定評価artifact",
                "family": "Component store / person_tracks",
                "available": supported,
                "frame_count": self.source_videos[camera]["num_frames"],
                "status": "v5 saved"
                if supported
                else f"historical v{reference['version']}",
                "reason": None
                if supported
                else "旧schemaの比較artifact。現行v5へ暗黙変換しません。",
                "node": node,
            }
        if not self.records:
            raise ValueError("Store has no published person_tracking artifact")

    def catalog(self) -> dict[str, Any]:
        self.index_file.unchanged()
        return {
            "name": f"Component · {self.source['clip_id']}",
            "clips": 1,
            "available": sum(record["available"] for record in self.records.values()),
            "description": "保存済みcamera別tracking。v5の実観測と保存GSIを別maskで表示。旧schemaは履歴として列挙します。",
            "status_counts": {},
            "records": list(self.records.values()),
        }

    def payload(self, node: str) -> tuple[dict[str, Any], dict[str, Any]]:
        reference = self.document["artifacts"][node]
        assert_current_component_lineage(
            self.document, self.directory, {node: reference}
        )
        descriptor = read_component_descriptor(
            self.directory,
            reference,
            node=node,
            source_sha256=self.document["source_sha256"],
        )
        location = contained_file(self.directory, reference["path"]).parent
        payload = unpack_value(descriptor["payload"], location, descriptor["arrays"])
        if not isinstance(payload, dict):
            raise ValueError("Component tracking requires a structured payload")
        return payload, descriptor

    def load(self, key: str) -> ReviewSequence:
        self.index_file.unchanged()
        record = self.records[key]
        if not record["available"]:
            raise ValueError(str(record["reason"]))
        node, camera = str(record["node"]), str(record["camera_id"])
        raw, descriptor = self.payload(node)
        source = self.source_videos[camera]
        if (
            raw["camera_id"] != camera
            or raw["observed"].shape[1] != source["num_frames"]
        ):
            raise ValueError("Tracking camera/timeline differs from its source video")
        images = VideoFrames(source, self.resolver)
        evidence, reconstruction = raw["evidence"], raw["reconstruction"]
        if reconstruction is not None and not np.array_equal(
            reconstruction["observed"], raw["observed"]
        ):
            raise ValueError(
                "Reconstruction observation mask differs from the raw tracking"
            )
        selected, group_ids = None, None
        selection_note = "コート選別の保存artifactなし。選手/非選手は未判定です。"
        selection_node = f"player_selection/{camera}"
        selection_kind = "unavailable"
        if selection_node in self.document["artifacts"]:
            selection_ref = self.document["artifacts"][selection_node]
            if (selection_ref["schema"], selection_ref["version"]) != (
                "selected_player_tracks",
                2,
            ):
                raise ValueError(
                    "Selection must use the current selected_player_tracks v2 contract"
                )
            selection, selection_descriptor = self.payload(selection_node)
            if (
                selection_descriptor["dependencies"]["tracks"]
                != self.document["artifacts"][node]
                or not np.array_equal(selection["raw_track_ids"], raw["track_ids"])
                or selection["camera_id"] != camera
            ):
                raise ValueError(
                    "Selection belongs to a different raw tracking artifact"
                )
            selected = selection["selected"]
            group_ids = np.full(raw["observed"].shape, -1, np.int64)
            origins, group = selection["origin_rows"], selection["tracks"]
            if (
                origins.shape != group["observed"].shape
                or not np.array_equal(origins >= 0, group["observed"])
                or (origins < -1).any()
                or (origins >= len(raw["track_ids"])).any()
            ):
                raise ValueError("Court group origins differ from saved observations")
            groups, frames = np.nonzero(group["observed"])
            raw_rows = origins[groups, frames]
            if not selected[raw_rows, frames].all() or not np.array_equal(
                group["boxes_xyxy"][groups, frames], raw["boxes_xyxy"][raw_rows, frames]
            ):
                raise ValueError("Court group must preserve the original selected box")
            group_ids[raw_rows, frames] = group["track_ids"][groups]
            selection_kind = "court_candidate"
            selection_note = "保存済みコート選別の候補mask。group IDはraw tracker IDと別です。選手GTではありません。"
        label, label_record = None, None
        if self.source["clip_id"] in self.references:
            label, label_file = self.references[self.source["clip_id"]]
            label_file.unchanged()
            if label.num_frames != source["num_frames"] or camera not in label.cameras:
                raise ValueError(
                    "Partial reference camera/frame axes differ from the declared source"
                )
            label_record = label_file.record()
        settings = descriptor["identity"]["settings"]
        profile = settings.get("profile")
        metadata = {
            "family": record["family"],
            "source": record["source"],
            "split": record["split"],
            "tracking_profile": profile,
            "status": "固定評価artifact / 現行v5契約",
            "selection_kind": selection_kind,
            "selection_note": selection_note,
            "synthetic_note": "GSI補間は保存reconstruction maskだけを橙破線で表示。実観測/ラベル/選別へ追加しません。"
            if reconstruction is not None
            else "補間boxの保存なし。欠測を埋めません。",
            "label_note": "部分参照は保存観測boxの人物ラベル。未検出人物のrecallを保証しません。raw IDへ自動対応させません。"
            if label is not None
            else "参照ラベル未指定。未被覆を負例やGTに変換しません。",
            "reference": None
            if label is None
            else {
                "boxes": len(label.cameras[camera].frames),
                "provenance": label.provenance,
                "binding": "明示指定したclip/camera/frame。ラベルには独立RGB hashがありません。",
            },
            "saved_link_candidates": raw["offline_link_candidates"],
            "schema": "person_tracks v5",
            "provenance": {
                "index": self.index_file.record(),
                "descriptor": {
                    "path": str(
                        self.directory / self.document["artifacts"][node]["path"]
                    ),
                    "sha256": self.document["artifacts"][node]["sha256"],
                },
                "source_video": images.file.record(),
                "source_sha256": self.document["source_sha256"],
                "origin": descriptor["provenance"],
                "reference_labels": label_record,
            },
        }
        return ReviewSequence(
            key,
            str(self.source["clip_id"]),
            camera,
            int(source["width"]),
            int(source["height"]),
            np.arange(int(source["num_frames"]), dtype=np.int64),
            images.pts,
            images.time_base,
            np.array(raw["track_ids"], copy=True),
            np.array(raw["boxes_xyxy"], copy=True),
            np.array(raw["observed"], copy=True),
            None
            if evidence is None
            else np.array(evidence["detection_rows"], copy=True),
            tuple(tuple(ids) for ids in raw["source_track_ids"]),
            None
            if reconstruction is None
            else np.array(reconstruction["boxes"], copy=True),
            None
            if reconstruction is None
            else np.array(reconstruction["interpolated"], copy=True),
            None if selected is None else np.array(selected, copy=True),
            group_ids,
            (),
            label,
            metadata,
            images,
        )

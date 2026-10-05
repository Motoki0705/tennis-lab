"""Discover pose datasets and serve bounded overlays without running inference."""

from __future__ import annotations

from collections import Counter, OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any

from src.tasks.ball_detection.data.store import SHARDS_DIR, BallFrameStore, shard_name
from src.tennis_scene.chat_annotation.player_pose.dataset import PlayerPoseStore

from .readers import (
    file_stamp,
    frame_people,
    read_object,
    read_raw,
    validate_join,
    within,
)

STATUS_LABELS = {
    "approved": "採用済み",
    "needs_review": "要確認",
    "review_pending": "レビュー待ち",
    "not_generated": "未生成",
    "skipped": "対象外",
    "failed": "生成エラー",
}


class PlayerSource:
    def __init__(self, directory: Path, *, data_root: Path, output_root: Path) -> None:
        self.directory = within(directory, data_root)
        self.path = within(directory / "manifest.json", self.directory)
        self.stamp = file_stamp(self.path)
        self.manifest = read_object(self.path)
        if (
            self.manifest.get("schema") != "ball_detection_player_poses.v1"
            or self.manifest.get("coordinate_system") != "stored_jpeg_pixels"
        ):
            raise ValueError("Unsupported player pose schema/coordinate system")
        binding = self.manifest["ball_store"]
        if set(binding["hashes"]) != {"metadata.json", "index.npz"}:
            raise ValueError("Pose store must pin metadata.json and index.npz")
        self.store_path = within(Path(binding["directory"]), data_root, output_root)
        self.store_stamps = {
            name: file_stamp(within(self.store_path / name, self.store_path))
            for name in binding["hashes"]
        }
        self.version = str(read_object(self.store_path / "metadata.json")["version"])
        self.campaign = within(Path(self.manifest["campaign"]), output_root)
        progress_path = within(self.campaign / "generation_status.json", self.campaign)
        progress = read_object(progress_path) if progress_path.is_file() else {}
        failed = set(progress.get("failed", []))
        self.entries = {entry["clip_id"]: entry for entry in self.manifest["clips"]}
        if len(self.entries) != len(self.manifest["clips"]):
            raise ValueError("Duplicate player manifest clip IDs")
        self.statuses: dict[str, dict[str, Any]] = {}
        for clip_id, entry in self.entries.items():
            if type(entry["index"]) is not int or entry["index"] < 0:
                raise ValueError("Invalid pose clip index")
            root = self.raw_root(entry)
            generation = within(root / "generation.json", self.campaign)
            raw = read_object(generation) if generation.is_file() else {}
            available = raw.get("status") == "complete"
            status = entry["pose_status"]
            if status == "pending":
                review_path = within(root / "review_status.json", self.campaign)
                review = read_object(review_path) if review_path.is_file() else {}
                if available:
                    status = (
                        "needs_review"
                        if review.get("status") == "needs_review"
                        else "review_pending"
                    )
                else:
                    status = (
                        "failed"
                        if raw.get("status") == "failed" or entry["index"] in failed
                        else "not_generated"
                    )
            if status not in STATUS_LABELS:
                raise ValueError(f"Unknown player pose status: {status}")
            self.statuses[clip_id] = {
                "status": status,
                "label": STATUS_LABELS[status],
                "reviewed_available": entry["pose_status"] == "approved",
                "raw_available": available,
                "reason": entry.get("reason", ""),
            }
        self.reader: PlayerPoseStore | None = None
        self.lock = RLock()
        self.cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self.cache_bytes = 0

    def raw_root(self, entry: dict[str, Any]) -> Path:
        root: Path = within(
            self.campaign / "clips" / f"clip-{entry['index']:05d}", self.campaign
        )
        return root

    def status(self, clip_id: str) -> dict[str, Any]:
        return dict(
            self.statuses.get(
                clip_id,
                {
                    "status": "not_generated",
                    "label": "未生成",
                    "reviewed_available": False,
                    "raw_available": False,
                    "reason": "outside_pose_campaign",
                },
            )
        )

    def _unchanged(self) -> None:
        if file_stamp(within(self.path, self.directory)) != self.stamp or any(
            file_stamp(within(self.store_path / name, self.store_path)) != stamp
            for name, stamp in self.store_stamps.items()
        ):
            raise ValueError(
                "Player dataset changed; refresh the catalog before continuing"
            )

    def arrays(self, public: BallFrameStore, clip_id: str, mode: str) -> dict[str, Any]:
        with self.lock:
            self._unchanged()
            if self.reader is None:
                self.reader = PlayerPoseStore(self.directory)
            entry = self.entries[clip_id]
            original = self.reader.ball_store
            shard_stamps = tuple(
                file_stamp(
                    within(
                        store.directory
                        / SHARDS_DIR
                        / shard_name(store.clip_by_id(clip_id).index),
                        store.directory,
                    )
                )
                for store in (public, original)
            )
            if mode == "reviewed":
                artifacts = [
                    within(self.directory / entry[key], self.directory)
                    for key in ("file", "review_file")
                ]
            else:
                root = self.raw_root(entry)
                artifacts = [
                    within(root / name, self.campaign)
                    for name in ("generation.json", "tracks.npz", "input.json")
                ]
            key = (
                public,
                clip_id,
                mode,
                shard_stamps,
                tuple(file_stamp(p) for p in artifacts),
            )
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            validate_join(public, original, clip_id)
            data: dict[str, Any] | None
            if mode == "reviewed":
                data = self.reader.read_clip(clip_id)
                if data is None:
                    raise ValueError("Reviewed poses are unavailable")
            else:
                data = read_raw(self.raw_root(entry), original, clip_id)
            size = sum(array.nbytes for array in data.values())
            budget = 128 * 1024 * 1024
            while self.cache and (
                len(self.cache) >= 2 or self.cache_bytes + size > budget
            ):
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= sum(array.nbytes for array in old.values())
            if size <= budget:
                self.cache[key] = data
                self.cache_bytes += size
            return data


class PlayerCatalog:
    def __init__(self, data_root: Path, project_root: Path) -> None:
        self.data_root, self.output_root = data_root, project_root / "outputs"
        self.sources: dict[str, PlayerSource] = {}
        self.descriptions: list[dict[str, Any]] | None = None

    def discover(self) -> list[dict[str, Any]]:
        if self.descriptions is not None:
            return self.descriptions
        descriptions = []
        for path in sorted((self.data_root / "ball_detection").glob("*/manifest.json")):
            identity = f"players/{path.parent.name}"
            try:
                within(path, self.data_root)
                manifest = read_object(path)
                if not str(manifest.get("schema", "")).startswith(
                    "ball_detection_player_poses."
                ):
                    continue
                source = PlayerSource(
                    path.parent, data_root=self.data_root, output_root=self.output_root
                )
                self.sources[identity] = source
                descriptions.append(
                    {
                        "id": identity,
                        "label": path.parent.name,
                        "ball_version": source.version,
                        "available": True,
                        "clips": len(source.entries),
                        "reviewed_clips": sum(status["reviewed_available"] for status in source.statuses.values()),
                        "raw_clips": sum(status["raw_available"] for status in source.statuses.values()),
                        "status_counts": dict(Counter(status["status"] for status in source.statuses.values())),
                        "ball_store_path": str(source.store_path),
                    }
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                descriptions.append(
                    {
                        "id": identity,
                        "label": path.parent.name,
                        "available": False,
                        "error": str(error),
                    }
                )
        self.descriptions = descriptions
        return descriptions

    def source(self, identity: str) -> PlayerSource:
        self.discover()
        if identity not in self.sources:
            raise ValueError(f"Unknown or unavailable player dataset: {identity}")
        return self.sources[identity]

    def preview(
        self,
        identity: str,
        public: BallFrameStore,
        clip_id: str,
        start: int,
        count: int,
        mode: str,
    ) -> dict[str, Any]:
        if mode not in {"reviewed", "raw"}:
            raise ValueError("Player mode must be reviewed or raw")
        clip = public.clip_by_id(clip_id)
        if not 1 <= count <= 64 or start < 0 or start + count > clip.frame_count:
            raise ValueError("Player preview range is outside the clip")
        source = self.source(identity)
        source._unchanged()
        status = source.status(clip_id)
        available = status[
            "reviewed_available" if mode == "reviewed" else "raw_available"
        ]
        data = source.arrays(public, clip_id, mode) if available else None
        breaks = public.frames["segment_break"][public.clip_rows(clip)]
        return {
            "dataset": identity,
            "clip_id": clip_id,
            "mode": mode,
            **status,
            "available": available,
            "items": [
                {
                    "index": index,
                    "people": frame_people(data, index, breaks)
                    if data is not None
                    else [],
                }
                for index in range(start, start + count)
            ],
        }

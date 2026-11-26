"""Read the original #2363 observations without using superseding scene exports."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, unpack_value
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256

Value = TypeVar("Value")
HISTORICAL_RUN = Path("knowledge/runs/run-i2363-trt-player-video")
WIDTH_REVISION = "63e9730fa23a57d46e7df4d9d4fed9e9204e9f94"
UPSTREAM_REVISION = "61fef16091952b5c0df06f89b32603656a29a745"
CAMERAS = ("cam0", "cam1", "cam2", "cam3")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_file(revision: str, path: str) -> str:
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)


@dataclass(frozen=True)
class ArchivedClip:
    directory: Path
    source_record: dict[str, Any]
    stop: dict[str, Any]
    camera_metrics: dict[str, dict[str, str]]
    upstream_ids: dict[str, str]
    alley_tracks: dict[str, int]

    @property
    def name(self) -> str:
        return self.directory.name

    def store(self) -> StoreReader:
        store = StoreReader(self.directory / "annotations/tennis_scene/scene.json")
        if (
            store.source["clip_id"] != self.source_record["clip_id"]
            or store.camera_ids != CAMERAS
        ):
            raise ValueError(
                f"{self.name}: scene source or camera order disagrees with #2363"
            )
        for node, artifact_id in self.upstream_ids.items():
            if store.references[node]["artifact_id"] != artifact_id:
                raise ValueError(
                    f"{self.name}/{node}: artifact differs from the recorded #2363 upstream"
                )
        return store

    def source(self, store: StoreReader, *, verify_media: bool) -> ClipSource:
        videos = []
        for header in store.source["videos"]:
            camera = header["camera_id"]
            video = SourceVideo(
                **{**header, "path": self.directory / "media" / f"{camera}.mp4"}
            )
            if any(
                getattr(video, key) != self.source_record[key]
                for key in ("num_frames", "width", "height", "fps")
            ):
                raise ValueError(
                    f"{self.name}/{camera}: video metadata differs from #2363"
                )
            if verify_media and dual_sha256(video.path) != video.sha256:
                raise ValueError(
                    f"{self.name}/{camera}: video content differs from #2363"
                )
            videos.append(video)
        return ClipSource(store.source["clip_id"], tuple(videos))


class Archive:
    def __init__(self, root: Path) -> None:
        receipt = read_json(HISTORICAL_RUN / "full-video.json")
        self.receipt = {
            "i2363_commit": receipt["commit"],
            "width_study_commit": WIDTH_REVISION,
            "upstream_reference_commit": UPSTREAM_REVISION,
        }
        for name in ("progress.json", "outcome.json"):
            reference = receipt["artifacts"][name]
            path = root / reference["path"]
            if dual_sha256(path) != reference["sha256"]:
                raise ValueError(f"{name} differs from the #2363 receipt")
            self.receipt[name] = reference["sha256"]
        self.progress = read_json(root / receipt["artifacts"]["progress.json"]["path"])
        outcome = read_json(root / receipt["artifacts"]["outcome.json"]["path"])[
            "statuses"
        ]
        request = read_json(root / "input.json")
        dataset = root / "inputs/dataset"
        dataset_sha256 = dual_sha256(dataset / "dataset.json")
        if (
            dataset_sha256 != request["dataset_sha256"]
            or dataset_sha256 != receipt["dataset"]["manifest_sha256"]
        ):
            raise ValueError("Dataset manifest differs from #2363")
        self.receipt["dataset.json"] = dataset_sha256
        records = {row["clip_id"]: row for row in request["clips"]}
        clips = self.progress["clips"]
        with (HISTORICAL_RUN / "clips.csv").open(newline="") as handle:
            clip_summary = {row["clip_id"]: row for row in csv.DictReader(handle)}
        with (HISTORICAL_RUN / "camera-tracks.csv").open(newline="") as handle:
            camera_metrics = {
                (row["clip_id"], row["camera"]): row for row in csv.DictReader(handle)
            }
        if (
            (self.progress["dataset_id"], self.progress["video_id"])
            != ("meiji_4cam", "video_000")
            or not set(records) == set(clips) == set(clip_summary) == set(outcome)
            or len(records) != 69
        ):
            raise ValueError("Expected the same 69 clips in the #2363 records")
        width_losses = json.loads(
            git_file(
                WIDTH_REVISION, "knowledge/runs/run-i2440-player-selection/loss.json"
            )
        )
        alley_tracks = {
            row["clip_id"]: {
                camera: track["track_id"] for camera, track in row["cameras"].items()
            }
            for row in width_losses["stopped"]
        }
        upstream_csv = git_file(
            UPSTREAM_REVISION,
            "knowledge/runs/run-meiji-doubles-player-reconstruction/cpu-upstream.csv",
        )
        upstream_ids: dict[str, dict[str, str]] = {name: {} for name in records}
        for row in csv.DictReader(upstream_csv.splitlines()):
            upstream_ids[row["clip_id"]][row["node"]] = row["artifact_id"]
        self.clips = {}
        for name, row in sorted(clips.items()):
            summary = clip_summary[name]
            if row["status"] != outcome[name] or row["status"] != summary["status"]:
                raise ValueError(f"{name}: status differs between #2363 records")
            if row["status"] != "failed":
                continue
            if row["reason"] != summary["reason"] or len(upstream_ids[name]) != 16:
                raise ValueError(
                    f"{name}: stop or upstream references differ from #2363"
                )
            self.clips[name] = ArchivedClip(
                dataset / records[name]["path"],
                records[name],
                row,
                {c: camera_metrics[name, c] for c in CAMERAS},
                upstream_ids[name],
                alley_tracks.get(name, {}),
            )
        if len(self.clips) != 14:
            raise ValueError("Expected the 14 stopped #2363 clips")


def read_artifact(store: StoreReader, node: str, output: type[Value]) -> Value:
    descriptor = store.descriptor(node)
    directory = (store.root / store.references[node]["path"]).parent
    return ArtifactCodec(output).load(
        descriptor["payload"], directory, descriptor["arrays"]
    )


def read_fields(
    store: StoreReader, node: str, fields: tuple[str, ...]
) -> dict[str, Any]:
    descriptor = store.descriptor(node)
    directory = (store.root / store.references[node]["path"]).parent
    result = {}
    for field in fields:
        value = descriptor["payload"]
        for part in field.split("."):
            value = value[part]
        result[field] = unpack_value(value, directory, descriptor["arrays"])
    return result


@dataclass(frozen=True)
class CameraRows:
    offsets: np.ndarray
    source_rows: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    track_ids: np.ndarray
    track_rows: np.ndarray
    selected: np.ndarray
    group_ids: np.ndarray
    group_origins: np.ndarray

    @classmethod
    def load(cls, clip: ArchivedClip, store: StoreReader, camera: str) -> CameraRows:
        names = [
            f"{name}/{camera}"
            for name in ("person_detection", "person_tracking", "player_selection")
        ]
        expected = (
            ("person_detections", 2),
            ("person_tracks", 7),
            ("selected_player_tracks", 3),
        )
        for node, schema in zip(names, expected, strict=True):
            reference = store.references[node]
            if (reference["schema"], reference["version"]) != schema:
                raise ValueError(f"{clip.name}/{node}: unrecognized schema or version")
        detection = read_fields(
            store,
            names[0],
            ("frame_offsets", "source_rows", "boxes_xyxy", "confidence"),
        )
        track = read_fields(
            store,
            names[1],
            ("track_ids", "observed", "evidence.detection_rows", "boxes_xyxy"),
        )
        select = read_fields(
            store,
            names[2],
            (
                "raw_track_ids",
                "selected",
                "tracks.track_ids",
                "origin_rows",
                "tracks.evidence.detection_rows",
            ),
        )
        result = cls(
            *detection.values(),
            track["track_ids"],
            track["evidence.detection_rows"],
            select["selected"],
            select["tracks.track_ids"],
            select["origin_rows"],
        )
        metrics = clip.camera_metrics[camera]
        tracked, frames = np.nonzero(track["observed"])
        source = result.track_rows[tracked, frames]
        indices = np.searchsorted(result.source_rows, source)
        groups, group_frames = np.nonzero(result.group_origins >= 0)
        origins = result.group_origins[groups, group_frames]
        if (
            len(result.offsets) != store.frame_count + 1
            or not np.array_equal(result.track_rows >= 0, track["observed"])
            or np.any(indices >= len(result.source_rows))
            or not np.array_equal(result.source_rows[indices], source)
            or not np.all(
                (result.offsets[frames] <= indices)
                & (indices < result.offsets[frames + 1])
            )
            or not np.array_equal(
                result.boxes[indices], track["boxes_xyxy"][tracked, frames]
            )
            or len(np.unique(source)) != len(source)
            or not np.array_equal(result.track_ids, select["raw_track_ids"])
            or not np.all(result.selected[origins, group_frames])
            or not np.array_equal(
                result.track_rows[origins, group_frames],
                select["tracks.evidence.detection_rows"][groups, group_frames],
            )
            or result.track_ids.tolist() != json.loads(metrics["stable_track_ids"])
            or len(frames) != int(metrics["tracking_observations"])
            or result.group_ids.tolist() != json.loads(metrics["selected_group_ids"])
            or (result.group_origins >= 0).sum(axis=1).tolist()
            != json.loads(metrics["selected_group_frames"])
            or result.selected.sum() != int(metrics["retained_raw_observations"])
        ):
            raise ValueError(
                f"{clip.name}/{camera}: detection rows, track origins or selection differ from #2363"
            )
        return result

    def rows_at(self, frame: int) -> list[dict[str, Any]]:
        tracks = {
            int(row): k for k, row in enumerate(self.track_rows[:, frame]) if row >= 0
        }
        groups = {
            int(raw): int(group)
            for raw, group in zip(
                self.group_origins[:, frame], self.group_ids, strict=True
            )
            if raw >= 0
        }
        start, end = self.offsets[frame : frame + 2]
        rows = []
        for i in range(start, end):
            row = int(self.source_rows[i])
            track = tracks.get(row)
            rows.append(
                {
                    "row": row,
                    "box": self.boxes[i].tolist(),
                    "score": float(self.scores[i]),
                    "track": int(self.track_ids[track]) if track is not None else None,
                    "selected": bool(self.selected[track, frame])
                    if track is not None
                    else False,
                    "group": groups.get(track) if track is not None else None,
                }
            )
        return rows

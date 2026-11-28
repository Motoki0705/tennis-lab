"""Read shot identities and checked data references from multiview-v4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import file_digest, sha256
from pathlib import Path
from typing import Any

import numpy as np

from src.tennis_scene.ball_event_labels import EVENT_CLASSES_V4, EVENT_SUBTYPES_V4

SPLITS = ("train", "val", "test")


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True, slots=True)
class FileRef:
    path: Path
    sha256: str

    @classmethod
    def from_manifest(
        cls, root: Path, row: dict[str, Any], *, within: Path | None = None
    ) -> FileRef:
        relative = Path(row["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(within or root):
            raise ValueError(f"V4 file path escapes its directory: {relative}")
        return cls(path, row["sha256"])

    def verify(self) -> Path:
        if file_sha256(self.path) != self.sha256:
            raise ValueError(f"V4 file SHA-256 mismatch: {self.path}")
        return self.path

    def read_json(self) -> dict[str, Any]:
        data = self.path.read_bytes()
        if sha256(data).hexdigest() != self.sha256:
            raise ValueError(f"V4 file SHA-256 mismatch: {self.path}")
        document = json.loads(data)
        if not isinstance(document, dict):
            raise ValueError(f"Expected a V4 JSON object: {self.path}")
        return document


@dataclass(frozen=True, slots=True)
class V4Shot:
    source_event_id: str
    frame: int
    subtype: str
    frame_uncertainty: int
    court_side: int


@dataclass(frozen=True, slots=True)
class V4Clip:
    clip_id: str
    split: str
    frames: int
    fps: float
    scene: FileRef
    metadata: FileRef
    event_component: FileRef
    clip_json: FileRef
    cameras: dict[str, FileRef]
    annotation_sha256: str
    shot_count: int

    def shots(self, archive: np.lib.npyio.NpzFile) -> list[V4Shot]:
        """Join the exported arrays to adopted V4 evidence in original row order."""
        payload = self.event_component.read_json()["payload"]
        evidence = payload["event_evidence"]
        events = evidence["events"]
        columns = tuple(
            archive[f"ball_event_{field}"]
            for field in ("frame", "kind", "subtype", "frame_uncertainty", "side")
        )
        if (int(archive["num_frames"]), float(archive["fps"])) != (
            self.frames,
            self.fps,
        ) or any(column.shape != (len(events),) for column in columns):
            raise ValueError(f"{self.clip_id}: event evidence and archive disagree")
        shots = []
        seen: set[str] = set()
        sides = evidence["player_hint_reference"]
        for event, (frame, kind, subtype, uncertainty, side) in zip(
            events, zip(*columns, strict=True), strict=True
        ):
            source_id = event["source_event_id"]
            if (
                not isinstance(source_id, str)
                or not source_id
                or source_id in seen
                or not 0 <= frame < self.frames
                or uncertainty < 0
                or side not in (-1, 0, 1)
                or not 0 <= kind < len(EVENT_CLASSES_V4)
                or not 0 <= subtype < len(EVENT_SUBTYPES_V4)
                or (
                    event["frame"],
                    event["kind"],
                    event["subtype"],
                    event["frame_uncertainty"],
                )
                != (
                    frame,
                    EVENT_CLASSES_V4[kind],
                    EVENT_SUBTYPES_V4[subtype],
                    uncertainty,
                )
                or sides.get(event.get("player_hint"), 0) != side
            ):
                raise ValueError(
                    f"{self.clip_id}: invalid V4 event match for {source_id!r}"
                )
            seen.add(source_id)
            if event["kind"] == "shot":
                shots.append(
                    V4Shot(
                        source_id,
                        int(frame),
                        event["subtype"],
                        int(uncertainty),
                        int(side),
                    )
                )
        if len(shots) != self.shot_count:
            raise ValueError(f"{self.clip_id}: shot count disagrees with the manifest")
        return shots


@dataclass(frozen=True, slots=True)
class V4Benchmark:
    benchmark_id: str
    manifest_sha256: str
    clips: dict[str, V4Clip]

    @classmethod
    def load(cls, root: Path, manifest_sha256: str) -> V4Benchmark:
        root = root.resolve()
        manifest = FileRef(root / "manifest.json", manifest_sha256).read_json()
        if (
            manifest["schema"] != "tennis_multiview_benchmark_v1"
            or manifest["scene_schema_version"] != 4
        ):
            raise ValueError("Expected a SceneResult v4 benchmark manifest")
        assignments: dict[str, str] = {}
        for split in SPLITS:
            section = manifest["splits"][split]
            if len(section["clip_ids"]) != section["clips"]:
                raise ValueError(f"V4 manifest has an inconsistent {split} clip count")
            for clip_id in section["clip_ids"]:
                if clip_id in assignments:
                    raise ValueError(f"Duplicate V4 split assignment: {clip_id}")
                assignments[clip_id] = split
        clips = {}
        for row in manifest["clips"]:
            clip_id = row["clip_id"]
            if clip_id in clips or clip_id not in assignments:
                raise ValueError(f"Duplicate or unassigned V4 clip: {clip_id}")
            scene_dir = FileRef.from_manifest(root, row["scene"]).path.parent
            clip = V4Clip(
                clip_id,
                assignments[clip_id],
                row["frames"],
                row["fps"],
                FileRef.from_manifest(root, row["exports"]["scene/scene"]),
                FileRef.from_manifest(root, row["exports"]["scene/metadata"]),
                FileRef.from_manifest(
                    scene_dir, row["ball_events"]["component"], within=root
                ),
                FileRef.from_manifest(root, row["clip_json"]),
                {
                    camera["camera_id"]: FileRef.from_manifest(root, camera)
                    for camera in row["cameras"]
                },
                row["ball_events"]["annotation_sha256"],
                row["ball_events"]["by_kind"]["shot"],
            )
            clips[clip_id] = clip
        if clips.keys() != assignments.keys():
            raise ValueError("V4 split assignment includes missing clips")
        return cls(manifest["benchmark_id"], manifest_sha256, clips)

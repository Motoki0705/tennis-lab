"""Read a frozen dataset, saved component evidence and exact source detection rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.player_association.pseudo_annotation.job import read_json
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    load_dataset_manifest,
)
from src.tennis_scene.pipeline.storage.codec import unpack_value
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera

SCENE_INDEX = "annotations/tennis_scene/scene.json"


@dataclass(frozen=True)
class FrozenClip:
    clip: ClipManifest
    store: StoreReader
    outcome: dict[str, Any]
    media_sha256: dict[str, str]

    def read(
        self, node: str, schema: str, version: int, *fields: str
    ) -> dict[str, Any]:
        """Decode the declared fields with the standard component codec."""
        reference = self.store.references[node]
        if (reference["schema"], reference["version"]) != (schema, version):
            raise ValueError(f"{self.clip.clip_id}: unsupported component for {node}")
        desc = self.store.descriptor(node)
        directory = (self.store.root / reference["path"]).resolve().parent
        result = {}
        for field in fields:
            value = desc["payload"]
            for key in field.split("."):
                value = value[key]
            result[field] = unpack_value(value, directory, desc["arrays"])
        return result

    def detections(
        self, camera: str
    ) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.int64]]:
        out = self.read(
            f"person_detection/{camera}",
            "person_detections",
            2,
            "camera_id",
            "frame_offsets",
            "boxes_xyxy",
            "source_rows",
        )
        offsets, boxes, rows = (
            out["frame_offsets"],
            out["boxes_xyxy"],
            out["source_rows"],
        )
        if (
            out["camera_id"] != camera
            or offsets.shape != (self.clip.num_frames + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(boxes)
            or (np.diff(offsets) < 0).any()
            or boxes.shape != (len(rows), 4)
            or (np.diff(rows) <= 0).any()
        ):
            raise ValueError(
                f"{self.clip.clip_id}/{camera}: invalid detection timeline"
            )
        return offsets, boxes, rows

    def local_cameras(self) -> dict[str, PinholeCamera]:
        calibration = self.read(
            "court_calibration", "local_court_calibration", 1, "calibration"
        )
        return {
            data["camera_id"]: PinholeCamera(
                data["camera_id"],
                data["intrinsic"],
                data["rotation"],
                data["translation"],
            )
            for view in calibration["calibration"]["views"]
            for data in [view["camera"]]
        }

    def raw(self, camera: str, *, rows: bool = False) -> dict[str, Any]:
        fields = ("evidence.detection_rows",) if rows else ()
        return self.read(
            f"person_tracking/{camera}",
            "person_tracks",
            7,
            "camera_id",
            "track_ids",
            "boxes_xyxy",
            "observed",
            *fields,
        )

    def selection(self, camera: str, *, origins: bool = False) -> dict[str, Any]:
        fields = ("origin_rows",) if origins else ()
        return self.read(
            f"player_selection/{camera}",
            "selected_player_tracks",
            3,
            "camera_id",
            "raw_track_ids",
            "selected",
            "tracks.track_ids",
            "tracks.observed",
            *fields,
        )

    def assignments(self) -> dict[str, dict[int, NDArray[np.int64]]] | None:
        if "player_association" not in self.store.references:
            if self.outcome["status"] == "generated":
                raise ValueError(
                    f"{self.clip.clip_id}: generated clip missing association"
                )
            return None
        values = self.read(
            "player_association",
            "person_identities",
            3,
            "camera_ids",
            "local_track_ids",
            "player_ids",
        )
        return {
            camera: {
                int(group): row
                for group, row in zip(groups, ids, strict=True)
                if group >= 0
            }
            for camera, groups, ids in zip(
                values["camera_ids"],
                values["local_track_ids"],
                values["player_ids"],
                strict=True,
            )
        }


class FrozenRun:
    """The snapshot SHA-256 binds the dataset index, original media and adopted artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest = self.root / "SHA256SUMS"
        self.sha256 = dual_sha256(manifest)
        self.checksums: dict[str, str] = {}
        for line in manifest.read_text().splitlines():
            digest, name = line.split("  ", 1)
            if (
                name in self.checksums
                or Path(name).is_absolute()
                or ".." in Path(name).parts
            ):
                raise ValueError(f"Invalid snapshot manifest path: {name}")
            self.checksums[name] = digest
        dataset_path = self.root / "dataset"
        self.verify(dataset_path / "dataset.json")
        self.dataset = load_dataset_manifest(dataset_path)
        self.progress: dict[str, Any] = read_json(
            self.verify(self.root / "progress.json")
        )
        if (
            set(self.progress["clips"]) != set(self.dataset.clips)
            or self.progress["dataset_id"] != self.dataset.dataset_id
        ):
            raise ValueError("Saved progress does not cover the frozen dataset")

    def verify(self, path: Path) -> Path:
        if dual_sha256(path) != self.checksums[path.relative_to(self.root).as_posix()]:
            raise ValueError(f"Snapshot checksum mismatch: {path}")
        return path

    def clip(self, clip_id: str) -> FrozenClip:
        record = self.dataset.clips[clip_id]
        directory = self.root / "dataset" / record.path
        self.verify(directory / "clip.json")
        clip = ClipManifest.load(directory)
        store = StoreReader(self.verify(directory / SCENE_INDEX))
        if (
            clip.clip_id != clip_id
            or clip.dataset_id != self.dataset.dataset_id
            or store.source["clip_id"] != clip_id
            or store.camera_ids != clip.camera_ids
            or store.frame_count != clip.num_frames
            or store.size != (clip.width, clip.height)
        ):
            raise ValueError(
                f"Snapshot recording grid differs from the adopted store: {clip_id}"
            )
        media_sha = {}
        for video in store.source["videos"]:
            camera = video["camera_id"]
            path = clip.media_path(camera)
            expected = self.checksums[path.relative_to(self.root).as_posix()]
            if (
                (video["width"], video["height"], video["num_frames"])
                != (clip.width, clip.height, clip.num_frames)
                or abs(video["fps"] - clip.fps) > 1e-5
                or expected != video["sha256"]
            ):
                raise ValueError(
                    f"Synchronized media identity mismatch: {clip_id}/{camera}"
                )
            media_sha[camera] = expected
        return FrozenClip(clip, store, self.progress["clips"][clip_id], media_sha)

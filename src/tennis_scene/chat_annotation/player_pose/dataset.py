from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_detection.data.store import BallFrameStore

from .storage import digest, read_json


class PlayerPoseStore:
    """Reviewed poses on stored-JPEG pixels, with raw scores and explicit missing masks."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.manifest: dict[str, Any] = read_json(directory / "manifest.json")
        if self.manifest["schema"] != "ball_detection_player_poses.v1":
            raise ValueError("Unsupported player-pose schema")
        store_path = Path(self.manifest["ball_store"]["directory"])
        for name, expected in self.manifest["ball_store"]["hashes"].items():
            if digest(store_path / name) != expected:
                raise ValueError("Pose dataset belongs to a different ball store")
        self.ball_store = BallFrameStore(store_path)
        self.clips = {c["clip_id"]: c for c in self.manifest["clips"]}
        if set(self.clips) != {c.clip_id for c in self.ball_store.clips}:
            raise ValueError("Pose manifest must account for every ball-store clip")

    def read_clip(self, clip_id: str) -> dict[str, Any] | None:
        entry = self.clips[clip_id]
        if entry["pose_status"] == "skipped":
            return None
        if entry["pose_status"] != "approved":
            raise RuntimeError(
                f"Player poses not approved: {clip_id} ({entry['pose_status']})"
            )
        path = self.directory / entry["file"]
        if (
            not path.resolve().is_relative_to(self.directory.resolve())
            or digest(path) != entry["sha256"]
        ):
            raise ValueError("Pose artifact path/checksum mismatch")
        review = self.directory / entry["review_file"]
        if (
            not review.resolve().is_relative_to(self.directory.resolve())
            or digest(review) != entry["review_sha256"]
        ):
            raise ValueError("Review artifact path/checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        clip = self.ball_store.clip_by_id(clip_id)
        rows = self.ball_store.clip_rows(clip)
        if not np.array_equal(
            arrays["frame_index"], self.ball_store.frames["frame_index"][rows]
        ) or not np.array_equal(arrays["pts"], self.ball_store.frames["pts"][rows]):
            raise ValueError("Pose/frame time axes differ from ball images")
        frames, people = clip.frame_count, len(arrays["player_ids"])
        expected = {
            "boxes_xyxy": (frames, people, 4),
            "keypoints": (frames, people, 17, 3),
            "observed": (frames, people),
            "raw_track_ids": (frames, people),
            "detection_rows": (frames, people),
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"Pose axis mismatch: {name}")
        if (
            not np.isfinite(arrays["keypoints"]).all()
            or not np.isfinite(arrays["boxes_xyxy"]).all()
        ):
            raise ValueError("Nonfinite pose coordinates")
        absent = ~arrays["observed"]
        if (
            (arrays["keypoints"][absent] != 0).any()
            or (arrays["boxes_xyxy"][absent] != 0).any()
            or (arrays["raw_track_ids"][absent] != -1).any()
        ):
            raise ValueError("Missing pose observations must stay explicitly masked")
        return arrays

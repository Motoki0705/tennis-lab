"""Decode historical manual association records; no pipeline/UI execution."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.utils.io import load_json, save_json

LOGGER = logging.getLogger(__name__)


@dataclass
class PlayerAssociationSegment:
    """A temporal assignment segment with frame interval [start_frame, end_frame)."""

    start_frame: int
    end_frame: int
    assignments: NDArray[np.int32]  # (P, N), local GVHMR player axis per camera

    def to_dict(self) -> dict:
        """Convert segment to JSON-serializable dict."""
        return {
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "assignments": self.assignments.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlayerAssociationSegment:
        """Create segment from dict."""
        return cls(
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            assignments=np.asarray(data["assignments"], dtype=np.int32),
        )


@dataclass
class PlayerAssociationResult:
    """Manual player association result.

    Attributes:
        camera_ids: Camera identifiers aligned to N.
        canonical_player_ids: Stable player IDs aligned to output P.
        segments: Temporal assignments. Each assignment is shaped (P, N) and maps
            canonical player/camera to a local GVHMR player axis index.
        reference_camera: Camera ID used for SMPL arrays without a camera axis.
    """

    camera_ids: list[str]
    canonical_player_ids: NDArray[np.int32]
    segments: list[PlayerAssociationSegment]
    reference_camera: str

    def to_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        return {
            "camera_ids": self.camera_ids,
            "canonical_player_ids": self.canonical_player_ids.tolist(),
            "segments": [segment.to_dict() for segment in self.segments],
            "reference_camera": self.reference_camera,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlayerAssociationResult:
        """Create result from dict."""
        return cls(
            camera_ids=[str(camera_id) for camera_id in data["camera_ids"]],
            canonical_player_ids=np.asarray(
                data["canonical_player_ids"],
                dtype=np.int32,
            ),
            segments=[
                PlayerAssociationSegment.from_dict(segment)
                for segment in data["segments"]
            ],
            reference_camera=str(data["reference_camera"]),
        )

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        save_json(self.to_dict(), path)
        LOGGER.info(f"Saved player association result to {path}")

    @classmethod
    def load(cls, path: str | Path) -> PlayerAssociationResult:
        """Load result from JSON file."""
        return cls.from_dict(load_json(path))

    def reference_camera_index(self) -> int:
        """Return the reference camera index."""
        try:
            return self.camera_ids.index(self.reference_camera)
        except ValueError as exc:
            raise ValueError(
                f"reference_camera={self.reference_camera!r} is not in camera_ids"
            ) from exc

    def validate(
        self,
        *,
        num_frames: int,
        local_player_counts: Sequence[int],
    ) -> tuple[bool, list[str]]:
        """Validate association coverage and local player assignments."""
        errors: list[str] = []
        num_cameras = len(self.camera_ids)
        num_players = int(self.canonical_player_ids.shape[0])

        if self.canonical_player_ids.ndim != 1:
            errors.append(
                "canonical_player_ids must have shape (P,), "
                f"got {self.canonical_player_ids.shape}"
            )
            num_players = 0
        if len(local_player_counts) != num_cameras:
            errors.append(
                "local_player_counts length must match camera_ids length, "
                f"got {len(local_player_counts)} and {num_cameras}"
            )
        if not self.segments:
            errors.append("segments must not be empty")
            return False, errors
        if self.reference_camera not in self.camera_ids:
            errors.append(
                f"reference_camera={self.reference_camera!r} is not in camera_ids"
            )

        expected_start = 0
        for segment_index, segment in enumerate(self.segments):
            if segment.start_frame != expected_start:
                errors.append(
                    f"segment {segment_index} must start at {expected_start}, "
                    f"got {segment.start_frame}"
                )
            if segment.end_frame <= segment.start_frame:
                errors.append(
                    f"segment {segment_index} must have end_frame > start_frame"
                )
            if segment.assignments.shape != (num_players, num_cameras):
                errors.append(
                    f"segment {segment_index} assignments must have shape "
                    f"{(num_players, num_cameras)}, got {segment.assignments.shape}"
                )
                expected_start = segment.end_frame
                continue
            for camera_index in range(num_cameras):
                assigned = segment.assignments[:, camera_index]
                if np.any(assigned < 0):
                    errors.append(
                        f"segment {segment_index} camera {camera_index} has "
                        "negative local player index"
                    )
                if camera_index < len(local_player_counts) and np.any(
                    assigned >= int(local_player_counts[camera_index])
                ):
                    errors.append(
                        f"segment {segment_index} camera {camera_index} local "
                        "player index is out of range"
                    )
                if len(np.unique(assigned)) != len(assigned):
                    errors.append(
                        f"segment {segment_index} camera {camera_index} assigns "
                        "the same local player to multiple canonical players"
                    )
            expected_start = segment.end_frame

        if expected_start != num_frames:
            errors.append(
                f"segments must cover [0, {num_frames}), got end {expected_start}"
            )
        return len(errors) == 0, errors



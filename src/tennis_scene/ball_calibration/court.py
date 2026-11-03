"""Shared camera-local raw court measurements, separate from fitted geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CourtLineObservations:
    frame_indices: NDArray[np.int64]  # One source frame per measured line point.
    uv_px: NDArray[np.float64]
    scores: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        n = len(self.frame_indices)
        if (
            self.frame_indices.shape != (n,)
            or self.frame_indices.dtype != np.int64
            or n < 1
            or np.any(self.frame_indices < 0)
            or self.uv_px.shape != (n, 2)
            or self.uv_px.dtype != np.float64
            or self.scores.shape != (n,)
            or self.scores.dtype != np.float64
            or self.valid.shape != (n,)
            or self.valid.dtype != bool
            or not np.isfinite(self.scores).all()
            or np.any((self.scores < 0) | (self.scores > 1))
            or not np.isfinite(self.uv_px[self.valid]).all()
        ):
            raise ValueError("Invalid raw court LINE observation arrays")
        for array in (self.frame_indices, self.uv_px, self.scores, self.valid):
            array.setflags(write=False)


@dataclass(frozen=True, slots=True)
class CourtViewObservations:
    camera_id: str
    source_size_wh: tuple[int, int]
    source_num_frames: int
    frame_indices: NDArray[np.int64]  # Actual inference frames; never broadcast.
    keypoints_px: NDArray[np.float64]  # F,14,2; raw detector points, not fitted14.
    scores: NDArray[np.float64]
    valid: NDArray[np.bool_]
    selected_for_homography: NDArray[np.bool_]
    keypoint_schema: str
    template_orientations: tuple[str | None, ...]
    lines: CourtLineObservations | None
    line_missing_reason: str | None

    def __post_init__(self) -> None:
        n = len(self.frame_indices)
        if (
            not self.camera_id
            or type(self.keypoint_schema) is not str
            or not self.keypoint_schema
            or len(self.source_size_wh) != 2
            or any(type(x) is not int or x < 2 for x in self.source_size_wh)
            or type(self.source_num_frames) is not int
            or self.source_num_frames < 1
            or self.frame_indices.shape != (n,)
            or self.frame_indices.dtype != np.int64
            or n < 1
            or np.any(self.frame_indices < 0)
            or np.any(self.frame_indices >= self.source_num_frames)
            or np.any(np.diff(self.frame_indices) <= 0)
        ):
            raise ValueError("Raw court source camera/frame/size identity mismatch")
        if (
            self.keypoints_px.shape != (n, 14, 2)
            or self.keypoints_px.dtype != np.float64
            or self.scores.shape != (n, 14)
            or self.scores.dtype != np.float64
            or self.valid.shape != (n, 14)
            or self.valid.dtype != bool
            or self.selected_for_homography.shape != (n, 14)
            or self.selected_for_homography.dtype != bool
            or len(self.template_orientations) != n
            or any(
                orientation not in {None, "identity", "x_reflected"}
                for orientation in self.template_orientations
            )
            or not np.isfinite(self.scores).all()
            or np.any((self.scores < 0) | (self.scores > 1))
            or not np.isfinite(self.keypoints_px[self.valid]).all()
            or np.any(self.selected_for_homography & ~self.valid)
        ):
            raise ValueError("Invalid raw court keypoint/score/valid/selection arrays")
        if (self.lines is None) != bool(self.line_missing_reason):
            raise ValueError(
                "Missing court LINE observations require an explicit reason"
            )
        if self.lines is not None:
            if not np.isin(self.lines.frame_indices, self.frame_indices).all():
                raise ValueError(
                    "Court LINE frame axis differs from measured court frames"
                )
            self._in_image(self.lines.uv_px[self.lines.valid])
        self._in_image(self.keypoints_px[self.valid])
        for array in (
            self.frame_indices,
            self.keypoints_px,
            self.scores,
            self.valid,
            self.selected_for_homography,
        ):
            array.setflags(write=False)

    def _in_image(self, uv: NDArray[np.float64]) -> None:
        if np.any(uv < 0) or np.any(uv > np.asarray(self.source_size_wh) - 1):
            raise ValueError(
                "Valid raw court observations must use native image pixels"
            )

    def require_lines(self) -> CourtLineObservations:
        if self.lines is None:
            raise ValueError(
                f"{self.camera_id}: raw court LINE observations unavailable: {self.line_missing_reason}"
            )
        return self.lines


def court_views_from_diagnostics(
    diagnostics: dict[str, Any] | None,
    camera_ids: tuple[str, ...],
    source_size_wh: tuple[int, int],
    source_num_frames: int,
    *,
    require_lines: bool = False,
) -> tuple[CourtViewObservations, ...]:
    """Read raw JSON retained by court_calibration; never use its fitted14 arrays.

    Historical court_observations v2 stores raw KP/selection in diagnostics but
    only summary LINE support, not line coordinates. This absence is explicit.
    """
    if (
        diagnostics is None
        or diagnostics.get("output_keypoint_contract") != "camera_view_v2"
        or diagnostics.get("temporal_policy") != "static_first_frame_broadcast"
        or diagnostics.get("observed_frame_indices") != [0]
        or diagnostics.get("source_frame_count") != source_num_frames
        or not isinstance(diagnostics.get("cameras"), list)
        or tuple(c.get("camera_id") for c in diagnostics["cameras"]) != camera_ids
    ):
        raise ValueError("Missing or mismatched raw court camera/frame provenance")
    result = []
    for camera_id, camera in zip(camera_ids, diagnostics["cameras"], strict=True):
        frames = camera.get("frames")
        if not isinstance(frames, list) or len(frames) != 1:
            raise ValueError(f"{camera_id}: missing raw court source frame")
        row = frames[0]
        required = {
            "frame_index",
            "raw_keypoints_px",
            "raw_scores",
            "raw_valid",
            "selected_keypoints",
            "keypoint_schema",
            "kp_only_prosac_orientation",
            "original_size_hw",
            "output_keypoint_contract",
        }
        if (
            not required <= set(row)
            or type(row["frame_index"]) is not int
            or row["frame_index"] != 0
            or row["original_size_hw"] != list(reversed(source_size_wh))
            or row["output_keypoint_contract"] != "camera_view_v2"
        ):
            raise ValueError(
                f"{camera_id}: missing or mismatched raw court fields/frame/size"
            )
        valid = np.asarray(row["raw_valid"])
        selected: NDArray[np.bool_] = np.zeros((1, 14), bool)
        indices = row["selected_keypoints"]
        if (
            valid.dtype != bool
            or not isinstance(indices, list)
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < 14 for i in indices)
        ):
            raise ValueError(f"{camera_id}: invalid raw court validity or selection")
        selected[0, indices] = True
        lines: CourtLineObservations | None = None
        missing_reason: str | None = (
            "source court_observations v2 recorded LINE support summaries only"
        )
        if "raw_line_observations" in row:
            raw = row["raw_line_observations"]
            if (
                not isinstance(raw, dict)
                or set(raw) != {"uv_px", "scores", "valid"}
                or np.asarray(raw["valid"]).dtype != bool
            ):
                raise ValueError(f"{camera_id}: invalid raw court LINE fields")
            lines = CourtLineObservations(
                np.zeros(len(raw["uv_px"]), np.int64),
                np.asarray(raw["uv_px"], np.float64),
                np.asarray(raw["scores"], np.float64),
                np.asarray(raw["valid"], bool),
            )
            missing_reason = None
        view = CourtViewObservations(
            camera_id,
            source_size_wh,
            source_num_frames,
            np.array([0], np.int64),
            np.asarray(row["raw_keypoints_px"], np.float64)[None],
            np.asarray(row["raw_scores"], np.float64)[None],
            valid[None],
            selected,
            row["keypoint_schema"],
            (row["kp_only_prosac_orientation"],),
            lines,
            missing_reason,
        )
        if require_lines:
            view.require_lines()
        result.append(view)
    return tuple(result)

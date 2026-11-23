"""Version 1 frame candidates and their observation and quality contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from src.utils.geometry.triangulation import PointRejection

NODE = "multiview_pose"
SCHEMA = "frame_pose_candidates"
VERSION = 1


@dataclass(frozen=True)
class PoseConfig:
    min_joint_score: float = 0.25
    strong_joint_score: float = 0.55
    reprojection_px: float = 24.0  # At 1920 x 1080; scaled by source image diagonal.
    mixed_reprojection_px: float = 60.0
    min_ray_angle_deg: float = 1.0
    height_range_m: tuple[float, float] = (-0.4, 4.0)
    max_abs_xy_m: float = 40.0
    ambiguity_relative_gap: float = 0.05

    def __post_init__(self) -> None:
        if (
            not all(
                isfinite(value)
                for value in (
                    self.min_joint_score,
                    self.strong_joint_score,
                    self.reprojection_px,
                    self.mixed_reprojection_px,
                    self.min_ray_angle_deg,
                    *self.height_range_m,
                    self.max_abs_xy_m,
                    self.ambiguity_relative_gap,
                )
            )
            or not 0 < self.min_joint_score <= self.strong_joint_score
            or not 0 < self.reprojection_px < self.mixed_reprojection_px
            or not 0 < self.min_ray_angle_deg < 90
            or self.height_range_m[0] >= self.height_range_m[1]
            or self.max_abs_xy_m <= 0
            or not 0 <= self.ambiguity_relative_gap < 1
        ):
            raise ValueError("Invalid multiview pose thresholds")


class RowRejection(IntEnum):
    ASSIGNED = 0
    INSUFFICIENT_VIEWS = 1
    DEGENERATE_RAYS = 2
    INCONSISTENT_GEOMETRY = 3
    MIXED_POSE = 4
    CONFLICT = 5


@dataclass(frozen=True)
class FramePoseCandidates:
    camera_ids: tuple[str, ...]
    frame_offsets: NDArray[
        np.int64
    ]  # (T+1,); candidate ID is the index inside a frame slice.
    source_rows: NDArray[
        np.int64
    ]  # (M,V), -1 when the camera does not support the candidate.
    positions_m: NDArray[np.float32]  # (M,17,3) in the reference court coordinates.
    joint_valid: NDArray[np.bool_]  # (M,17)
    joint_rejection_codes: NDArray[np.uint8]  # (M,17), PointRejection codes.
    inlier_mask: NDArray[np.bool_]  # (M,V,17)
    reprojection_px: NDArray[np.float32]  # (M,V,17)
    reprojection_valid: NDArray[np.bool_]  # (M,V,17), includes measured outliers.
    scores: NDArray[np.float32]  # (M,)
    competing_assignment: NDArray[np.bool_]  # (M,), an alternative was compared.
    relative_score_gap: NDArray[np.float32]  # (M,), zero unless compared.
    ambiguous: NDArray[np.bool_]  # (M,), alternative within the configured margin.
    detection_offsets: tuple[NDArray[np.int64], ...]  # one (T+1,) array per camera.
    row_rejection_codes: tuple[
        NDArray[np.uint8], ...
    ]  # one (N,) array in DINO row order per camera.

    def __post_init__(self) -> None:
        count, views = len(self.positions_m), len(self.camera_ids)
        offsets = self.frame_offsets
        if (
            not 2 <= views <= 5
            or len(set(self.camera_ids)) != views
            or any(not c for c in self.camera_ids)
            or offsets.ndim != 1
            or offsets.dtype != np.int64
            or len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != count
            or (np.diff(offsets) < 0).any()
        ):
            raise ValueError("Invalid frame pose timeline or camera IDs")
        for array, dtype, shape in (
            (self.source_rows, np.int64, (count, views)),
            (self.positions_m, np.float32, (count, 17, 3)),
            (self.joint_valid, np.bool_, (count, 17)),
            (self.joint_rejection_codes, np.uint8, (count, 17)),
            (self.inlier_mask, np.bool_, (count, views, 17)),
            (self.reprojection_px, np.float32, (count, views, 17)),
            (self.reprojection_valid, np.bool_, (count, views, 17)),
            (self.scores, np.float32, (count,)),
            (self.competing_assignment, np.bool_, (count,)),
            (self.relative_score_gap, np.float32, (count,)),
            (self.ambiguous, np.bool_, (count,)),
        ):
            if (
                array.shape != shape
                or array.dtype != dtype
                or not np.isfinite(array).all()
            ):
                raise ValueError("Invalid frame pose array shape, dtype or value")
        present = self.source_rows >= 0
        if (
            (self.source_rows < -1).any()
            or (present.sum(axis=1) < 2).any()
            or not np.array_equal(self.inlier_mask.any(axis=2), present)
            or (
                self.reprojection_valid
                & ~(present[..., None] & self.joint_valid[:, None])
            ).any()
            or (self.inlier_mask & ~self.reprojection_valid).any()
            or not np.array_equal(self.inlier_mask.sum(axis=1) >= 2, self.joint_valid)
            or not np.array_equal(
                self.joint_rejection_codes == PointRejection.VALID, self.joint_valid
            )
            or (self.joint_rejection_codes > max(PointRejection)).any()
            or (self.positions_m[~self.joint_valid] != 0).any()
            or (self.reprojection_px[~self.reprojection_valid] != 0).any()
            or (self.reprojection_px < 0).any()
        ):
            raise ValueError("Frame pose support, errors and validity disagree")
        if (
            (self.scores <= 0).any()
            or (self.relative_score_gap < 0).any()
            or (self.relative_score_gap > 1).any()
            or (self.relative_score_gap[~self.competing_assignment] != 0).any()
            or (self.ambiguous & ~self.competing_assignment).any()
        ):
            raise ValueError("Invalid frame pose assignment diagnostics")
        if (
            len(self.detection_offsets) != views
            or len(self.row_rejection_codes) != views
        ):
            raise ValueError(
                "Every camera requires raw detection offsets and rejection codes"
            )
        frames: NDArray[np.int64] = np.repeat(
            np.arange(len(offsets) - 1), np.diff(offsets)
        )
        for camera, (axis, codes) in enumerate(
            zip(self.detection_offsets, self.row_rejection_codes, strict=True)
        ):
            if (
                axis.shape != offsets.shape
                or axis.dtype != np.int64
                or axis[0] != 0
                or (np.diff(axis) < 0).any()
                or codes.shape != (axis[-1],)
                or codes.dtype != np.uint8
                or (codes > max(RowRejection)).any()
            ):
                raise ValueError("Invalid DINO row diagnostics")
            rows = self.source_rows[present[:, camera], camera]
            selected_frames = frames[present[:, camera]]
            if (
                (rows < axis[selected_frames]).any()
                or (rows >= axis[selected_frames + 1]).any()
                or len(np.unique(rows)) != len(rows)
                or not np.array_equal(
                    np.sort(rows), np.flatnonzero(codes == RowRejection.ASSIGNED)
                )
            ):
                raise ValueError(
                    "A source row is reused, belongs to another frame or disagrees with its status"
                )

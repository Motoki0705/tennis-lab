"""Method-visible types deliberately contain neither annotations nor reference 3D."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.inference.trajectory_gate import TrajectoryGateConfig
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_points import BallPointsOutput
from src.tennis_scene.pipeline.components.camera_geometry import LocalCourtCalibration
from src.utils.geometry.temporal_smoothing import TemporalSmoothingConfig
from src.utils.geometry.triangulation import PinholeCamera


def readonly_arrays(value: Any) -> None:
    """Freeze arrays recursively without copying native heatmaps."""
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            readonly_arrays(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            readonly_arrays(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            readonly_arrays(item)


@dataclass(frozen=True, slots=True)
class DetectorReplayConfig:
    score_threshold: float
    subpixel_refine: bool
    trajectory_gate: TrajectoryGateConfig

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.score_threshold)
            or not 0 <= self.score_threshold <= 1
        ):
            raise ValueError("Invalid detector replay threshold")
        if type(self.subpixel_refine) is not bool:
            raise ValueError("subpixel_refine must be bool")


@dataclass(frozen=True, slots=True)
class ViewInput:
    camera: PinholeCamera
    local_calibration: LocalCourtCalibration
    half_turn: bool
    detections: BallDetectionOutput
    points: BallPointsOutput
    detector_config: DetectorReplayConfig
    smoothing_config: TemporalSmoothingConfig

    def __post_init__(self) -> None:
        camera_id = self.camera.camera_id
        if any(
            name != camera_id
            for name in (
                self.local_calibration.camera.camera_id,
                self.detections.camera_id,
                self.points.camera_id,
            )
        ):
            raise ValueError("View camera axis mismatch")
        if (
            self.detections.score_semantics != "model_score"
            or self.detections.evidence is None
        ):
            raise ValueError(
                "Method input requires model evidence; annotation leakage is forbidden"
            )
        if type(self.half_turn) is not bool:
            raise ValueError("Court side must be bool")
        expected = self.local_calibration.camera.half_turned(self.half_turn)
        if any(
            not np.array_equal(getattr(expected, field), getattr(self.camera, field))
            for field in ("intrinsic", "rotation", "translation")
        ):
            raise ValueError(
                "Aligned camera disagrees with local calibration and court side"
            )
        if self.points.source_size_wh != self.detections.evidence.source_size_wh:
            raise ValueError("View source image size mismatch")
        if not np.array_equal(self.points.frame_indices, self.detections.frame_indices):
            raise ValueError("View frame axis mismatch")
        evidence = self.detections.evidence
        if not np.array_equal(
            evidence.selected_window_start + evidence.selected_time_index,
            self.detections.frame_indices,
        ):
            raise ValueError(
                "Evidence window provenance disagrees with the source frame axis"
            )
        if not np.array_equal(
            self.points.smoothing.source_observed, self.detections.observed
        ):
            raise ValueError("Point source mask differs from detector observations")
        readonly_arrays(self)


@dataclass(frozen=True, slots=True)
class MethodInput:
    clip_id: str
    frame_indices: NDArray[np.int64]
    fps: float
    source_size_wh: tuple[int, int]
    views: tuple[ViewInput, ...]

    def __post_init__(self) -> None:
        if not self.clip_id or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Invalid method clip/fps")
        if (
            self.frame_indices.dtype != np.int64
            or self.frame_indices.ndim != 1
            or not len(self.frame_indices)
            or not np.array_equal(
                self.frame_indices, np.arange(len(self.frame_indices))
            )
        ):
            raise ValueError("Method input requires the complete source frame axis")
        if (
            len(self.source_size_wh) != 2
            or any(type(n) is not int or n < 2 for n in self.source_size_wh)
            or len(self.views) < 2
            or len(set(self.camera_ids)) != len(self.views)
        ):
            raise ValueError("Method requires distinct cameras and source image size")
        for view in self.views:
            if view.points.source_size_wh != self.source_size_wh or not np.array_equal(
                view.detections.frame_indices, self.frame_indices
            ):
                raise ValueError("Method camera/frame/image axes disagree")
        readonly_arrays(self)

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(view.camera.camera_id for view in self.views)

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)

    def select_cameras(self, camera_ids: tuple[str, ...]) -> MethodInput:
        if (
            len(set(camera_ids)) != len(camera_ids)
            or len(camera_ids) < 2
            or any(camera not in self.camera_ids for camera in camera_ids)
        ):
            raise ValueError("Select at least two distinct available camera IDs")
        return MethodInput(
            self.clip_id,
            self.frame_indices,
            self.fps,
            self.source_size_wh,
            tuple(self.views[self.camera_ids.index(camera)] for camera in camera_ids),
        )


@dataclass(frozen=True, slots=True)
class Ball3DResult:
    """Court metres on the full source timeline; used_views is (V,T).

    Optional cameras/offsets describe the selected views, in camera_ids order.
    A camera's frame t observes world time t/fps + offset_seconds. No offset
    or calibration is inferred for a held-out camera.
    """

    camera_ids: tuple[str, ...]
    frame_indices: NDArray[np.int64]
    positions: NDArray[np.float32]
    valid: NDArray[np.bool_]
    used_views: NDArray[np.bool_]
    covariance: NDArray[np.float64] | None = None
    cameras: tuple[PinholeCamera, ...] | None = None
    time_offsets_seconds: NDArray[np.float64] | None = None
    diagnostics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        t, v = len(self.frame_indices), len(self.camera_ids)
        if (
            t < 1
            or v < 2
            or len(set(self.camera_ids)) != v
            or self.frame_indices.dtype != np.int64
            or not np.array_equal(self.frame_indices, np.arange(t))
        ):
            raise ValueError(
                "Result requires unique camera IDs and the complete source frame axis"
            )
        if (
            self.positions.shape != (t, 3)
            or self.positions.dtype != np.float32
            or self.valid.shape != (t,)
            or self.valid.dtype != bool
            or self.used_views.shape != (v, t)
            or self.used_views.dtype != bool
        ):
            raise ValueError("Result array shape/dtype mismatch")
        if not np.isfinite(self.positions).all() or np.any(
            self.positions[~self.valid] != 0
        ):
            raise ValueError(
                "Invalid 3D positions must be zero, valid positions must be finite"
            )
        if np.any(self.used_views[:, ~self.valid]) or np.any(
            self.valid & (self.used_views.sum(0) < 2)
        ):
            raise ValueError(
                "A valid 3D point needs at least two used views; invalid points use none"
            )
        if self.covariance is not None:
            cov = self.covariance
            if (
                cov.shape != (t, 3, 3)
                or cov.dtype != np.float64
                or not np.isfinite(cov).all()
                or np.any(cov[~self.valid] != 0)
                or not np.allclose(cov, cov.transpose(0, 2, 1))
                or np.any(np.linalg.eigvalsh(cov) < -1e-9)
            ):
                raise ValueError(
                    "Covariance must be symmetric PSD and zero for invalid points"
                )
        if (
            self.cameras is not None
            and tuple(c.camera_id for c in self.cameras) != self.camera_ids
        ):
            raise ValueError("Result camera axis mismatch")
        if self.time_offsets_seconds is not None:
            offsets = self.time_offsets_seconds
            if (
                offsets.shape != (v,)
                or offsets.dtype != np.float64
                or not np.isfinite(offsets).all()
            ):
                raise ValueError(
                    "Result offsets must be finite seconds in camera order"
                )
        readonly_arrays(self)


class BallMethod(Protocol):
    """Implementations get only the selected fixed views, never scoring labels."""

    @property
    def name(self) -> str: ...

    @property
    def config(self) -> Mapping[str, Any]: ...

    def __call__(self, inputs: MethodInput) -> Ball3DResult: ...

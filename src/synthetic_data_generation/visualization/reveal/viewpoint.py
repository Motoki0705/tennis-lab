"""Reveal cameras: court-frame look-at -> canonical NHT scene camera, and its path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.rendering.nht import (
    NHTAppearanceControls,
    NHTRenderCamera,
)
from src.synthetic_data_generation.scene_contract import (
    MultiCourtLayout,
    RigidTransform,
)
from src.synthetic_data_generation.visualization.reveal.contracts import (
    RevealCameraPath,
    RevealKeyframe,
    RevealOutput,
    RevealViewpoint,
)

# Parameter samples per spline segment for the arc-length table.
_ARC_LENGTH_SAMPLES_PER_SEGMENT = 512


@dataclass(frozen=True, slots=True)
class RevealCamera:
    """One OpenCV pinhole camera in canonical NHT scene coordinates."""

    width: int
    height: int
    intrinsics: NDArray[np.float64]
    camera_to_scene: NDArray[np.float64]

    def project(
        self, points_scene: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return pixel coordinates ``(N, 2)`` and camera-Z depth ``(N,)``."""
        rotation = self.camera_to_scene[:3, :3]
        centre = self.camera_to_scene[:3, 3]
        camera = (np.asarray(points_scene, dtype=np.float64) - centre) @ rotation
        depth = camera[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = camera[:, :2] / depth[:, None]
        pixels = normalized * np.diag(self.intrinsics)[:2] + self.intrinsics[:2, 2]
        return pixels, depth

    def nht_render_camera(
        self, camera_id: str, appearance: NHTAppearanceControls
    ) -> NHTRenderCamera:
        """Return the public ``nht-render`` request camera for one state."""
        return NHTRenderCamera(
            camera_id=camera_id,
            width=self.width,
            height=self.height,
            intrinsics=tuple(float(value) for value in self.intrinsics.ravel()),
            camera_to_scene=RigidTransform.from_matrix(self.camera_to_scene),
            appearance=appearance,
        )


def look_at_camera_to_frame(
    eye: NDArray[np.float64], target: NDArray[np.float64], up: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return an OpenCV (+x right, +y down, +z forward) camera-to-frame pose."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def build_reveal_camera(
    viewpoint: RevealViewpoint,
    layout: MultiCourtLayout,
    adapter: MetricSceneAdapter,
    output: RevealOutput,
) -> RevealCamera:
    """Place the court-frame look-at camera in canonical NHT scene space.

    ``court -> metric scene`` is the accepted court transform and
    ``metric scene -> NHT scene`` is the alignment's uniform similarity, whose
    scale is removed from the rotation so the result stays a proper SE(3) pose.
    """
    courts = {court.court_instance_id: court for court in layout.courts}
    if viewpoint.anchor_court_id not in courts:
        raise ValueError(
            f"Unknown anchor court {viewpoint.anchor_court_id!r}; "
            f"accepted courts are {sorted(courts)}."
        )
    camera_to_court = look_at_camera_to_frame(
        np.asarray(viewpoint.eye_court_m, dtype=np.float64),
        np.asarray(viewpoint.target_court_m, dtype=np.float64),
        np.asarray(viewpoint.up_court, dtype=np.float64),
    )
    camera_to_metric = (
        courts[viewpoint.anchor_court_id].scene_from_court.matrix() @ camera_to_court
    )
    similarity = np.asarray(adapter.nht_scene_from_metric_scene, dtype=np.float64).reshape(
        4, 4
    )
    scale = float(adapter.nht_scene_units_per_metre)
    camera_to_scene = np.eye(4)
    camera_to_scene[:3, :3] = (similarity[:3, :3] / scale) @ camera_to_metric[:3, :3]
    camera_to_scene[:3, 3] = similarity[:3, :3] @ camera_to_metric[:3, 3] + similarity[:3, 3]
    # Validate the proper rigid pose through the shared scene contract.
    RigidTransform.from_matrix(camera_to_scene)

    focal = (output.width / 2.0) / math.tan(math.radians(viewpoint.horizontal_fov_deg) / 2.0)
    intrinsics = np.asarray(
        (
            (focal, 0.0, (output.width - 1) / 2.0),
            (0.0, focal, (output.height - 1) / 2.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return RevealCamera(
        width=output.width,
        height=output.height,
        intrinsics=intrinsics,
        camera_to_scene=camera_to_scene,
    )


def _catmull_rom(
    controls: NDArray[np.float64], segment: int, t: float
) -> NDArray[np.float64]:
    """Evaluate the uniform Catmull-Rom segment ``segment`` at local ``t``.

    ``controls`` carries one reflected phantom point at each end, so segment
    ``i`` runs from ``controls[i + 1]`` to ``controls[i + 2]``.
    """
    p0, p1, p2, p3 = controls[segment : segment + 4]
    result: NDArray[np.float64] = 0.5 * (
        2.0 * p1
        + (p2 - p0) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t * t * t
    )
    return result


class RevealCameraTrajectory:
    """Constant-speed camera move from the reveal viewpoint through waypoints.

    Eye and target follow one uniform Catmull-Rom spline (C1, through every
    waypoint). ``progress`` is the fraction of the path's arc length, measured
    as eye plus target travel in court metres, so the move has no speed jumps
    between waypoints of different spacing.
    """

    def __init__(
        self,
        viewpoint: RevealViewpoint,
        path: RevealCameraPath,
        layout: MultiCourtLayout,
        adapter: MetricSceneAdapter,
        output: RevealOutput,
    ) -> None:
        self._viewpoint = viewpoint
        self._layout = layout
        self._adapter = adapter
        self._output = output
        waypoints = np.asarray(
            [
                (*viewpoint.eye_court_m, *viewpoint.target_court_m),
                *((*item.eye_court_m, *item.target_court_m) for item in path.keyframes),
            ],
            dtype=np.float64,
        )
        self._segments = len(waypoints) - 1
        self._controls = np.concatenate(
            (
                2.0 * waypoints[:1] - waypoints[1:2],
                waypoints,
                2.0 * waypoints[-1:] - waypoints[-2:-1],
            )
        )
        samples = _ARC_LENGTH_SAMPLES_PER_SEGMENT
        self._parameters = np.linspace(0.0, float(self._segments), self._segments * samples + 1)
        points = np.stack([self._point(value) for value in self._parameters])
        steps = np.diff(points, axis=0)
        travel = np.linalg.norm(steps[:, :3], axis=1) + np.linalg.norm(steps[:, 3:], axis=1)
        self._arc_length = np.concatenate(([0.0], np.cumsum(travel)))
        if self._arc_length[-1] <= 1.0e-9:
            raise ValueError("camera_path keyframes must move the camera.")
        self._cache: dict[float, RevealCamera] = {}

    @property
    def length_m(self) -> float:
        """Return the eye-plus-target travel of the complete move, in metres."""
        return float(self._arc_length[-1])

    def _point(self, parameter: float) -> NDArray[np.float64]:
        segment = min(int(parameter), self._segments - 1)
        return _catmull_rom(self._controls, segment, parameter - segment)

    def viewpoint_at(self, progress: float) -> RevealViewpoint:
        """Return the look-at viewpoint at ``progress`` in ``[0, 1]`` of the path."""
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must lie in [0, 1].")
        if progress == 0.0:
            return self._viewpoint
        parameter = float(
            np.interp(progress * self._arc_length[-1], self._arc_length, self._parameters)
        )
        point = self._point(parameter)
        return self._viewpoint.moved_to(
            RevealKeyframe(
                eye_court_m=(float(point[0]), float(point[1]), float(point[2])),
                target_court_m=(float(point[3]), float(point[4]), float(point[5])),
            )
        )

    def camera_at(self, progress: float) -> RevealCamera:
        """Return the NHT scene camera at ``progress`` (cached per value)."""
        if progress not in self._cache:
            self._cache[progress] = build_reveal_camera(
                self.viewpoint_at(progress), self._layout, self._adapter, self._output
            )
        return self._cache[progress]

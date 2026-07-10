"""Explicit Matplotlib 3D camera views backed by physical camera conventions.

Matplotlib's 3D axes are not a physical-camera rasterizer.  This adapter maps a
world-space look-at camera (or saved pinhole-camera extrinsics) to Matplotlib's
``elev``, ``azim`` and ``roll`` angles.  Perspective field of view is only an
approximation because Matplotlib frames the current axes box rather than an
image plane; exact projection must continue to use
``src.utils.projection.camera_projector``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from src.utils.projection.camera_projector import Camera, make_look_at_camera

ViewMode = Literal["default", "look_at", "scene_camera"]
ProjectionMode = Literal["perspective", "orthographic"]

_DEFAULT_ELEV_DEG = 30.0
_DEFAULT_AZIM_DEG = -60.0
_DEFAULT_ROLL_DEG = 0.0


class _ConfigMapping(Protocol):
    def get(self, key: str, default: Any = ...) -> Any: ...


class _Axes3D(Protocol):
    def view_init(
        self,
        elev: float | None = None,
        azim: float | None = None,
        roll: float | None = None,
    ) -> None: ...

    def set_proj_type(
        self, proj_type: str, focal_length: float | None = None
    ) -> None: ...

    def get_box_aspect(self) -> Any: ...

    def set_box_aspect(self, aspect: Any, *, zoom: float = 1.0) -> None: ...


@dataclass(frozen=True)
class ResolvedCameraView3D:
    """Matplotlib camera parameters after resolving a configured view source."""

    elev_deg: float
    azim_deg: float
    roll_deg: float
    projection: ProjectionMode
    focal_length: float | None
    zoom: float


@dataclass(frozen=True)
class CameraView3DConfig:
    """Deterministic 3D visualization camera configuration.

    ``default`` explicitly reproduces Matplotlib's historical camera.  In
    ``look_at`` mode, ``center`` and ``look_at`` are metres in the court world
    coordinate system (XY ground plane, +Z up).  ``scene_camera`` reads the
    selected saved camera's ``params`` mapping.
    """

    mode: ViewMode = "default"
    center: tuple[float, float, float] = (0.0, -31.885, 7.0)
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.5)
    roll_deg: float = 0.0
    projection: ProjectionMode = "perspective"
    hfov_deg: float = 35.0
    scene_camera_index: int = 0
    zoom: float = 1.3

    @classmethod
    def from_mapping(cls, cfg: _ConfigMapping | None) -> CameraView3DConfig:
        """Parse a dict/OmegaConf mapping and reject ambiguous camera values."""
        if cfg is None:
            return cls()
        if not hasattr(cfg, "get"):
            raise TypeError(
                "view_3d must be a mapping-like config with a .get() method, "
                f"got {type(cfg).__name__}."
            )

        defaults = cls()
        mode = str(cfg.get("mode", defaults.mode))
        if mode not in {"default", "look_at", "scene_camera"}:
            raise ValueError(
                "view_3d.mode must be 'default', 'look_at', or 'scene_camera', "
                f"got {mode!r}."
            )
        projection = str(cfg.get("projection", defaults.projection))
        if projection not in {"perspective", "orthographic"}:
            raise ValueError(
                "view_3d.projection must be 'perspective' or 'orthographic', "
                f"got {projection!r}."
            )

        center = _float_triplet(cfg.get("center", defaults.center), "view_3d.center")
        look_at = _float_triplet(
            cfg.get("look_at", defaults.look_at), "view_3d.look_at"
        )
        if np.allclose(center, look_at, rtol=0.0, atol=1e-9):
            raise ValueError("view_3d.center and view_3d.look_at must be different.")

        hfov_deg = float(cfg.get("hfov_deg", defaults.hfov_deg))
        if not math.isfinite(hfov_deg) or not 0.0 < hfov_deg < 180.0:
            raise ValueError(
                f"view_3d.hfov_deg must be finite and in (0, 180), got {hfov_deg}."
            )
        roll_deg = float(cfg.get("roll_deg", defaults.roll_deg))
        if not math.isfinite(roll_deg):
            raise ValueError(f"view_3d.roll_deg must be finite, got {roll_deg}.")
        scene_camera_index = int(
            cfg.get("scene_camera_index", defaults.scene_camera_index)
        )
        if scene_camera_index < 0:
            raise ValueError(
                "view_3d.scene_camera_index must be non-negative, "
                f"got {scene_camera_index}."
            )
        zoom = float(cfg.get("zoom", defaults.zoom))
        if not math.isfinite(zoom) or zoom <= 0.0:
            raise ValueError(f"view_3d.zoom must be positive and finite, got {zoom}.")

        return cls(
            mode=cast(ViewMode, mode),
            center=center,
            look_at=look_at,
            roll_deg=roll_deg,
            projection=cast(ProjectionMode, projection),
            hfov_deg=hfov_deg,
            scene_camera_index=scene_camera_index,
            zoom=zoom,
        )

    def resolve(
        self,
        scene_cameras: Sequence[Any] | None = None,
    ) -> ResolvedCameraView3D:
        """Resolve this config to concrete Matplotlib camera parameters."""
        if self.mode == "default":
            return ResolvedCameraView3D(
                elev_deg=_DEFAULT_ELEV_DEG,
                azim_deg=_DEFAULT_AZIM_DEG,
                roll_deg=_DEFAULT_ROLL_DEG,
                projection=self.projection,
                focal_length=(
                    _focal_length_from_hfov(90.0)
                    if self.projection == "perspective"
                    else None
                ),
                zoom=self.zoom,
            )

        if self.mode == "look_at":
            camera = make_look_at_camera(
                self.center,
                self.look_at,
                hfov_deg=self.hfov_deg,
            )
            elev, azim, _ = camera_to_matplotlib_angles(camera)
            return ResolvedCameraView3D(
                elev_deg=elev,
                azim_deg=azim,
                roll_deg=self.roll_deg,
                projection=self.projection,
                focal_length=(
                    _focal_length_from_hfov(self.hfov_deg)
                    if self.projection == "perspective"
                    else None
                ),
                zoom=self.zoom,
            )

        camera = camera_from_scene_cameras(scene_cameras, self.scene_camera_index)
        elev, azim, camera_roll = camera_to_matplotlib_angles(camera)
        return ResolvedCameraView3D(
            elev_deg=elev,
            azim_deg=azim,
            roll_deg=camera_roll + self.roll_deg,
            projection=self.projection,
            focal_length=(
                _focal_length_from_camera(camera)
                if self.projection == "perspective"
                else None
            ),
            zoom=self.zoom,
        )

    def apply(
        self,
        ax: _Axes3D,
        scene_cameras: Sequence[Any] | None = None,
    ) -> ResolvedCameraView3D:
        """Apply the resolved view to an axes and return it for inspection."""
        view = self.resolve(scene_cameras)
        ax.view_init(elev=view.elev_deg, azim=view.azim_deg, roll=view.roll_deg)
        if view.projection == "orthographic":
            ax.set_proj_type("ortho")
        else:
            if view.focal_length is None:
                raise RuntimeError("Perspective view requires a focal length.")
            ax.set_proj_type("persp", focal_length=view.focal_length)
        ax.set_box_aspect(ax.get_box_aspect(), zoom=view.zoom)
        return view


def camera_to_matplotlib_angles(camera: Camera) -> tuple[float, float, float]:
    """Convert OpenCV world-to-camera axes to Matplotlib view angles."""
    center = _numpy_triplet(camera.C, "camera.C")
    rotation = np.asarray(camera.R, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"camera.R must be a finite 3x3 matrix, got {rotation.shape}.")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4, rtol=1e-4):
        raise ValueError("camera.R must be an orthonormal world-to-camera rotation.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-4):
        raise ValueError("camera.R must be a proper rotation with determinant +1.")

    forward = rotation[2]
    camera_direction = -forward
    elev = math.degrees(
        math.atan2(
            float(camera_direction[2]),
            math.hypot(float(camera_direction[0]), float(camera_direction[1])),
        )
    )
    azim = math.degrees(
        math.atan2(float(camera_direction[1]), float(camera_direction[0]))
    )

    target = center + forward
    zero_roll = make_look_at_camera(center, target)
    base_right = np.asarray(zero_roll.R[0], dtype=np.float64)
    actual_right = rotation[0]
    roll = math.degrees(
        math.atan2(
            float(np.dot(np.cross(base_right, actual_right), forward)),
            float(np.dot(base_right, actual_right)),
        )
    )
    return elev, azim, roll


def camera_from_scene_cameras(
    scene_cameras: Sequence[Any] | None,
    camera_index: int,
) -> Camera:
    """Build a physical camera from a saved scene camera record."""
    if scene_cameras is None:
        raise ValueError(
            "view_3d.mode='scene_camera' requires scene camera metadata, but none "
            "was provided by this renderer."
        )
    if camera_index >= len(scene_cameras):
        raise ValueError(
            f"view_3d.scene_camera_index={camera_index} is out of range for "
            f"{len(scene_cameras)} scene cameras."
        )

    record = scene_cameras[camera_index]
    if isinstance(record, Camera):
        return record
    params: Any
    if isinstance(record, Mapping):
        params = record.get("params", record)
    else:
        params = getattr(record, "params", None)
    if not isinstance(params, Mapping):
        raise ValueError(
            f"Scene camera {camera_index} must expose a 'params' mapping with "
            "C, R, f, cx, cy, w, and h."
        )

    required = ("C", "R", "f", "cx", "cy", "w", "h")
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(
            f"Scene camera {camera_index} params are missing required fields: "
            f"{', '.join(missing)}."
        )

    import torch

    camera = Camera(
        C=torch.as_tensor(params["C"], dtype=torch.float32),
        R=torch.as_tensor(params["R"], dtype=torch.float32),
        f=float(params["f"]),
        cx=float(params["cx"]),
        cy=float(params["cy"]),
        w=int(params["w"]),
        h=int(params["h"]),
    )
    if not math.isfinite(camera.f) or camera.f <= 0.0:
        raise ValueError(f"Scene camera {camera_index} f must be positive and finite.")
    if camera.w <= 0 or camera.h <= 0:
        raise ValueError(f"Scene camera {camera_index} w and h must be positive.")
    return camera


def scene_cameras_from_scene(scene: Any) -> Sequence[Any] | None:
    """Return saved camera records from task scenes or integrated metadata."""
    if isinstance(scene, Mapping):
        cameras = scene.get("cameras")
        if cameras is not None:
            return _scene_camera_sequence(cameras)
    else:
        cameras = getattr(scene, "cameras", None)
        if cameras is not None:
            return _scene_camera_sequence(cameras)
        metadata = getattr(scene, "metadata", None)
        if isinstance(metadata, Mapping):
            cameras = metadata.get("cameras")
            if cameras is not None:
                return _scene_camera_sequence(cameras)
    return None


def _scene_camera_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            "Scene camera metadata must be a sequence of camera records, "
            f"got {type(value).__name__}."
        )
    return cast("Sequence[Any]", value)


def _float_triplet(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a length-3 sequence.")
    try:
        values = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a length-3 numeric sequence.") from exc
    if len(values) != 3 or not all(math.isfinite(component) for component in values):
        raise ValueError(f"{name} must contain exactly three finite values.")
    return values


def _numpy_triplet(value: Any, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite length-3 vector, got {array.shape}.")
    return cast("NDArray[np.float64]", array)


def _focal_length_from_hfov(hfov_deg: float) -> float:
    return 1.0 / math.tan(math.radians(hfov_deg) / 2.0)


def _focal_length_from_camera(camera: Camera) -> float:
    return float(camera.f) / (0.5 * float(camera.w))


__all__ = [
    "CameraView3DConfig",
    "ResolvedCameraView3D",
    "camera_from_scene_cameras",
    "camera_to_matplotlib_angles",
    "scene_cameras_from_scene",
]

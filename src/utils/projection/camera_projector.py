"""Camera projection utilities shared across dataset generators."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from src.utils.schema.court import (
    BASELINE_CLEAR,
    FENCE_HEIGHT,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    SIDELINE_CLEAR,
    CourtConfig,
    court_keypoints_3d,
)


@dataclass(frozen=True)
class Camera:
    """Simple pinhole camera defined by extrinsics and intrinsics."""

    C: Tensor
    R: Tensor
    f: float
    cx: float
    cy: float
    w: int
    h: int


def make_look_at_camera(
    center: Iterable[float],
    look_at: Iterable[float] = (0.0, 0.0, 0.5),
    image_size: tuple[int, int] = (1280, 720),
    hfov_deg: float = 60.0,
) -> Camera:
    """Create a pinhole camera pointed at ``look_at``."""
    center_t = torch.tensor(center, dtype=torch.float32)
    look_t = torch.tensor(look_at, dtype=torch.float32)

    z_cam = look_t - center_t
    z_cam = z_cam / (z_cam.norm() + 1e-8)

    up_world = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    x_cam = torch.cross(up_world, z_cam, dim=0)
    x_norm = x_cam.norm()
    if x_norm < 1e-6:
        # Fallback when looking straight up/down
        up_world = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        x_cam = torch.cross(up_world, z_cam, dim=0)
        x_norm = x_cam.norm()
    x_cam = x_cam / (x_norm + 1e-8)
    y_cam = torch.cross(z_cam, x_cam, dim=0)

    R = torch.stack([x_cam, y_cam, z_cam], dim=0)  # world -> camera

    w, h = image_size
    hfov_rad = math.radians(float(hfov_deg))
    f = 0.5 * float(w) / math.tan(0.5 * hfov_rad)
    cx = 0.5 * float(w)
    cy = 0.5 * float(h)

    return Camera(C=center_t, R=R, f=float(f), cx=cx, cy=cy, w=w, h=h)


def project_points(cam: Camera, xyz: Tensor) -> tuple[Tensor, Tensor]:
    """Project world coordinates into the camera's image plane."""
    if xyz.numel() == 0:
        return xyz.new_zeros((0, 2)), xyz.new_zeros((0,), dtype=torch.bool)

    # world -> camera
    X = xyz - cam.C.view(1, 3)
    Xc = X @ cam.R.t()

    z = Xc[:, 2]
    mask = z > 1e-6
    z_safe = torch.where(mask, z, torch.ones_like(z))

    # Flip v direction so that up = top of screen
    u = cam.f * (Xc[:, 0] / z_safe) + cam.cx
    v = cam.f * (-Xc[:, 1] / z_safe) + cam.cy

    uv = torch.stack([u, v], dim=-1)
    return uv, mask


@dataclass
class CameraConfig:
    """Configuration for camera generation.

    Supports per-camera perturbation of intrinsics and look-at direction
    for dataset variation.
    """

    placement_mode: str = "random"  # "random" | "fixed_8"
    z_min: float = 3.0
    z_max: float = 5.0
    hfov_deg: float = 60.0
    image_size: tuple[int, int] = (1280, 720)
    fixed_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fixed_baseline_clear_extra: float = 0.0
    fixed_position_noise_radius: float = 0.0
    fixed_look_at_xy_radius: float = 0.0
    target_x_range: tuple[float, float] = (-2.0, 2.0)
    target_y_range: tuple[float, float] = (-2.0, 2.0)
    target_z_range: tuple[float, float] = (0.5, 1.5)

    # --- Perturbation parameters ---
    # Gaussian noise on hfov (degrees, sigma)
    hfov_noise_deg: float = 0.0
    # Gaussian noise on look-at target (metres, sigma)
    look_at_noise_std: float = 0.0


@dataclass
class CameraView:
    """Generic camera view container."""

    camera: Camera
    camera_params: dict
    court_kp_uv: Tensor
    court_kp_visible: Tensor
    points_uv: Tensor | None
    points_visible: Tensor | None


class CameraProjector:
    """Generates cameras and projects 3D points to 2D UV coordinates."""

    def __init__(
        self,
        config: CameraConfig | None = None,
        court_config: CourtConfig | None = None,
    ) -> None:
        self.config = config or CameraConfig()
        if self.config.placement_mode not in {"random", "fixed_8"}:
            raise ValueError(
                "camera placement_mode must be one of {'random', 'fixed_8'}, "
                f"got '{self.config.placement_mode}'"
            )
        self.court_config = court_config
        self.court_kp_3d = court_keypoints_3d(court_config)

    @staticmethod
    def _fence_extents(baseline_clear_extra: float = 0.0) -> tuple[float, float]:
        """Return outer fence extents in x/y from court center."""
        fence_x = HALF_DOUBLES_WIDTH + SIDELINE_CLEAR
        fence_y = HALF_LENGTH + BASELINE_CLEAR + max(0.0, baseline_clear_extra)
        return fence_x, fence_y

    def sample_camera(self) -> Camera:
        """Sample a camera position around the court."""
        cfg = self.config

        fence_x, fence_y = self._fence_extents()

        perimeter = 2 * (2 * fence_x + 2 * fence_y)
        t = random.uniform(0, perimeter)

        if t < 2 * fence_x:
            cam_x = -fence_x + t
            cam_y = fence_y
        elif t < 2 * fence_x + 2 * fence_y:
            cam_x = fence_x
            cam_y = fence_y - (t - 2 * fence_x)
        elif t < 4 * fence_x + 2 * fence_y:
            cam_x = fence_x - (t - 2 * fence_x - 2 * fence_y)
            cam_y = -fence_y
        else:
            cam_x = -fence_x
            cam_y = -fence_y + (t - 4 * fence_x - 2 * fence_y)

        cam_z = random.uniform(cfg.z_min, cfg.z_max)
        cam_z = max(cam_z, FENCE_HEIGHT)

        target_x = random.uniform(*cfg.target_x_range)
        target_y = random.uniform(*cfg.target_y_range)
        target_z = random.uniform(*cfg.target_z_range)

        # Apply look-at perturbation
        if cfg.look_at_noise_std > 0:
            target_x += random.gauss(0, cfg.look_at_noise_std)
            target_y += random.gauss(0, cfg.look_at_noise_std)
            target_z += random.gauss(0, cfg.look_at_noise_std * 0.5)

        # Apply hfov perturbation
        hfov = cfg.hfov_deg
        if cfg.hfov_noise_deg > 0:
            hfov += random.gauss(0, cfg.hfov_noise_deg)
            hfov = max(20.0, min(120.0, hfov))  # clamp to sane range

        return make_look_at_camera(
            center=(cam_x, cam_y, cam_z),
            look_at=(target_x, target_y, target_z),
            hfov_deg=hfov,
            image_size=cfg.image_size,
        )

    @staticmethod
    def _sample_uniform_offset_in_ball(radius: float) -> tuple[float, float, float]:
        """Sample a volume-uniform 3D offset inside a ball."""
        if radius <= 0.0:
            return 0.0, 0.0, 0.0

        radius_sq = radius * radius
        while True:
            dx = random.uniform(-radius, radius)
            dy = random.uniform(-radius, radius)
            dz = random.uniform(-radius, radius)
            if dx * dx + dy * dy + dz * dz <= radius_sq:
                return dx, dy, dz

    @staticmethod
    def _sample_uniform_xy_in_disk(radius: float) -> tuple[float, float]:
        """Sample an area-uniform XY location inside a disk."""
        if radius <= 0.0:
            return 0.0, 0.0

        theta = random.uniform(0.0, 2.0 * math.pi)
        r = radius * math.sqrt(random.random())
        return r * math.cos(theta), r * math.sin(theta)

    def fixed_cameras(self) -> list[Camera]:
        """Build the fixed 8-camera layout (4 corners + 4 edge midpoints).

        Corner cameras use ``z_min``. Midpoint cameras use ``z_max``.
        When both fixed-camera noise radii are zero, the legacy layout is
        preserved exactly.
        """
        cfg = self.config
        fence_x, fence_y = self._fence_extents(cfg.fixed_baseline_clear_extra)
        look_at = cfg.fixed_look_at

        corners = [
            (-fence_x, +fence_y, cfg.z_min),
            (+fence_x, +fence_y, cfg.z_min),
            (+fence_x, -fence_y, cfg.z_min),
            (-fence_x, -fence_y, cfg.z_min),
        ]
        midpoints = [
            (0.0, +fence_y, cfg.z_max),
            (+fence_x, 0.0, cfg.z_max),
            (0.0, -fence_y, cfg.z_max),
            (-fence_x, 0.0, cfg.z_max),
        ]

        cams: list[Camera] = []
        for base_center in corners + midpoints:
            dx, dy, dz = self._sample_uniform_offset_in_ball(
                cfg.fixed_position_noise_radius
            )
            center = (
                base_center[0] + dx,
                base_center[1] + dy,
                base_center[2] + dz,
            )

            if cfg.fixed_look_at_xy_radius > 0.0:
                target_x, target_y = self._sample_uniform_xy_in_disk(
                    cfg.fixed_look_at_xy_radius
                )
                look_at_target = (target_x, target_y, 0.0)
            else:
                look_at_target = look_at

            cams.append(
                make_look_at_camera(
                    center=center,
                    look_at=look_at_target,
                    hfov_deg=cfg.hfov_deg,
                    image_size=cfg.image_size,
                )
            )
        return cams

    def project_points_to_uv(
        self,
        points_3d: Tensor,
        camera: Camera,
    ) -> tuple[Tensor, Tensor]:
        """Project 3D points to normalized UV coordinates."""
        if points_3d.numel() == 0:
            uv = points_3d.new_zeros((*points_3d.shape[:-1], 2))
            visible = points_3d.new_zeros(points_3d.shape[:-1], dtype=torch.bool)
            return uv, visible

        original_shape = points_3d.shape[:-1]
        flat = points_3d.reshape(-1, 3)

        uv, in_front = project_points(camera, flat)
        uv[:, 0] /= float(camera.w)
        uv[:, 1] /= float(camera.h)

        in_bounds = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] <= 1)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] <= 1)
        )
        visible = in_bounds & in_front

        uv = uv.reshape(*original_shape, 2)
        visible = visible.reshape(*original_shape)
        return uv, visible

    def project_court_keypoints(self, camera: Camera) -> tuple[Tensor, Tensor]:
        """Project CourtKP20 to UV coordinates."""
        return self.project_points_to_uv(self.court_kp_3d, camera)

    def project_subject_points(
        self,
        points_3d: Tensor,
        camera: Camera,
    ) -> tuple[Tensor, Tensor]:
        """Project subject points. Override for custom behavior."""
        return self.project_points_to_uv(points_3d, camera)

    def generate_camera_view(
        self,
        points_3d: Tensor | None = None,
        camera: Camera | None = None,
    ) -> CameraView:
        """Generate a camera view with optional subject projection."""
        if camera is None:
            camera = self.sample_camera()

        court_kp_uv, court_kp_visible = self.project_court_keypoints(camera)

        points_uv = None
        points_visible = None
        if points_3d is not None:
            points_uv, points_visible = self.project_subject_points(points_3d, camera)

        camera_params = {
            "C": camera.C.tolist(),
            "R": camera.R.tolist(),
            "f": camera.f,
            "cx": camera.cx,
            "cy": camera.cy,
            "w": camera.w,
            "h": camera.h,
        }

        return CameraView(
            camera=camera,
            camera_params=camera_params,
            court_kp_uv=court_kp_uv,
            court_kp_visible=court_kp_visible,
            points_uv=points_uv,
            points_visible=points_visible,
        )

    def generate_multiple_views(
        self,
        points_3d: Tensor | None,
        num_cameras: int,
    ) -> list[CameraView]:
        """Generate multiple camera views for a subject."""
        views = []
        for _ in range(num_cameras):
            views.append(self.generate_camera_view(points_3d))
        return views

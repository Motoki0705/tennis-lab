"""Camera projection for BLCS (blcs.md §7 compliant).

Provides camera generation and 3D to 2D projection for:
- Court keypoints (CourtKP20)
- Ball trajectories
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from src.utils.geometry import (
    BASELINE_CLEAR,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    SIDELINE_CLEAR,
    Camera,
    court_keypoints_3d,
    make_look_at_camera,
    project_points,
)

if TYPE_CHECKING:
    pass


@dataclass
class CameraConfig:
    """Configuration for camera generation."""

    # Camera height range
    z_min: float = 3.0
    z_max: float = 5.0

    # Horizontal field of view (degrees)
    hfov_deg: float = 60.0

    # Image size
    image_size: tuple[int, int] = (1280, 720)

    # Look-at target offset ranges
    target_x_range: tuple[float, float] = (-2.0, 2.0)
    target_y_range: tuple[float, float] = (-2.0, 2.0)
    target_z_range: tuple[float, float] = (0.5, 1.5)


@dataclass
class CameraView:
    """Data for a single camera view."""

    # Camera parameters
    camera: Camera
    camera_params: dict  # Serializable camera parameters

    # Court keypoints (fixed for scene)
    court_kp_uv: Tensor  # [20, 2] UV coordinates
    court_kp_visible: Tensor  # [20] visibility flags

    # Ball trajectory
    ball_uv: Tensor  # [T, 2] UV coordinates
    ball_visible: Tensor  # [T] visibility flags


class CameraProjector:
    """Generates cameras and projects 3D points to 2D UV coordinates.

    Camera placement follows PLCS conventions:
    - Cameras placed on fence perimeter
    - Random height within configured range
    - Looking at court center with random offset
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        """Initialize camera projector.

        Args:
            config: Camera configuration.

        """
        self.config = config or CameraConfig()

        # Get court keypoints
        self.court_kp_3d = court_keypoints_3d()  # [20, 3]

    def sample_camera(self) -> Camera:
        """Sample a camera position around the court.

        Camera is placed on the fence perimeter rectangle.

        Returns:
            Camera: Camera instance.

        """
        cfg = self.config

        # Fence perimeter
        fence_x = HALF_DOUBLES_WIDTH + SIDELINE_CLEAR
        fence_y = HALF_LENGTH + BASELINE_CLEAR

        # Sample position on fence rectangle perimeter
        perimeter = 2 * (2 * fence_x + 2 * fence_y)
        t = random.uniform(0, perimeter)

        if t < 2 * fence_x:
            # Top side (far baseline)
            cam_x = -fence_x + t
            cam_y = fence_y
        elif t < 2 * fence_x + 2 * fence_y:
            # Right side
            cam_x = fence_x
            cam_y = fence_y - (t - 2 * fence_x)
        elif t < 4 * fence_x + 2 * fence_y:
            # Bottom side (near baseline)
            cam_x = fence_x - (t - 2 * fence_x - 2 * fence_y)
            cam_y = -fence_y
        else:
            # Left side
            cam_x = -fence_x
            cam_y = -fence_y + (t - 4 * fence_x - 2 * fence_y)

        # Random height
        cam_z = random.uniform(cfg.z_min, cfg.z_max)

        # Look-at target with random offset
        target_x = random.uniform(*cfg.target_x_range)
        target_y = random.uniform(*cfg.target_y_range)
        target_z = random.uniform(*cfg.target_z_range)

        return make_look_at_camera(
            center=(cam_x, cam_y, cam_z),
            look_at=(target_x, target_y, target_z),
            hfov_deg=cfg.hfov_deg,
            image_size=cfg.image_size,
        )

    def project_points_to_uv(
        self,
        points_3d: Tensor,
        camera: Camera,
    ) -> tuple[Tensor, Tensor]:
        """Project 3D points to UV coordinates.

        Args:
            points_3d: 3D points [N, 3] or [T, N, 3].
            camera: Camera instance.

        Returns:
            tuple: (uv [N, 2] or [T, N, 2], visible [N] or [T, N])

        """
        # Convert to numpy for projection
        if points_3d.dim() == 2:
            pts_np = points_3d.numpy()
            uv_np, visible_np = self._project_batch(pts_np, camera)
            uv = torch.from_numpy(uv_np).float()
            visible = torch.from_numpy(visible_np).float()
        else:
            # [T, N, 3]
            T = points_3d.shape[0]
            uv_list = []
            vis_list = []
            for t in range(T):
                pts_np = points_3d[t].numpy()
                uv_np, vis_np = self._project_batch(pts_np, camera)
                uv_list.append(torch.from_numpy(uv_np).float())
                vis_list.append(torch.from_numpy(vis_np).float())
            uv = torch.stack(uv_list, dim=0)
            visible = torch.stack(vis_list, dim=0)

        return uv, visible

    def _project_batch(
        self,
        points_3d: np.ndarray,
        camera: Camera,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project a batch of 3D points.

        Args:
            points_3d: [N, 3] numpy array.
            camera: Camera instance.

        Returns:
            tuple: (uv [N, 2], visible [N])

        """
        cfg = self.config
        w, h = cfg.image_size

        # Use plcs project_points (note: argument order is (camera, points))
        pts_tensor = torch.from_numpy(points_3d).float()
        uv_tensor, in_front_mask = project_points(camera, pts_tensor)
        uv_np = uv_tensor.numpy()

        # Normalize to [0, 1]
        uv_normalized = np.zeros_like(uv_np)
        uv_normalized[:, 0] = uv_np[:, 0] / w
        uv_normalized[:, 1] = uv_np[:, 1] / h

        # Visibility check
        # Point is visible if:
        # 1. In front of camera (from project_points)
        # 2. Within image bounds [0, 1]
        in_bounds = (
            (uv_normalized[:, 0] >= 0)
            & (uv_normalized[:, 0] <= 1)
            & (uv_normalized[:, 1] >= 0)
            & (uv_normalized[:, 1] <= 1)
        )

        # Use in_front_mask from project_points
        in_front = in_front_mask.numpy()
        visible = in_bounds & in_front

        return uv_normalized, visible.astype(np.float32)

    def project_court_keypoints(
        self,
        camera: Camera,
    ) -> tuple[Tensor, Tensor]:
        """Project CourtKP20 to UV coordinates.

        Args:
            camera: Camera instance.

        Returns:
            tuple: (court_kp_uv [20, 2], court_kp_visible [20])

        """
        return self.project_points_to_uv(self.court_kp_3d, camera)

    def project_trajectory(
        self,
        trajectory: Tensor,
        camera: Camera,
    ) -> tuple[Tensor, Tensor]:
        """Project ball trajectory to UV coordinates.

        Args:
            trajectory: Ball trajectory [T, 3].
            camera: Camera instance.

        Returns:
            tuple: (ball_uv [T, 2], ball_visible [T])

        """
        # Expand to [T, 1, 3] for consistent handling
        traj_expanded = trajectory.unsqueeze(1)  # [T, 1, 3]
        uv, visible = self.project_points_to_uv(traj_expanded, camera)

        # Squeeze back to [T, 2] and [T]
        return uv.squeeze(1), visible.squeeze(1)

    def generate_camera_view(
        self,
        trajectory: Tensor,
        camera: Camera | None = None,
    ) -> CameraView:
        """Generate a complete camera view for a trajectory.

        Args:
            trajectory: Ball trajectory [T, 3].
            camera: Optional camera (samples new one if None).

        Returns:
            CameraView: Complete camera view data.

        """
        if camera is None:
            camera = self.sample_camera()

        # Project court keypoints
        court_kp_uv, court_kp_visible = self.project_court_keypoints(camera)

        # Project trajectory
        ball_uv, ball_visible = self.project_trajectory(trajectory, camera)

        # Extract camera parameters for serialization
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
            ball_uv=ball_uv,
            ball_visible=ball_visible,
        )

    def generate_multiple_views(
        self,
        trajectory: Tensor,
        num_cameras: int,
    ) -> list[CameraView]:
        """Generate multiple camera views for a trajectory.

        Args:
            trajectory: Ball trajectory [T, 3].
            num_cameras: Number of cameras to generate.

        Returns:
            list: List of CameraView instances.

        """
        views = []
        for _ in range(num_cameras):
            view = self.generate_camera_view(trajectory)
            views.append(view)
        return views

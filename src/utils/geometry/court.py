"""Court geometry helpers shared by the tennis simulator.

Standard tennis court dimensions according to ITF regulations.
All measurements are in meters.

Court coordinate system:
- Origin at center of court (net center)
- X-axis: sideline direction (positive = right when facing net)
- Y-axis: baseline direction (positive = far side)
- Z-axis: vertical (positive = up)
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

# -----------------------------
# ITF Standard Court Dimensions (meters)
# -----------------------------

COURT_LENGTH: float = 23.77
HALF_LENGTH: float = COURT_LENGTH / 2.0  # 11.885

SINGLES_WIDTH: float = 8.23
HALF_SINGLES_WIDTH: float = SINGLES_WIDTH / 2.0  # 4.115

DOUBLES_WIDTH: float = 10.97
HALF_DOUBLES_WIDTH: float = DOUBLES_WIDTH / 2.0  # 5.485

SERVICE_LINE_DISTANCE: float = 6.40  # Distance from net to service line
CENTER_MARK_LENGTH: float = 0.10  # Length of center mark on baseline

# Net dimensions
NET_HEIGHT_CENTER: float = 0.914  # Net height at center (3 feet)
NET_HEIGHT_POST: float = 1.07  # Net height at posts (3.5 feet)

# Net post offset from doubles sideline
NET_POST_OFFSET_X: float = 0.914

# -----------------------------
# Fence (Run-off) Dimensions
# -----------------------------

BASELINE_CLEAR: float = 6.40
SIDELINE_CLEAR: float = 3.66
FENCE_HEIGHT: float = 3.0

X_MIN: float = -(HALF_DOUBLES_WIDTH + SIDELINE_CLEAR)  # -9.145
X_MAX: float = +(HALF_DOUBLES_WIDTH + SIDELINE_CLEAR)  # +9.145
Y_MIN: float = -(HALF_LENGTH + BASELINE_CLEAR)  # -18.285
Y_MAX: float = +(HALF_LENGTH + BASELINE_CLEAR)  # +18.285


def court_keypoints_3d() -> Tensor:
    """Return 20 court keypoints (idx 0..19) as a (20, 3) tensor.

    Keypoint indices follow the CourtKP20 specification:

    0..3:  far/near doubles corners
    4..7:  far/near singles corners
    8..11: service line endpoints
    12,13: service T (far, near)
    14:    net center (ground)
    15..18: net posts (base/top, left/right)
    19:    center strap top
    """
    xs = HALF_SINGLES_WIDTH
    xd = HALF_DOUBLES_WIDTH
    yB = HALF_LENGTH
    yS = SERVICE_LINE_DISTANCE

    x_post_L = -(xd + NET_POST_OFFSET_X)
    x_post_R = +(xd + NET_POST_OFFSET_X)

    pts = [
        (-xd, +yB, 0.0),  # 0 far doubles corner left
        (+xd, +yB, 0.0),  # 1 far doubles corner right
        (-xd, -yB, 0.0),  # 2 near doubles corner left
        (+xd, -yB, 0.0),  # 3 near doubles corner right
        (-xs, +yB, 0.0),  # 4 far singles corner left
        (-xs, -yB, 0.0),  # 5 near singles corner left
        (+xs, +yB, 0.0),  # 6 far singles corner right
        (+xs, -yB, 0.0),  # 7 near singles corner right
        (-xs, +yS, 0.0),  # 8 far service-line endpoint left
        (+xs, +yS, 0.0),  # 9 far service-line endpoint right
        (-xs, -yS, 0.0),  # 10 near service-line endpoint left
        (+xs, -yS, 0.0),  # 11 near service-line endpoint right
        (0.0, +yS, 0.0),  # 12 far service T
        (0.0, -yS, 0.0),  # 13 near service T
        (0.0, 0.0, 0.0),  # 14 net center (ground)
        (x_post_L, 0.0, 0.0),  # 15 left net post base
        (x_post_L, 0.0, NET_HEIGHT_POST),  # 16 left net post top
        (x_post_R, 0.0, 0.0),  # 17 right net post base
        (x_post_R, 0.0, NET_HEIGHT_POST),  # 18 right net post top
        (0.0, 0.0, NET_HEIGHT_CENTER),  # 19 center strap top
    ]
    return torch.tensor(pts, dtype=torch.float32)


# -----------------------------
# Simple Pinhole Camera
# -----------------------------


@dataclass(frozen=True)
class Camera:
    """Simple pinhole camera defined by extrinsics and intrinsics.

    Attributes:
        C: Camera center in world coords, shape (3,)
        R: Rotation matrix (world -> camera), shape (3,3)
        f: Focal length in pixels
        cx, cy: Principal point in pixels
        w, h: Image size in pixels

    """

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
    """Create a pinhole camera pointed at ``look_at``.

    Args:
        center (Iterable[float]): Camera position (x, y, z) in world coordinates.
        look_at (Iterable[float]): Target the camera should look at.
        image_size (tuple[int, int]): Output image width/height in pixels.
        hfov_deg (float): Horizontal field of view in degrees.

    Returns:
        Camera: Dataclass containing extrinsics and intrinsics.

    """
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
    """Project world coordinates into the camera's image plane.

    Args:
        cam (Camera): Camera returned by :func:`make_look_at_camera`.
        xyz (Tensor): ``(N, 3)`` tensor of world coordinates.

    Returns:
        tuple[Tensor, Tensor]: Pixel coordinates shaped ``(N, 2)`` and a boolean
        mask indicating whether the projected point lies in front of the camera.

    """
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


# -----------------------------
# Camera Position Sampling on Fence
# -----------------------------


def sample_camera_position_on_fence(t: float, side: str) -> tuple[float, float, float]:
    r"""Sample a camera center along the perimeter fence.

    Args:
        t (float): Parameter in ``[0, 1]`` describing the relative position.
        side (str): One of ``{"near", "far", "left", "right"}``.

    Returns:
        tuple[float, float, float]: Position (x, y, z) near fence height.

    Raises:
        ValueError: If ``side`` is not supported.

    """
    t = float(t)
    t = max(0.0, min(1.0, t))

    if side == "near":
        x = X_MIN + t * (X_MAX - X_MIN)
        y = Y_MIN
    elif side == "far":
        x = X_MIN + t * (X_MAX - X_MIN)
        y = Y_MAX
    elif side == "left":
        x = X_MIN
        y = Y_MIN + t * (Y_MAX - Y_MIN)
    elif side == "right":
        x = X_MAX
        y = Y_MIN + t * (Y_MAX - Y_MIN)
    else:
        raise ValueError(f"unknown side: {side!r}")

    z = FENCE_HEIGHT
    return float(x), float(y), float(z)

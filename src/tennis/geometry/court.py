# src/tennis/geometry/court.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import Tensor

# -----------------------------
# ITF 規格ベースのコート寸法
# -----------------------------

COURT_LENGTH = 23.77
HALF_LENGTH = COURT_LENGTH / 2.0  # 11.885

SINGLES_WIDTH = 8.23
HALF_SINGLES_WIDTH = SINGLES_WIDTH / 2.0  # 4.115

DOUBLES_WIDTH = 10.97
HALF_DOUBLES_WIDTH = DOUBLES_WIDTH / 2.0  # 5.485

SERVICE_LINE_DISTANCE = 6.40

NET_HEIGHT_CENTER = 0.914
NET_HEIGHT_POST = 1.07

# doubles サイドラインから外側へのポストオフセット
NET_POST_OFFSET_X = 0.914

# -----------------------------
# フェンス（ランオフ）寸法
# -----------------------------

BASELINE_CLEAR = 6.40
SIDELINE_CLEAR = 3.66
FENCE_HEIGHT = 3.0

X_MIN = -(HALF_DOUBLES_WIDTH + SIDELINE_CLEAR)  # -9.145
X_MAX = +(HALF_DOUBLES_WIDTH + SIDELINE_CLEAR)  # +9.145
Y_MIN = -(HALF_LENGTH + BASELINE_CLEAR)  # -18.285
Y_MAX = +(HALF_LENGTH + BASELINE_CLEAR)  # +18.285


def court_keypoints_3d() -> Tensor:
    """Return 20 court keypoints (idx 0..19) as a (20, 3) tensor.

    idx の意味はドキュメント通り:

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
        (x_post_L, 0.0, 0.0),              # 15 left net post base
        (x_post_L, 0.0, NET_HEIGHT_POST),  # 16 left net post top
        (x_post_R, 0.0, 0.0),              # 17 right net post base
        (x_post_R, 0.0, NET_HEIGHT_POST),  # 18 right net post top
        (0.0, 0.0, NET_HEIGHT_CENTER),     # 19 center strap top
    ]
    return torch.tensor(pts, dtype=torch.float32)


# -----------------------------
# シンプルなピンホールカメラ
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
    """Create a Camera that looks from `center` to `look_at`.

    Args:
        center: Camera position in world coords.
        look_at: Target point the camera looks at.
        image_size: (width, height) in pixels.
        hfov_deg: Horizontal field of view in degrees.
    """
    center_t = torch.tensor(center, dtype=torch.float32)
    look_t = torch.tensor(look_at, dtype=torch.float32)

    z_cam = look_t - center_t
    z_cam = z_cam / (z_cam.norm() + 1e-8)

    up_world = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    x_cam = torch.cross(up_world, z_cam)
    x_norm = x_cam.norm()
    if x_norm < 1e-6:
        # 真上/真下などで up と平行になったときのフォールバック
        up_world = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        x_cam = torch.cross(up_world, z_cam)
        x_norm = x_cam.norm()
    x_cam = x_cam / (x_norm + 1e-8)
    y_cam = torch.cross(z_cam, x_cam)

    R = torch.stack([x_cam, y_cam, z_cam], dim=0)  # world -> camera

    w, h = image_size
    hfov_rad = float(hfov_deg) * torch.pi / 180.0
    f = 0.5 * float(w) / torch.tan(0.5 * hfov_rad)
    cx = 0.5 * float(w)
    cy = 0.5 * float(h)

    return Camera(C=center_t, R=R, f=float(f), cx=cx, cy=cy, w=w, h=h)


def project_points(cam: Camera, xyz: Tensor) -> tuple[Tensor, Tensor]:
    """Project 3D points to image plane.

    Returns:
        uv: (N,2) tensor of pixel coordinates (u,v).
        mask: (N,) boolean tensor; True if point is in front of camera (z>0).
    """
    if xyz.numel() == 0:
        return xyz.new_zeros((0, 2)), xyz.new_zeros((0,), dtype=torch.bool)

    # world -> camera
    X = xyz - cam.C.view(1, 3)
    Xc = X @ cam.R.t()

    z = Xc[:, 2]
    mask = z > 1e-6
    z_safe = torch.where(mask, z, torch.ones_like(z))

    # ★ v 方向の符号を反転して「上 = 画面上」に揃える
    u = cam.f * (Xc[:, 0] / z_safe) + cam.cx
    v = cam.f * (-Xc[:, 1] / z_safe) + cam.cy

    uv = torch.stack([u, v], dim=-1)
    return uv, mask


# -----------------------------
# フェンス上のカメラ位置サンプリング
# -----------------------------


def sample_camera_position_on_fence(t: float, side: str) -> Tuple[float, float, float]:
    """Sample a 3D camera position on the fence rectangle.

    Args:
        t: Parameter in [0,1] along the chosen side.
        side: One of {"near", "far", "left", "right"}.
    Returns:
        (x, y, z) position on the fence at height ~FENCE_HEIGHT.
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

    z = FENCE_HEIGHT  # とりあえず 3.0m 固定（必要ならランダム化）
    return float(x), float(y), float(z)

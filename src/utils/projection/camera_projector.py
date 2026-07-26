"""Camera projection utilities shared across dataset generators."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor

from src.utils.schema.court import (
    BASELINE_CLEAR,
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
    """Create a pinhole camera pointed at ``look_at``.

    Uses the OpenCV camera convention: ``x`` = image right, ``y`` = image
    down, ``z`` = forward, so ``R`` is a proper rotation and the projection
    ``u = f*Xc/Zc + cx``, ``v = f*Yc/Zc + cy`` matches what a physical camera
    (and cv2.solvePnP) produces. The previous basis (``x = up × z`` combined
    with a v-flip in :func:`project_points`) rendered a left-right mirrored
    image, which is undetectable on the bilaterally symmetric court but
    breaks sim-to-real transfer of keypoint-indexed models.
    """
    center_t = torch.tensor(center, dtype=torch.float32)
    look_t = torch.tensor(look_at, dtype=torch.float32)

    z_cam = look_t - center_t
    z_cam = z_cam / (z_cam.norm() + 1e-8)

    up_world = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    x_cam = torch.cross(z_cam, up_world, dim=0)  # image right
    x_norm = x_cam.norm()
    if x_norm < 1e-6:
        # Fallback when looking straight up/down
        up_world = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        x_cam = torch.cross(z_cam, up_world, dim=0)
        x_norm = x_cam.norm()
    x_cam = x_cam / (x_norm + 1e-8)
    y_cam = torch.cross(z_cam, x_cam, dim=0)  # image down

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

    # OpenCV convention: y_cam already points image-down, so no v-flip.
    u = cam.f * (Xc[:, 0] / z_safe) + cam.cx
    v = cam.f * (Xc[:, 1] / z_safe) + cam.cy

    uv = torch.stack([u, v], dim=-1)
    return uv, mask


def camera_from_mapping(params: _CameraConfigMapping) -> Camera:
    """Reconstruct a :class:`Camera` from serialized scene parameters."""
    required = ("C", "R", "f", "cx", "cy", "w", "h")
    missing = [key for key in required if params.get(key, None) is None]
    if missing:
        raise KeyError(f"Serialized camera parameters are missing: {missing}")
    return Camera(
        C=torch.as_tensor(params.get("C"), dtype=torch.float32),
        R=torch.as_tensor(params.get("R"), dtype=torch.float32),
        f=float(params.get("f")),
        cx=float(params.get("cx")),
        cy=float(params.get("cy")),
        w=int(params.get("w")),
        h=int(params.get("h")),
    )


FIXED_LAYOUT = "fixed"
BROADCAST_LAYOUT = "broadcast"
SUPPORTED_LAYOUTS = (FIXED_LAYOUT, BROADCAST_LAYOUT)


class _CameraConfigMapping(Protocol):
    """Minimal config interface shared by dict and OmegaConf DictConfig."""

    def get(self, key: str, default: Any = ...) -> Any: ...


@dataclass
class CameraConfig:
    """Configuration for camera generation.

    Supports per-camera perturbation of intrinsics and look-at direction
    for dataset variation.

    Two camera layouts are available, selected by ``layout``:

    - ``"fixed"`` (default): the surveillance-style 6-camera rig
      (4 fence corners + 2 baseline midpoints). See :meth:`CameraProjector.fixed_cameras`.
    - ``"broadcast"``: two TV "high main" cameras, one behind each baseline,
      centred on the court and elevated, looking down the court's length.
      This mirrors public-broadcast tennis framing so that models trained on
      monocular views transfer to real broadcast footage.
      See :meth:`CameraProjector.broadcast_cameras`.
    """

    z_min: float = 3.0
    z_max: float = 5.0
    hfov_deg: float = 60.0
    image_size: tuple[int, int] = (1280, 720)
    fixed_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fixed_baseline_clear_extra: float = 0.0
    fixed_position_noise_radius: float = 2.0
    fixed_look_at_xy_radius: float = 1.0

    # Camera layout selector. See class docstring for supported values.
    layout: str = FIXED_LAYOUT

    # --- Broadcast ("high main") layout parameters ---------------------
    # Grounded by overlaying the projected court on a real ATP broadcast frame:
    # a distant, elevated telephoto camera ~20 m behind the baseline, ~7 m high,
    # ~35 deg HFOV, framing the whole court with realistic perspective
    # compression.
    broadcast_setback: float = 20.0  # metres behind the baseline (beyond HALF_LENGTH)
    broadcast_height: float = 7.0  # camera elevation above the court (m)
    broadcast_hfov_deg: float = 35.0  # telephoto horizontal FOV (deg)
    broadcast_look_at_y: float = 0.0  # look-at target Y (0 = court centre)
    broadcast_look_at_height: float = 0.5  # look-at target Z above the court (m)
    broadcast_position_noise_radius: float = 1.0  # per-scene center jitter (m)
    broadcast_look_at_xy_radius: float = 1.0  # per-scene look-at jitter on z=0 plane (m)
    broadcast_hfov_jitter_deg: float = 2.0  # per-scene zoom (HFOV) jitter (deg)

    # --- Optional wide per-camera randomization (sim-to-real) -----------
    # When a range is set it replaces the corresponding fixed value above and
    # is sampled uniformly per camera. ``broadcast_court_width_frac_range``
    # samples the apparent width of the camera-side baseline (doubles-corner
    # u-span as a fraction of image width) and solves the HFOV that realizes
    # it, guaranteeing sane framing across the whole setback/height range;
    # it is mutually exclusive with ``broadcast_hfov_jitter_deg`` > 0.
    broadcast_setback_range: tuple[float, float] | None = None
    broadcast_height_range: tuple[float, float] | None = None
    broadcast_court_width_frac_range: tuple[float, float] | None = None


def _optional_range(
    cfg: _CameraConfigMapping, key: str
) -> tuple[float, float] | None:
    """Parse an optional ``[lo, hi]`` range from a camera config mapping."""
    value = cfg.get(key, None)
    if value is None:
        return None
    lo, hi = (float(v) for v in value)
    if lo > hi:
        raise ValueError(f"{key} must satisfy lo <= hi, got [{lo}, {hi}].")
    return lo, hi


def camera_config_from_mapping(cfg: _CameraConfigMapping) -> CameraConfig:
    """Build a :class:`CameraConfig` from a Hydra/OmegaConf ``camera`` section.

    Centralises the mapping used by both the BLCS and PLCS scene generators so
    the two do not drift. Any key absent from ``cfg`` falls back to the
    :class:`CameraConfig` default, which keeps existing ``camera/default.yaml``
    files (that predate the broadcast-layout fields) working unchanged.

    Args:
        cfg: A mapping-like config (``dict`` or OmegaConf ``DictConfig``).
    """
    if not hasattr(cfg, "get"):
        raise TypeError(
            "camera_config_from_mapping expects a mapping-like config with a "
            f".get() method, got {type(cfg).__name__}."
        )

    defaults = CameraConfig()

    def _f(key: str, default: float) -> float:
        return float(cfg.get(key, default))

    return CameraConfig(
        z_min=_f("z_min", defaults.z_min),
        z_max=_f("z_max", defaults.z_max),
        hfov_deg=_f("hfov_deg", defaults.hfov_deg),
        image_size=tuple(cfg.get("image_size", defaults.image_size)),
        fixed_look_at=tuple(cfg.get("fixed_look_at", defaults.fixed_look_at)),
        fixed_baseline_clear_extra=_f(
            "fixed_baseline_clear_extra", defaults.fixed_baseline_clear_extra
        ),
        fixed_position_noise_radius=_f(
            "fixed_position_noise_radius", defaults.fixed_position_noise_radius
        ),
        fixed_look_at_xy_radius=_f(
            "fixed_look_at_xy_radius", defaults.fixed_look_at_xy_radius
        ),
        layout=str(cfg.get("layout", defaults.layout)),
        broadcast_setback=_f("broadcast_setback", defaults.broadcast_setback),
        broadcast_height=_f("broadcast_height", defaults.broadcast_height),
        broadcast_hfov_deg=_f("broadcast_hfov_deg", defaults.broadcast_hfov_deg),
        broadcast_look_at_y=_f("broadcast_look_at_y", defaults.broadcast_look_at_y),
        broadcast_look_at_height=_f(
            "broadcast_look_at_height", defaults.broadcast_look_at_height
        ),
        broadcast_position_noise_radius=_f(
            "broadcast_position_noise_radius", defaults.broadcast_position_noise_radius
        ),
        broadcast_look_at_xy_radius=_f(
            "broadcast_look_at_xy_radius", defaults.broadcast_look_at_xy_radius
        ),
        broadcast_hfov_jitter_deg=_f(
            "broadcast_hfov_jitter_deg", defaults.broadcast_hfov_jitter_deg
        ),
        broadcast_setback_range=_optional_range(cfg, "broadcast_setback_range"),
        broadcast_height_range=_optional_range(cfg, "broadcast_height_range"),
        broadcast_court_width_frac_range=_optional_range(
            cfg, "broadcast_court_width_frac_range"
        ),
    )


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
        self.court_config = court_config
        self.court_kp_3d = court_keypoints_3d(court_config)

    @staticmethod
    def _fence_extents(baseline_clear_extra: float = 0.0) -> tuple[float, float]:
        """Return outer fence extents in x/y from court center."""
        fence_x = HALF_DOUBLES_WIDTH + SIDELINE_CLEAR
        fence_y = HALF_LENGTH + BASELINE_CLEAR + max(0.0, baseline_clear_extra)
        return fence_x, fence_y

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
        """Build the fixed 6-camera layout (4 corners + 2 baseline midpoints).

        Corner cameras use ``z_min``. Midpoint cameras use ``z_max``.
        When both fixed-camera noise radii are zero, the legacy layout is
        preserved except that the left/right side midpoints are omitted.
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
        baseline_midpoints = [
            (0.0, +fence_y, cfg.z_max),
            (0.0, -fence_y, cfg.z_max),
        ]

        cams: list[Camera] = []
        for base_center in corners + baseline_midpoints:
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

    def broadcast_cameras(self) -> list[Camera]:
        """Build the 2-camera TV "high main" broadcast layout.

        One elevated camera sits behind each baseline, centred on the court
        (x=0) and looking down the court's length toward the far side. This
        reproduces the framing of public-broadcast tennis coverage so that a
        model trained on these monocular views transfers to real footage.

        Per-scene variation is applied via ``broadcast_position_noise_radius``
        (center jitter), ``broadcast_look_at_xy_radius`` (look-at jitter) and
        ``broadcast_hfov_jitter_deg`` (zoom jitter). With all three set to zero
        the two cameras are deterministic mirror images across the net.

        When the wide randomization ranges (``broadcast_setback_range``,
        ``broadcast_height_range``, ``broadcast_court_width_frac_range``) are
        set, setback/height are sampled uniformly per camera and the HFOV is
        solved so the camera-side baseline spans the sampled fraction of the
        image width. Real broadcast footage varies far beyond the narrow
        jitter defaults (e.g. tennis_clip.mp4 PnP-fits to setback 32.7 m,
        height 11.4 m, HFOV 28 deg), so sim-to-real training should use the
        ranges.
        """
        cfg = self.config
        if (
            cfg.broadcast_court_width_frac_range is not None
            and cfg.broadcast_hfov_jitter_deg > 0.0
        ):
            raise ValueError(
                "broadcast_court_width_frac_range and broadcast_hfov_jitter_deg "
                "are mutually exclusive; set broadcast_hfov_jitter_deg to 0 "
                "when using the framing-fraction range."
            )

        cams: list[Camera] = []
        for side in (-1.0, +1.0):  # behind near / far baseline
            setback = (
                random.uniform(*cfg.broadcast_setback_range)
                if cfg.broadcast_setback_range is not None
                else cfg.broadcast_setback
            )
            height = (
                random.uniform(*cfg.broadcast_height_range)
                if cfg.broadcast_height_range is not None
                else cfg.broadcast_height
            )
            base_center = (0.0, side * (HALF_LENGTH + setback), height)

            dx, dy, dz = self._sample_uniform_offset_in_ball(
                cfg.broadcast_position_noise_radius
            )
            center = (
                base_center[0] + dx,
                base_center[1] + dy,
                base_center[2] + dz,
            )

            target_x, target_y = self._sample_uniform_xy_in_disk(
                cfg.broadcast_look_at_xy_radius
            )
            look_at_target = (
                target_x,
                cfg.broadcast_look_at_y + target_y,
                cfg.broadcast_look_at_height,
            )

            hfov = cfg.broadcast_hfov_deg
            if cfg.broadcast_hfov_jitter_deg > 0.0:
                hfov += random.uniform(
                    -cfg.broadcast_hfov_jitter_deg, cfg.broadcast_hfov_jitter_deg
                )

            cam = make_look_at_camera(
                center=center,
                look_at=look_at_target,
                hfov_deg=hfov,
                image_size=cfg.image_size,
            )
            if cfg.broadcast_court_width_frac_range is not None:
                frac = random.uniform(*cfg.broadcast_court_width_frac_range)
                cam = self._solve_broadcast_framing(cam, side=side, width_frac=frac)
            cams.append(cam)
        return cams

    def _solve_broadcast_framing(
        self, cam: Camera, *, side: float, width_frac: float
    ) -> Camera:
        """Rescale focal length so the camera-side baseline spans ``width_frac``.

        For fixed extrinsics the projected u-span of the near-baseline doubles
        corners is exactly proportional to the focal length, so a single
        provisional projection determines the required HFOV in closed form.
        """
        corners = torch.tensor(
            [
                [-HALF_DOUBLES_WIDTH, side * HALF_LENGTH, 0.0],
                [+HALF_DOUBLES_WIDTH, side * HALF_LENGTH, 0.0],
            ],
            dtype=torch.float32,
        )
        uv, in_front = project_points(cam, corners)
        if not bool(in_front.all()):
            raise ValueError(
                "Broadcast framing solve failed: baseline corners behind the "
                f"camera (center={cam.C.tolist()})."
            )
        span_norm = float((uv[1, 0] - uv[0, 0]).abs()) / float(cam.w)
        if span_norm <= 1e-6:
            raise ValueError(
                "Broadcast framing solve failed: degenerate baseline span "
                f"(center={cam.C.tolist()})."
            )
        f_new = cam.f * (width_frac / span_norm)
        return Camera(
            C=cam.C, R=cam.R, f=f_new, cx=cam.cx, cy=cam.cy, w=cam.w, h=cam.h
        )

    def cameras(self) -> list[Camera]:
        """Return the camera rig for the configured ``layout``.

        Dispatches to :meth:`fixed_cameras` or :meth:`broadcast_cameras`.
        Raises on an unknown layout rather than silently falling back.
        """
        layout = self.config.layout
        if layout == FIXED_LAYOUT:
            return self.fixed_cameras()
        if layout == BROADCAST_LAYOUT:
            return self.broadcast_cameras()
        raise ValueError(
            f"Unknown camera layout {layout!r}. "
            f"Supported layouts: {list(SUPPORTED_LAYOUTS)}."
        )

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

    def generate_camera_view(
        self,
        points_3d: Tensor,
        camera: Camera,
    ) -> CameraView:
        """Generate a camera view with optional subject projection."""
        court_kp_uv, court_kp_visible = self.project_court_keypoints(camera)
        points_uv, points_visible = self.project_points_to_uv(points_3d, camera)

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

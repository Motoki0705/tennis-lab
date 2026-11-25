"""Monocular camera, synthetic bounding-box, and CLIFF coordinate transforms for PLCS.

Implements the exact geometric contract shared by monocular PLCS training,
HTD-Refine trajectory optimization, and ``tennis_scene`` pipeline inference:

1. Synthetic square ``bbx_xys = [cx, cy, s]`` generation from visible pixel
   COCO-17 2D keypoints using the 192:256 detector aspect fit, with explicit
   ``frame_observed`` masking so occluded/zeroed keypoints never corrupt the box.
2. Detector-realistic bounding-box perturbation (scale jitter, center shift,
   gap-isolated AR(1) temporal drift, and partial-occlusion cropping).
3. Bounding-box local 2D normalization into ``[-1, 1]`` with strict zeroing of
   invisible or out-of-box joints.
4. Generalized CLIFF bbox features and exact forward/inverse transforms between
   CLIFF root parameters ``[s, tx, ty]`` + hip-centered camera-local 3D offsets
   and physical court-space 3D COCO-17 skeletons ``(T, 17, 3)`` [m] under the
   OpenCV pinhole convention ``X_c = R_w2c (X_w - C_w)``.
5. Dimensionless 21-D camera intrinsics/extrinsics Early Fusion features and
   Calibration Noise Augmentation modeling real ``court_calibration`` PnP errors.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, cast

import torch
from torch import Tensor

from src.utils.schema.court_normalization import normalize_court_position
from src.utils.schema.player import COCO_KP_IDX, NUM_HUMAN_KP

LEFT_HIP_IDX: Final[int] = COCO_KP_IDX["left_hip"]
RIGHT_HIP_IDX: Final[int] = COCO_KP_IDX["right_hip"]


def _fp32_geometry_context(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    """Disable mixed-precision autocast for exact FP32/FP64 camera geometry math."""
    if device.type in {"cpu", "cuda", "mps", "xpu"}:
        return cast(
            "contextlib.AbstractContextManager[Any]",
            torch.amp.autocast(device_type=device.type, enabled=False),
        )
    return contextlib.nullcontext()


def _geometry_dtype(*tensors: Tensor) -> torch.dtype:
    """Promote low-precision (BF16/FP16) geometry operands to FP32 while keeping FP64."""
    for tensor in tensors:
        if tensor.dtype == torch.float64:
            return torch.float64
    return torch.float32

BBX_XYS_DIM: Final[int] = 3
CLIFFCAM_DIM: Final[int] = 3
CAMERA_FEATURE_DIM: Final[int] = 21
_MIN_DEPTH_M: Final[float] = 1e-4
_MIN_SCALE_SAFE: Final[float] = 1e-6


@dataclass(frozen=True, slots=True)
class SyntheticBBoxConfig:
    """Deterministic synthetic square bbox extraction settings."""

    aspect_ratio_wh: tuple[float, float] = (192.0, 256.0)
    base_enlarge: float = 1.2
    min_visible_joints: int = 2
    min_box_size_px: float = 16.0
    vis_threshold: float = 0.0

    def __post_init__(self) -> None:
        w_ref, h_ref = self.aspect_ratio_wh
        if not (math.isfinite(w_ref) and math.isfinite(h_ref) and w_ref > 0 and h_ref > 0):
            raise ValueError("aspect_ratio_wh must contain two positive finite floats.")
        if not (math.isfinite(self.base_enlarge) and self.base_enlarge > 0):
            raise ValueError("base_enlarge must be positive and finite.")
        if self.min_visible_joints < 1 or self.min_visible_joints > NUM_HUMAN_KP:
            raise ValueError(
                f"min_visible_joints must be in [1, {NUM_HUMAN_KP}], "
                f"got {self.min_visible_joints}."
            )
        if not (math.isfinite(self.min_box_size_px) and self.min_box_size_px > 0):
            raise ValueError("min_box_size_px must be positive and finite.")
        if not (math.isfinite(self.vis_threshold) and self.vis_threshold >= 0.0):
            raise ValueError("vis_threshold must be non-negative and finite.")


@dataclass(frozen=True, slots=True)
class BBoxAugmentationConfig:
    """Train-time detector bbox jitter, AR(1) drift, and partial-crop settings."""

    enabled: bool = True
    prob: float = 1.0
    scale_range: tuple[float, float] = (1.05, 1.35)
    shift_xy_frac: float = 0.08
    scale_drift_std: float = 0.02
    center_drift_frac_std: float = 0.02
    drift_decay: float = 0.85
    partial_crop_prob: float = 0.15
    partial_crop_scale_range: tuple[float, float] = (0.72, 0.92)
    min_box_size_px: float = 16.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError("prob must be within [0, 1].")
        s_lo, s_hi = self.scale_range
        if not (0.0 < s_lo <= s_hi and math.isfinite(s_lo) and math.isfinite(s_hi)):
            raise ValueError("scale_range must be positive and ordered.")
        c_lo, c_hi = self.partial_crop_scale_range
        if not (0.0 < c_lo <= c_hi <= 1.0 and math.isfinite(c_lo) and math.isfinite(c_hi)):
            raise ValueError("partial_crop_scale_range must be ordered within (0, 1].")
        if not 0.0 <= self.partial_crop_prob <= 1.0:
            raise ValueError("partial_crop_prob must be within [0, 1].")
        if not 0.0 <= self.drift_decay <= 1.0:
            raise ValueError("drift_decay must be within [0, 1].")
        for name, val in (
            ("shift_xy_frac", self.shift_xy_frac),
            ("scale_drift_std", self.scale_drift_std),
            ("center_drift_frac_std", self.center_drift_frac_std),
        ):
            if not (math.isfinite(val) and val >= 0.0):
                raise ValueError(f"{name} must be non-negative and finite.")
        if not (math.isfinite(self.min_box_size_px) and self.min_box_size_px > 0.0):
            raise ValueError("min_box_size_px must be positive and finite.")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> BBoxAugmentationConfig:
        if mapping is None:
            return cls()
        scale_raw = cast("tuple[float, float] | list[float]", mapping["scale_range"])
        crop_raw = cast(
            "tuple[float, float] | list[float]", mapping["partial_crop_scale_range"]
        )
        return cls(
            enabled=bool(mapping["enabled"]),
            prob=float(cast("float | int", mapping["prob"])),
            scale_range=(float(scale_raw[0]), float(scale_raw[1])),
            shift_xy_frac=float(cast("float | int", mapping["shift_xy_frac"])),
            scale_drift_std=float(cast("float | int", mapping["scale_drift_std"])),
            center_drift_frac_std=float(
                cast("float | int", mapping["center_drift_frac_std"])
            ),
            drift_decay=float(cast("float | int", mapping["drift_decay"])),
            partial_crop_prob=float(cast("float | int", mapping["partial_crop_prob"])),
            partial_crop_scale_range=(float(crop_raw[0]), float(crop_raw[1])),
            min_box_size_px=float(
                cast("float | int", mapping.get("min_box_size_px", 16.0))
            ),
        )


@dataclass(frozen=True, slots=True)
class CalibrationNoiseConfig:
    """Train-time court camera calibration perturbation settings (~few-px court RMSE)."""

    enabled: bool = True
    prob: float = 1.0
    focal_rel_std: float = 0.0025
    principal_point_rel_std: float = 0.0012
    rotation_deg_std: float = 0.08
    translation_m_std: float = 0.03

    def __post_init__(self) -> None:
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError("prob must be within [0, 1].")
        for name, val in (
            ("focal_rel_std", self.focal_rel_std),
            ("principal_point_rel_std", self.principal_point_rel_std),
            ("rotation_deg_std", self.rotation_deg_std),
            ("translation_m_std", self.translation_m_std),
        ):
            if not (math.isfinite(val) and val >= 0.0):
                raise ValueError(f"{name} must be non-negative and finite.")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> CalibrationNoiseConfig:
        if mapping is None:
            return cls()
        return cls(
            enabled=bool(mapping["enabled"]),
            prob=float(cast("float | int", mapping["prob"])),
            focal_rel_std=float(cast("float | int", mapping["focal_rel_std"])),
            principal_point_rel_std=float(
                cast("float | int", mapping["principal_point_rel_std"])
            ),
            rotation_deg_std=float(cast("float | int", mapping["rotation_deg_std"])),
            translation_m_std=float(cast("float | int", mapping["translation_m_std"])),
        )


_DEFAULT_SYNTHETIC_BBOX_CONFIG: Final[SyntheticBBoxConfig] = SyntheticBBoxConfig()
_DEFAULT_BBOX_AUG_CONFIG: Final[BBoxAugmentationConfig] = BBoxAugmentationConfig()
_DEFAULT_CALIB_NOISE_CONFIG: Final[CalibrationNoiseConfig] = CalibrationNoiseConfig()


@dataclass(frozen=True, slots=True)
class PerturbedCameraParams:
    """Calibrated camera parameters passed as model input after optional noise."""

    R_w2c: Tensor
    C_w: Tensor
    fx: Tensor
    fy: Tensor
    cx0: Tensor
    cy0: Tensor


def _rand_uniform(
    shape: tuple[int, ...],
    *,
    low: float,
    high: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    return u * (high - low) + low


def _randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def so3_exp_map(omega: Tensor) -> Tensor:
    """Map axis-angle vectors ``(..., 3)`` [rad] to proper rotation matrices ``(..., 3, 3)``."""
    if omega.shape[-1] != 3:
        raise ValueError(f"omega must have shape (..., 3), got {tuple(omega.shape)}.")
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq.clamp(min=1e-12))
    small = theta_sq < 1e-8
    coef_a = torch.where(small, 1.0 - theta_sq / 6.0, torch.sin(theta) / theta)
    coef_b = torch.where(
        small,
        0.5 - theta_sq / 24.0,
        (1.0 - torch.cos(theta)) / theta_sq.clamp(min=1e-12),
    )

    wx, wy, wz = omega.unbind(dim=-1)
    zero = torch.zeros_like(wx)
    skew = torch.stack(
        [
            torch.stack([zero, -wz, wy], dim=-1),
            torch.stack([wz, zero, -wx], dim=-1),
            torch.stack([-wy, wx, zero], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(skew)
    return eye + coef_a.unsqueeze(-1) * skew + coef_b.unsqueeze(-1) * (skew @ skew)


def validate_pinhole_camera_tensors(
    R_w2c: Tensor,
    C_w: Tensor,
    *,
    fx: Tensor,
    fy: Tensor,
    cx0: Tensor,
    cy0: Tensor,
    image_w: Tensor,
    image_h: Tensor,
) -> None:
    """Validate calibrated OpenCV pinhole camera tensors with zero silent fallbacks."""
    for name, tensor in (
        ("R_w2c", R_w2c),
        ("C_w", C_w),
        ("fx", fx),
        ("fy", fy),
        ("cx0", cx0),
        ("cy0", cy0),
        ("image_w", image_w),
        ("image_h", image_h),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Camera parameter {name!r} contains non-finite values.")
    if R_w2c.ndim < 2 or R_w2c.shape[-2:] != (3, 3):
        raise ValueError(
            f"R_w2c must have shape (..., 3, 3), got {tuple(R_w2c.shape)}."
        )
    if C_w.ndim < 1 or C_w.shape[-1] != 3:
        raise ValueError(f"C_w must have shape (..., 3), got {tuple(C_w.shape)}.")
    if bool((fx <= 0).any()) or bool((fy <= 0).any()):
        raise ValueError("Focal lengths fx and fy must be strictly positive.")
    if bool((image_w <= 0).any()) or bool((image_h <= 0).any()):
        raise ValueError("Image dimensions image_w and image_h must be strictly positive.")
    eye = torch.eye(3, device=R_w2c.device, dtype=R_w2c.dtype)
    ortho_err = (R_w2c @ R_w2c.transpose(-1, -2) - eye).abs().amax()
    if float(ortho_err.item()) > 1e-3:
        raise ValueError(
            f"R_w2c must be orthonormal (max |R R^T - I| = {float(ortho_err.item()):.3e})."
        )
    det = torch.linalg.det(R_w2c)
    if bool(((det - 1.0).abs() > 1e-3).any()):
        raise ValueError("R_w2c must be a proper rotation matrix with det(R) == +1.")


def compute_synthetic_bbx_xys(
    kp2d_px: Tensor,
    kp_vis: Tensor,
    *,
    config: SyntheticBBoxConfig = _DEFAULT_SYNTHETIC_BBOX_CONFIG,
) -> tuple[Tensor, Tensor]:
    """Compute GVHMR-style square ``bbx_xys = [cx, cy, s]`` from visible 2D joints.

    Args:
        kp2d_px: Pixel 2D keypoints ``(..., T, J, 2)``.
        kp_vis: Joint visibility / confidence ``(..., T, J)``.
        config: Synthetic bbox configuration.

    Returns:
        ``(bbx_xys, frame_observed)``:
        - ``bbx_xys``: ``(..., T, 3)`` in pixel units; strictly ``0`` where unobserved.
        - ``frame_observed``: ``(..., T)`` boolean mask of frames with at least
          ``config.min_visible_joints`` finite visible joints.
    """
    if kp2d_px.ndim < 3 or kp2d_px.shape[-1] != 2:
        raise ValueError(
            f"kp2d_px must have shape (..., T, J, 2), got {tuple(kp2d_px.shape)}."
        )
    if kp_vis.shape != kp2d_px.shape[:-1]:
        raise ValueError(
            f"kp_vis shape {tuple(kp_vis.shape)} must match {tuple(kp2d_px.shape[:-1])}."
        )
    if not torch.isfinite(kp2d_px).all() or not torch.isfinite(kp_vis).all():
        raise ValueError("kp2d_px and kp_vis must be finite.")

    vis_mask = kp_vis > config.vis_threshold
    vis_count = vis_mask.sum(dim=-1)
    frame_observed = vis_count >= config.min_visible_joints

    pos_inf = torch.tensor(float("inf"), device=kp2d_px.device, dtype=kp2d_px.dtype)
    neg_inf = torch.tensor(float("-inf"), device=kp2d_px.device, dtype=kp2d_px.dtype)

    x = kp2d_px[..., 0]
    y = kp2d_px[..., 1]
    min_x = torch.where(vis_mask, x, pos_inf).amin(dim=-1)
    max_x = torch.where(vis_mask, x, neg_inf).amax(dim=-1)
    min_y = torch.where(vis_mask, y, pos_inf).amin(dim=-1)
    max_y = torch.where(vis_mask, y, neg_inf).amax(dim=-1)

    min_x = torch.where(frame_observed, min_x, torch.zeros_like(min_x))
    max_x = torch.where(frame_observed, max_x, torch.zeros_like(max_x))
    min_y = torch.where(frame_observed, min_y, torch.zeros_like(min_y))
    max_y = torch.where(frame_observed, max_y, torch.zeros_like(max_y))

    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    width = (max_x - min_x).clamp(min=0.0)
    height = (max_y - min_y).clamp(min=0.0)

    aspect = config.aspect_ratio_wh[0] / config.aspect_ratio_wh[1]
    fitted_h = torch.where(width > aspect * height, width / aspect, height)
    fitted_w = torch.where(width < aspect * height, height * aspect, width)
    size = (torch.maximum(fitted_w, fitted_h) * config.base_enlarge).clamp(
        min=config.min_box_size_px
    )

    bbx_xys = torch.stack([center_x, center_y, size], dim=-1)
    bbx_xys = torch.where(
        frame_observed.unsqueeze(-1), bbx_xys, torch.zeros_like(bbx_xys)
    )
    return bbx_xys, frame_observed


def augment_bbx_xys(
    bbx_xys: Tensor,
    frame_observed: Tensor,
    *,
    base_enlarge: float = 1.2,
    config: BBoxAugmentationConfig = _DEFAULT_BBOX_AUG_CONFIG,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply detector scale/shift jitter, gap-safe AR(1) drift, and partial crop."""
    if bbx_xys.ndim < 2 or bbx_xys.shape[-1] != BBX_XYS_DIM:
        raise ValueError(
            f"bbx_xys must have shape (..., T, 3), got {tuple(bbx_xys.shape)}."
        )
    if frame_observed.shape != bbx_xys.shape[:-1] or frame_observed.dtype != torch.bool:
        raise ValueError(
            "frame_observed must be a bool tensor matching bbx_xys[..., 0]."
        )
    if not config.enabled or config.prob <= 0.0:
        return torch.where(
            frame_observed.unsqueeze(-1), bbx_xys, torch.zeros_like(bbx_xys)
        )

    prefix_shape = bbx_xys.shape[:-2]
    seq_len = bbx_xys.shape[-2]
    device = bbx_xys.device
    dtype = bbx_xys.dtype

    flat_bbx = bbx_xys.reshape(-1, seq_len, BBX_XYS_DIM).clone()
    flat_obs = frame_observed.reshape(-1, seq_len)
    num_tracks = flat_bbx.shape[0]

    active_tracks = (
        torch.rand((num_tracks,), device=device, dtype=dtype, generator=generator)
        < config.prob
    )
    if not bool(active_tracks.any()):
        return torch.where(
            frame_observed.unsqueeze(-1), bbx_xys, torch.zeros_like(bbx_xys)
        )

    raw_size = flat_bbx[..., 2] / max(base_enlarge, _MIN_SCALE_SAFE)
    scale_factor = _rand_uniform(
        (num_tracks, seq_len),
        low=config.scale_range[0],
        high=config.scale_range[1],
        device=device,
        dtype=dtype,
        generator=generator,
    )
    shift_x = _rand_uniform(
        (num_tracks, seq_len),
        low=-config.shift_xy_frac,
        high=config.shift_xy_frac,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    shift_y = _rand_uniform(
        (num_tracks, seq_len),
        low=-config.shift_xy_frac,
        high=config.shift_xy_frac,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    # Gap-isolated AR(1) drift: resets cleanly whenever a frame is unobserved so
    # drift state never leaks across missing track intervals.
    if config.scale_drift_std > 0.0 or config.center_drift_frac_std > 0.0:
        eps_s = _randn(
            (num_tracks, seq_len), device=device, dtype=dtype, generator=generator
        ) * config.scale_drift_std
        eps_xy = _randn(
            (num_tracks, seq_len, 2), device=device, dtype=dtype, generator=generator
        ) * config.center_drift_frac_std
        drift_s = torch.zeros((num_tracks,), device=device, dtype=dtype)
        drift_xy = torch.zeros((num_tracks, 2), device=device, dtype=dtype)
        decay = config.drift_decay
        for t in range(seq_len):
            obs_t = flat_obs[:, t]
            drift_s = torch.where(
                obs_t, decay * drift_s + eps_s[:, t], torch.zeros_like(drift_s)
            )
            drift_xy = torch.where(
                obs_t.unsqueeze(-1),
                decay * drift_xy + eps_xy[:, t],
                torch.zeros_like(drift_xy),
            )
            scale_factor[:, t] = (scale_factor[:, t] + drift_s).clamp(min=0.5)
            shift_x[:, t] = shift_x[:, t] + drift_xy[:, 0]
            shift_y[:, t] = shift_y[:, t] + drift_xy[:, 1]

    new_size = raw_size * scale_factor
    new_cx = flat_bbx[..., 0] + raw_size * shift_x
    new_cy = flat_bbx[..., 1] + raw_size * shift_y

    if config.partial_crop_prob > 0.0:
        crop_mask = (
            torch.rand(
                (num_tracks, seq_len), device=device, dtype=dtype, generator=generator
            )
            < config.partial_crop_prob
        )
        crop_scale = _rand_uniform(
            (num_tracks, seq_len),
            low=config.partial_crop_scale_range[0],
            high=config.partial_crop_scale_range[1],
            device=device,
            dtype=dtype,
            generator=generator,
        )
        crop_dir = _rand_uniform(
            (num_tracks, seq_len, 2),
            low=-0.25,
            high=0.25,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        cropped_size = new_size * crop_scale
        new_cx = torch.where(crop_mask, new_cx + new_size * crop_dir[..., 0], new_cx)
        new_cy = torch.where(crop_mask, new_cy + new_size * crop_dir[..., 1], new_cy)
        new_size = torch.where(crop_mask, cropped_size, new_size)

    new_size = new_size.clamp(min=config.min_box_size_px)
    perturbed = torch.stack([new_cx, new_cy, new_size], dim=-1)
    apply_mask = active_tracks.unsqueeze(-1) & flat_obs
    out = torch.where(apply_mask.unsqueeze(-1), perturbed, flat_bbx)
    out = torch.where(flat_obs.unsqueeze(-1), out, torch.zeros_like(out))
    return out.reshape(*prefix_shape, seq_len, BBX_XYS_DIM)


def normalize_kp2d_bbox(
    kp2d_px: Tensor,
    kp_vis: Tensor,
    bbx_xys: Tensor,
    frame_observed: Tensor,
    *,
    vis_threshold: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Normalize pixel 2D joints into ``[-1, 1]`` square bbox coordinates.

    Joints that are invisible (``kp_vis <= vis_threshold``), fall outside the
    square ``bbx_xys``, or belong to an unobserved frame (``~frame_observed``)
    are strictly zeroed to ``[0, 0, 0]`` with ``joint_visible == False``.
    """
    if kp2d_px.ndim < 3 or kp2d_px.shape[-1] != 2:
        raise ValueError(
            f"kp2d_px must have shape (..., T, J, 2), got {tuple(kp2d_px.shape)}."
        )
    if kp_vis.shape != kp2d_px.shape[:-1]:
        raise ValueError("kp_vis must match kp2d_px[..., 0] shape.")
    if bbx_xys.shape != (*kp2d_px.shape[:-2], BBX_XYS_DIM):
        raise ValueError("bbx_xys must have shape (..., T, 3) matching kp2d_px.")
    if frame_observed.shape != kp2d_px.shape[:-2] or frame_observed.dtype != torch.bool:
        raise ValueError("frame_observed must be a bool tensor of shape (..., T).")

    center = bbx_xys[..., :2].unsqueeze(-2)
    size = bbx_xys[..., 2:3].unsqueeze(-2)
    half = 0.5 * size

    in_box = (
        (kp2d_px[..., 0] >= center[..., 0] - half[..., 0])
        & (kp2d_px[..., 0] <= center[..., 0] + half[..., 0])
        & (kp2d_px[..., 1] >= center[..., 1] - half[..., 0])
        & (kp2d_px[..., 1] <= center[..., 1] + half[..., 0])
    )
    valid_box = frame_observed & (bbx_xys[..., 2] > _MIN_SCALE_SAFE)
    joint_visible = valid_box.unsqueeze(-1) & (kp_vis > vis_threshold) & in_box

    safe_size = size.clamp(min=_MIN_SCALE_SAFE)
    xy_norm = 2.0 * (kp2d_px - center) / safe_size
    xy_norm = torch.where(
        joint_visible.unsqueeze(-1), xy_norm, torch.zeros_like(xy_norm)
    )
    conf = torch.where(
        joint_visible,
        kp_vis.clamp(min=0.0, max=1.0),
        torch.zeros_like(kp_vis),
    )
    return torch.cat([xy_norm, conf.unsqueeze(-1)], dim=-1), joint_visible


def _validate_intrinsics_tensors(
    *,
    fx: Tensor,
    fy: Tensor,
    cx0: Tensor,
    cy0: Tensor,
) -> None:
    """Validate finite, strictly positive focal lengths and finite principal points."""
    for name, tensor in (("fx", fx), ("fy", fy), ("cx0", cx0), ("cy0", cy0)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Camera parameter {name!r} contains non-finite values.")
    if bool((fx <= 0).any()) or bool((fy <= 0).any()):
        raise ValueError("Focal lengths fx and fy must be strictly positive.")


def _validate_extrinsics_and_intrinsics(
    R_w2c: Tensor,
    C_w: Tensor,
    *,
    fx: Tensor,
    fy: Tensor,
    cx0: Tensor,
    cy0: Tensor,
) -> None:
    """Validate calibrated extrinsics (proper SO(3) R_w2c, finite C_w) and intrinsics."""
    _validate_intrinsics_tensors(fx=fx, fy=fy, cx0=cx0, cy0=cy0)
    for name, tensor in (("R_w2c", R_w2c), ("C_w", C_w)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Camera parameter {name!r} contains non-finite values.")
    if R_w2c.ndim < 2 or R_w2c.shape[-2:] != (3, 3):
        raise ValueError(
            f"R_w2c must have shape (..., 3, 3), got {tuple(R_w2c.shape)}."
        )
    if C_w.ndim < 1 or C_w.shape[-1] != 3:
        raise ValueError(f"C_w must have shape (..., 3), got {tuple(C_w.shape)}.")
    geom_dtype = _geometry_dtype(R_w2c, C_w, fx, fy, cx0, cy0)
    with _fp32_geometry_context(R_w2c.device):
        R_fp = R_w2c.to(dtype=geom_dtype)
        eye = torch.eye(3, device=R_fp.device, dtype=geom_dtype)
        ortho_err = (R_fp @ R_fp.transpose(-1, -2) - eye).abs().amax()
        if float(ortho_err.item()) > 1e-3:
            raise ValueError(
                f"R_w2c must be orthonormal (max |R R^T - I| = {float(ortho_err.item()):.3e})."
            )
        det = torch.linalg.det(R_fp)
        if bool(((det - 1.0).abs() > 1e-3).any()):
            raise ValueError(
                "R_w2c must be a proper rotation matrix with det(R) == +1."
            )


def _broadcast_camera_time(
    param: Tensor,
    *,
    batch_shape: tuple[int, ...],
    seq_len: int,
    trailing_dims: int,
    name: str,
) -> Tensor:
    """Broadcast static ``(*batch_shape, *tail)`` or per-frame ``(*batch_shape, T, *tail)`` camera tensors.

    Matching against the explicit ``batch_shape`` (which has length ``k``) vs
    ``(*batch_shape, seq_len)`` (length ``k + 1``) eliminates all ambiguity even
    when ``batch_size == seq_len``.
    """
    lead = param.shape if trailing_dims == 0 else param.shape[:-trailing_dims]
    tail = () if trailing_dims == 0 else param.shape[-trailing_dims:]
    expected_time_lead = (*batch_shape, seq_len)
    if lead == expected_time_lead:
        return param
    if lead == batch_shape:
        if trailing_dims == 0:
            return param.unsqueeze(-1).expand(*batch_shape, seq_len)
        return param.reshape(*batch_shape, 1, *tail).expand(
            *batch_shape, seq_len, *tail
        )
    raise ValueError(
        f"Camera parameter {name!r} leading shape {tuple(lead)} does not match "
        f"static batch shape {batch_shape} or time-varying shape {expected_time_lead}."
    )


def compute_cliffcam_features(
    bbx_xys: Tensor,
    frame_observed: Tensor,
    *,
    fx: Tensor,
    fy: Tensor,
    cx0: Tensor,
    cy0: Tensor,
) -> Tensor:
    """Compute generalized CLIFF camera-bbox features ``[(cx-cx0)/fx, (cy-cy0)/fy, s/sqrt(fx*fy)]``."""
    if bbx_xys.ndim < 2 or bbx_xys.shape[-1] != BBX_XYS_DIM:
        raise ValueError(
            f"bbx_xys must have shape (..., T, 3), got {tuple(bbx_xys.shape)}."
        )
    if frame_observed.shape != bbx_xys.shape[:-1] or frame_observed.dtype != torch.bool:
        raise ValueError(
            "frame_observed must be a bool tensor matching bbx_xys[..., 0]."
        )
    if not torch.isfinite(bbx_xys).all():
        raise ValueError("bbx_xys must contain only finite values.")
    _validate_intrinsics_tensors(fx=fx, fy=fy, cx0=cx0, cy0=cy0)

    batch_shape = tuple(bbx_xys.shape[:-2])
    seq_len = int(bbx_xys.shape[-2])
    fx_t = _broadcast_camera_time(
        fx, batch_shape=batch_shape, seq_len=seq_len, trailing_dims=0, name="fx"
    )
    fy_t = _broadcast_camera_time(
        fy, batch_shape=batch_shape, seq_len=seq_len, trailing_dims=0, name="fy"
    )
    cx0_t = _broadcast_camera_time(
        cx0, batch_shape=batch_shape, seq_len=seq_len, trailing_dims=0, name="cx0"
    )
    cy0_t = _broadcast_camera_time(
        cy0, batch_shape=batch_shape, seq_len=seq_len, trailing_dims=0, name="cy0"
    )

    f_eff = torch.sqrt(fx_t * fy_t)
    cliff_x = (bbx_xys[..., 0] - cx0_t) / fx_t
    cliff_y = (bbx_xys[..., 1] - cy0_t) / fy_t
    cliff_s = bbx_xys[..., 2] / f_eff
    cliff = torch.stack([cliff_x, cliff_y, cliff_s], dim=-1)
    return torch.where(
        frame_observed.unsqueeze(-1), cliff, torch.zeros_like(cliff)
    )


def compute_camera_features(
    R_w2c: Tensor,
    C_w: Tensor,
    *,
    fx: Tensor,
    fy: Tensor,
    cx0: Tensor,
    cy0: Tensor,
    image_w: Tensor,
    image_h: Tensor,
    seq_len: int,
    batch_shape: tuple[int, ...] | None = None,
) -> Tensor:
    """Build 21-D dimensionless camera intrinsics + court-extrinsics features per frame.

    Layout (21 dims):
    - ``[0:6]``: 6D continuous rotation of ``R_w2c`` (first two rows)
    - ``[6:9]``: normalized physical court camera center ``normalize_court_position(C_w)``
    - ``[9:11]``: ``[log(clamp(||C_w||, 0.1)/10), C_w_z / clamp(||C_w||, 0.1)]``
    - ``[11:15]``: ``[log(fx/diag), log(fy/diag), (cx0 - w/2)/w, (cy0 - h/2)/h]``
    - ``[15:21]``: 6D inter-frame relative camera rotation ``R_t @ R_{t-1}^T`` (``I`` at ``t=0``)
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}.")
    validate_pinhole_camera_tensors(
        R_w2c,
        C_w,
        fx=fx,
        fy=fy,
        cx0=cx0,
        cy0=cy0,
        image_w=image_w,
        image_h=image_h,
    )
    resolved_batch_shape = (
        tuple(image_w.shape) if batch_shape is None else tuple(batch_shape)
    )

    R_t = _broadcast_camera_time(
        R_w2c,
        batch_shape=resolved_batch_shape,
        seq_len=seq_len,
        trailing_dims=2,
        name="R_w2c",
    )
    C_t = _broadcast_camera_time(
        C_w,
        batch_shape=resolved_batch_shape,
        seq_len=seq_len,
        trailing_dims=1,
        name="C_w",
    )
    fx_t = _broadcast_camera_time(
        fx,
        batch_shape=resolved_batch_shape,
        seq_len=seq_len,
        trailing_dims=0,
        name="fx",
    )
    fy_t = _broadcast_camera_time(
        fy,
        batch_shape=resolved_batch_shape,
        seq_len=seq_len,
        trailing_dims=0,
        name="fy",
    )
    cx0_t = _broadcast_camera_time(
        cx0,
        batch_shape=resolved_batch_shape,
        seq_len=seq_len,
        trailing_dims=0,
        name="cx0",
    )
    cy0_t = _broadcast_camera_time(
        cy0,
        batch_shape=resolved_batch_shape,
        seq_len=seq_len,
        trailing_dims=0,
        name="cy0",
    )
    w_t = _broadcast_camera_time(
        image_w,
        batch_shape=resolved_batch_shape,
        seq_len=seq_len,
        trailing_dims=0,
        name="image_w",
    )
    h_t = _broadcast_camera_time(
        image_h,
        batch_shape=resolved_batch_shape,
        seq_len=seq_len,
        trailing_dims=0,
        name="image_h",
    )

    r6d = R_t[..., :2, :].reshape(*R_t.shape[:-2], 6)
    c_norm = normalize_court_position(C_t)

    cam_dist = torch.linalg.vector_norm(C_t, dim=-1).clamp(min=0.1)
    log_dist = torch.log(cam_dist / 10.0)
    sin_elev = C_t[..., 2] / cam_dist
    geom = torch.stack([log_dist, sin_elev], dim=-1)

    diag = torch.hypot(w_t, h_t)
    intr = torch.stack(
        [
            torch.log(fx_t / diag),
            torch.log(fy_t / diag),
            (cx0_t - 0.5 * w_t) / w_t,
            (cy0_t - 0.5 * h_t) / h_t,
        ],
        dim=-1,
    )

    eye = torch.eye(3, device=R_t.device, dtype=R_t.dtype).expand_as(R_t[..., :1, :, :])
    if seq_len == 1:
        rel_R = eye
    else:
        step_R = R_t[..., 1:, :, :] @ R_t[..., :-1, :, :].transpose(-1, -2)
        rel_R = torch.cat([eye, step_R], dim=-3)
    rel_r6d = rel_R[..., :2, :].reshape(*rel_R.shape[:-2], 6)

    features = torch.cat([r6d, c_norm, geom, intr, rel_r6d], dim=-1)
    if features.shape[-1] != CAMERA_FEATURE_DIM:
        raise RuntimeError(
            f"Expected {CAMERA_FEATURE_DIM} camera features, got {features.shape[-1]}."
        )
    return features


def perturb_camera_calibration(
    R_w2c: Tensor,
    C_w: Tensor,
    *,
    fx: Tensor,
    fy: Tensor,
    cx0: Tensor,
    cy0: Tensor,
    image_w: Tensor,
    image_h: Tensor,
    config: CalibrationNoiseConfig = _DEFAULT_CALIB_NOISE_CONFIG,
    generator: torch.Generator | None = None,
) -> PerturbedCameraParams:
    """Perturb calibrated camera inputs to simulate real ``court_calibration`` PnP errors."""
    validate_pinhole_camera_tensors(
        R_w2c,
        C_w,
        fx=fx,
        fy=fy,
        cx0=cx0,
        cy0=cy0,
        image_w=image_w,
        image_h=image_h,
    )
    if not config.enabled or config.prob <= 0.0:
        return PerturbedCameraParams(
            R_w2c=R_w2c.clone(),
            C_w=C_w.clone(),
            fx=fx.clone(),
            fy=fy.clone(),
            cx0=cx0.clone(),
            cy0=cy0.clone(),
        )

    batch_shape = R_w2c.shape[:-2]
    device = R_w2c.device
    dtype = R_w2c.dtype

    active = (
        torch.rand(batch_shape, device=device, dtype=dtype, generator=generator)
        < config.prob
    )
    if not bool(active.any()):
        return PerturbedCameraParams(
            R_w2c=R_w2c.clone(),
            C_w=C_w.clone(),
            fx=fx.clone(),
            fy=fy.clone(),
            cx0=cx0.clone(),
            cy0=cy0.clone(),
        )

    rot_rad_std = math.radians(config.rotation_deg_std)
    omega = (
        _randn((*batch_shape, 3), device=device, dtype=dtype, generator=generator)
        * rot_rad_std
    )
    delta_R = so3_exp_map(omega)
    R_pert = delta_R @ R_w2c

    delta_C = (
        _randn((*batch_shape, 3), device=device, dtype=dtype, generator=generator)
        * config.translation_m_std
    )
    C_pert = C_w + delta_C
    # Preserve positive camera height when original camera is above the court plane.
    C_pert_z = torch.where(C_w[..., 2] > 0.2, C_pert[..., 2].clamp(min=0.1), C_pert[..., 2])
    C_pert = torch.cat([C_pert[..., :2], C_pert_z.unsqueeze(-1)], dim=-1)

    focal_log_noise = (
        _randn(batch_shape, device=device, dtype=dtype, generator=generator)
        * config.focal_rel_std
    )
    focal_scale = torch.exp(focal_log_noise.clamp(min=-0.35, max=0.35))
    fx_pert = fx * focal_scale
    fy_pert = fy * focal_scale

    cx0_pert = (
        cx0
        + _randn(batch_shape, device=device, dtype=dtype, generator=generator)
        * config.principal_point_rel_std
        * image_w
    )
    cy0_pert = (
        cy0
        + _randn(batch_shape, device=device, dtype=dtype, generator=generator)
        * config.principal_point_rel_std
        * image_h
    )

    return PerturbedCameraParams(
        R_w2c=torch.where(active.unsqueeze(-1).unsqueeze(-1), R_pert, R_w2c),
        C_w=torch.where(active.unsqueeze(-1), C_pert, C_w),
        fx=torch.where(active, fx_pert, fx),
        fy=torch.where(active, fy_pert, fy),
        cx0=torch.where(active, cx0_pert, cx0),
        cy0=torch.where(active, cy0_pert, cy0),
    )


def _cliff_to_camera_translation_core(
    pred_cam: Tensor,
    bbx_xys: Tensor,
    *,
    fx_t: Tensor,
    fy_t: Tensor,
    cx0_t: Tensor,
    cy0_t: Tensor,
) -> Tensor:
    """Computation-only CLIFF ``[s, tx, ty]`` + ``bbx_xys`` -> camera translation."""
    s = pred_cam[..., 0]
    tx = pred_cam[..., 1]
    ty = pred_cam[..., 2]
    box_s = bbx_xys[..., 2]
    f_eff = torch.sqrt(fx_t * fy_t)
    tz = 2.0 * f_eff / (s * box_s)
    cam_x = tx + tz * (bbx_xys[..., 0] - cx0_t) / fx_t
    cam_y = ty + tz * (bbx_xys[..., 1] - cy0_t) / fy_t
    return torch.stack([cam_x, cam_y, tz], dim=-1)


def cliff_to_camera_translation(
    pred_cam: Tensor,
    bbx_xys: Tensor,
    *,
    fx: Tensor,
    fy: Tensor,
    cx0: Tensor,
    cy0: Tensor,
) -> Tensor:
    """Convert CLIFF weak-perspective ``[s, tx, ty]`` + ``bbx_xys`` to metric camera ``[X_c, Y_c, Z_c]``."""
    if pred_cam.ndim < 2 or pred_cam.shape != bbx_xys.shape or pred_cam.shape[-1] != 3:
        raise ValueError("pred_cam and bbx_xys must share shape (..., T, 3).")
    if not torch.isfinite(pred_cam).all() or not torch.isfinite(bbx_xys).all():
        raise ValueError("pred_cam and bbx_xys must contain only finite values.")
    _validate_intrinsics_tensors(fx=fx, fy=fy, cx0=cx0, cy0=cy0)

    batch_shape = tuple(pred_cam.shape[:-2])
    seq_len = int(pred_cam.shape[-2])
    geom_dtype = _geometry_dtype(pred_cam, bbx_xys, fx, fy, cx0, cy0)
    with _fp32_geometry_context(pred_cam.device):
        pred_cam_fp = pred_cam.to(dtype=geom_dtype)
        bbx_fp = bbx_xys.to(dtype=geom_dtype)
        fx_t = _broadcast_camera_time(
            fx.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=0,
            name="fx",
        )
        fy_t = _broadcast_camera_time(
            fy.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=0,
            name="fy",
        )
        cx0_t = _broadcast_camera_time(
            cx0.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=0,
            name="cx0",
        )
        cy0_t = _broadcast_camera_time(
            cy0.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=0,
            name="cy0",
        )

        s = pred_cam_fp[..., 0]
        box_s = bbx_fp[..., 2]
        if bool((s <= 0).any()) or bool((box_s <= 0).any()):
            raise ValueError(
                "cliff_to_camera_translation requires strictly positive scale s and bbx_xys[..., 2]."
            )

        return _cliff_to_camera_translation_core(
            pred_cam_fp,
            bbx_fp,
            fx_t=fx_t,
            fy_t=fy_t,
            cx0_t=cx0_t,
            cy0_t=cy0_t,
        )


def camera_translation_to_cliff(
    cam_t: Tensor,
    bbx_xys: Tensor,
    *,
    fx: Tensor,
    fy: Tensor,
    cx0: Tensor,
    cy0: Tensor,
) -> Tensor:
    """Exact analytical inverse of :func:`cliff_to_camera_translation`."""
    if cam_t.ndim < 2 or cam_t.shape != bbx_xys.shape or cam_t.shape[-1] != 3:
        raise ValueError("cam_t and bbx_xys must share shape (..., T, 3).")
    if not torch.isfinite(cam_t).all() or not torch.isfinite(bbx_xys).all():
        raise ValueError("cam_t and bbx_xys must contain only finite values.")
    _validate_intrinsics_tensors(fx=fx, fy=fy, cx0=cx0, cy0=cy0)

    batch_shape = tuple(cam_t.shape[:-2])
    seq_len = int(cam_t.shape[-2])
    geom_dtype = _geometry_dtype(cam_t, bbx_xys, fx, fy, cx0, cy0)
    with _fp32_geometry_context(cam_t.device):
        cam_t_fp = cam_t.to(dtype=geom_dtype)
        bbx_fp = bbx_xys.to(dtype=geom_dtype)
        fx_t = _broadcast_camera_time(
            fx.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=0,
            name="fx",
        )
        fy_t = _broadcast_camera_time(
            fy.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=0,
            name="fy",
        )
        cx0_t = _broadcast_camera_time(
            cx0.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=0,
            name="cx0",
        )
        cy0_t = _broadcast_camera_time(
            cy0.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=0,
            name="cy0",
        )

        tz = cam_t_fp[..., 2]
        box_s = bbx_fp[..., 2]
        if bool((tz <= _MIN_DEPTH_M).any()) or bool((box_s <= 0).any()):
            raise ValueError(
                "camera_translation_to_cliff requires positive depth Z_c and positive bbx_xys[..., 2]."
            )

        f_eff = torch.sqrt(fx_t * fy_t)
        s = 2.0 * f_eff / (tz * box_s)
        tx = cam_t_fp[..., 0] - tz * (bbx_fp[..., 0] - cx0_t) / fx_t
        ty = cam_t_fp[..., 1] - tz * (bbx_fp[..., 1] - cy0_t) / fy_t
        return torch.stack([s, tx, ty], dim=-1)


def _center_local_pose_on_hips_core(local_pose_cam: Tensor) -> Tensor:
    """Computation-only hip centering: subtract ``0.5 * (left_hip + right_hip)``."""
    pelvis = 0.5 * (
        local_pose_cam[..., LEFT_HIP_IDX, :] + local_pose_cam[..., RIGHT_HIP_IDX, :]
    )
    return local_pose_cam - pelvis.unsqueeze(-2)


def center_local_pose_on_hips(local_pose_cam: Tensor) -> Tensor:
    """Subtract the mid-hip offset so ``0.5 * (left_hip + right_hip) == 0``."""
    if local_pose_cam.ndim < 2 or local_pose_cam.shape[-2:] != (NUM_HUMAN_KP, 3):
        raise ValueError(
            f"local_pose_cam must have shape (..., 17, 3), got {tuple(local_pose_cam.shape)}."
        )
    return _center_local_pose_on_hips_core(local_pose_cam)


def _cliff_and_local_pose_to_world_coco17_core(
    pred_cam: Tensor,
    local_pose_cam: Tensor,
    bbx_xys: Tensor,
    *,
    R_t: Tensor,
    C_t: Tensor,
    fx_t: Tensor,
    fy_t: Tensor,
    cx0_t: Tensor,
    cy0_t: Tensor,
) -> Tensor:
    """Computation-only CLIFF + hip-centered local pose -> court 3D COCO-17 [m]."""
    root_cam = _cliff_to_camera_translation_core(
        pred_cam,
        bbx_xys,
        fx_t=fx_t,
        fy_t=fy_t,
        cx0_t=cx0_t,
        cy0_t=cy0_t,
    )
    centered_local = _center_local_pose_on_hips_core(local_pose_cam)
    kp_cam = root_cam.unsqueeze(-2) + centered_local
    # Since X_c = (X_w - C_w) @ R_w2c^T, the inverse is X_w = X_c @ R_w2c + C_w.
    return torch.einsum("...ji,...kj->...ki", R_t, kp_cam) + C_t.unsqueeze(-2)


def cliff_and_local_pose_to_world_coco17(
    pred_cam: Tensor,
    local_pose_cam: Tensor,
    bbx_xys: Tensor,
    *,
    R_w2c: Tensor,
    C_w: Tensor,
    fx: Tensor,
    fy: Tensor,
    cx0: Tensor,
    cy0: Tensor,
) -> Tensor:
    """Transform CLIFF ``pred_cam`` + hip-centered ``local_pose_cam`` to court 3D COCO-17 [m]."""
    if pred_cam.ndim < 2 or pred_cam.shape[-1] != 3:
        raise ValueError(
            f"pred_cam must have shape (..., T, 3), got {tuple(pred_cam.shape)}."
        )
    if local_pose_cam.shape != (*pred_cam.shape[:-1], NUM_HUMAN_KP, 3):
        raise ValueError(
            "local_pose_cam must have shape (..., T, 17, 3) matching pred_cam."
        )
    if not torch.isfinite(local_pose_cam).all():
        raise ValueError("local_pose_cam must contain only finite values.")
    _validate_extrinsics_and_intrinsics(
        R_w2c, C_w, fx=fx, fy=fy, cx0=cx0, cy0=cy0
    )

    batch_shape = tuple(pred_cam.shape[:-2])
    seq_len = int(pred_cam.shape[-2])
    geom_dtype = _geometry_dtype(
        pred_cam, local_pose_cam, bbx_xys, R_w2c, C_w, fx, fy, cx0, cy0
    )
    with _fp32_geometry_context(pred_cam.device):
        R_t = _broadcast_camera_time(
            R_w2c.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=2,
            name="R_w2c",
        )
        C_t = _broadcast_camera_time(
            C_w.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=1,
            name="C_w",
        )

        root_cam = cliff_to_camera_translation(
            pred_cam.to(dtype=geom_dtype),
            bbx_xys.to(dtype=geom_dtype),
            fx=fx.to(dtype=geom_dtype),
            fy=fy.to(dtype=geom_dtype),
            cx0=cx0.to(dtype=geom_dtype),
            cy0=cy0.to(dtype=geom_dtype),
        )
        centered_local = center_local_pose_on_hips(local_pose_cam.to(dtype=geom_dtype))
        kp_cam = root_cam.unsqueeze(-2) + centered_local
        # Since X_c = (X_w - C_w) @ R_w2c^T, the inverse is X_w = X_c @ R_w2c + C_w.
        return torch.einsum("...ji,...kj->...ki", R_t, kp_cam) + C_t.unsqueeze(-2)


def world_coco17_to_cliff_and_local_pose(
    world_kp_3d: Tensor,
    bbx_xys: Tensor,
    *,
    R_w2c: Tensor,
    C_w: Tensor,
    fx: Tensor,
    fy: Tensor,
    cx0: Tensor,
    cy0: Tensor,
) -> tuple[Tensor, Tensor]:
    """Exact inverse of :func:`cliff_and_local_pose_to_world_coco17`."""
    if world_kp_3d.ndim < 3 or world_kp_3d.shape[-2:] != (NUM_HUMAN_KP, 3):
        raise ValueError(
            f"world_kp_3d must have shape (..., T, 17, 3), got {tuple(world_kp_3d.shape)}."
        )
    if not torch.isfinite(world_kp_3d).all():
        raise ValueError("world_kp_3d must contain only finite values.")
    _validate_extrinsics_and_intrinsics(
        R_w2c, C_w, fx=fx, fy=fy, cx0=cx0, cy0=cy0
    )

    batch_shape = tuple(world_kp_3d.shape[:-3])
    seq_len = int(world_kp_3d.shape[-3])
    geom_dtype = _geometry_dtype(world_kp_3d, bbx_xys, R_w2c, C_w, fx, fy, cx0, cy0)
    with _fp32_geometry_context(world_kp_3d.device):
        R_t = _broadcast_camera_time(
            R_w2c.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=2,
            name="R_w2c",
        )
        C_t = _broadcast_camera_time(
            C_w.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=1,
            name="C_w",
        )

        rel_world = world_kp_3d.to(dtype=geom_dtype) - C_t.unsqueeze(-2)
        kp_cam = torch.einsum("...ij,...kj->...ki", R_t, rel_world)
        root_cam = 0.5 * (
            kp_cam[..., LEFT_HIP_IDX, :] + kp_cam[..., RIGHT_HIP_IDX, :]
        )
        local_pose_cam = kp_cam - root_cam.unsqueeze(-2)
        pred_cam = camera_translation_to_cliff(
            root_cam,
            bbx_xys.to(dtype=geom_dtype),
            fx=fx.to(dtype=geom_dtype),
            fy=fy.to(dtype=geom_dtype),
            cx0=cx0.to(dtype=geom_dtype),
            cy0=cy0.to(dtype=geom_dtype),
        )
        return pred_cam, local_pose_cam


def project_world_coco17_to_pixel(
    world_kp_3d: Tensor,
    *,
    R_w2c: Tensor,
    C_w: Tensor,
    fx: Tensor,
    fy: Tensor,
    cx0: Tensor,
    cy0: Tensor,
) -> tuple[Tensor, Tensor]:
    """Differentiably project court-space ``(..., T, J, 3)`` [m] to pixel ``(u, v)``."""
    if world_kp_3d.ndim < 3 or world_kp_3d.shape[-1] != 3:
        raise ValueError(
            f"world_kp_3d must have shape (..., T, J, 3), got {tuple(world_kp_3d.shape)}."
        )
    if not torch.isfinite(world_kp_3d).all():
        raise ValueError("world_kp_3d must contain only finite values.")
    _validate_extrinsics_and_intrinsics(
        R_w2c, C_w, fx=fx, fy=fy, cx0=cx0, cy0=cy0
    )

    batch_shape = tuple(world_kp_3d.shape[:-3])
    seq_len = int(world_kp_3d.shape[-3])
    geom_dtype = _geometry_dtype(world_kp_3d, R_w2c, C_w, fx, fy, cx0, cy0)
    with _fp32_geometry_context(world_kp_3d.device):
        R_t = _broadcast_camera_time(
            R_w2c.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=2,
            name="R_w2c",
        )
        C_t = _broadcast_camera_time(
            C_w.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=1,
            name="C_w",
        )
        fx_t = _broadcast_camera_time(
            fx.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=0,
            name="fx",
        ).unsqueeze(-1)
        fy_t = _broadcast_camera_time(
            fy.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=0,
            name="fy",
        ).unsqueeze(-1)
        cx0_t = _broadcast_camera_time(
            cx0.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=0,
            name="cx0",
        ).unsqueeze(-1)
        cy0_t = _broadcast_camera_time(
            cy0.to(dtype=geom_dtype),
            batch_shape=batch_shape,
            seq_len=seq_len,
            trailing_dims=0,
            name="cy0",
        ).unsqueeze(-1)

        kp_cam = torch.einsum(
            "...ij,...kj->...ki", R_t, world_kp_3d.to(dtype=geom_dtype) - C_t.unsqueeze(-2)
        )
        z = kp_cam[..., 2]
        in_front = z > _MIN_DEPTH_M
        z_safe = torch.where(in_front, z, torch.ones_like(z))
        u_px = fx_t * (kp_cam[..., 0] / z_safe) + cx0_t
        v_px = fy_t * (kp_cam[..., 1] / z_safe) + cy0_t
        uv_px = torch.stack([u_px, v_px], dim=-1)
        uv_px = torch.where(in_front.unsqueeze(-1), uv_px, torch.zeros_like(uv_px))
        return uv_px, in_front


__all__ = [
    "BBX_XYS_DIM",
    "BBoxAugmentationConfig",
    "CAMERA_FEATURE_DIM",
    "CLIFFCAM_DIM",
    "CalibrationNoiseConfig",
    "LEFT_HIP_IDX",
    "PerturbedCameraParams",
    "RIGHT_HIP_IDX",
    "SyntheticBBoxConfig",
    "augment_bbx_xys",
    "camera_translation_to_cliff",
    "center_local_pose_on_hips",
    "cliff_and_local_pose_to_world_coco17",
    "cliff_to_camera_translation",
    "compute_camera_features",
    "compute_cliffcam_features",
    "compute_synthetic_bbx_xys",
    "normalize_kp2d_bbox",
    "perturb_camera_calibration",
    "project_world_coco17_to_pixel",
    "so3_exp_map",
    "validate_pinhole_camera_tensors",
    "world_coco17_to_cliff_and_local_pose",
]

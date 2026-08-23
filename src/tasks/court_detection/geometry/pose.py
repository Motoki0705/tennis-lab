"""Strict target-court camera pose10D contracts for Court Detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.synthetic_data_generation.dataset.court.components.camera_view import (
    camera_view_canonicalization,
)
from src.synthetic_data_generation.scene_contract import CourtInstance
from src.tasks.court_detection.data.contracts import CourtPoseAuthority
from src.utils.geometry.rotation_conversions import rotation_6d_to_matrix
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d

POSE10D_SCHEMA = "pose10d_camera_to_canonical_row6d_logf_v1"
POSE10D_RAW_ORDER: tuple[str, ...] = (
    "tx",
    "ty",
    "tz",
    "a11",
    "a12",
    "a13",
    "a21",
    "a22",
    "a23",
    "logf",
)
ROTATION_DEGENERACY_EPS = 1.0e-6
SO3_ATOL = 1.0e-5
INTRINSICS_ATOL = 1.0e-6
PROJECTION_ATOL_PX = 1.0e-4


def _require_finite(value: Tensor, *, name: str) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values.")


def validate_square_intrinsics(
    intrinsics: Tensor,
    *,
    atol: float = INTRINSICS_ATOL,
) -> None:
    """Require a finite zero-skew pinhole K with one scalar pixel focal."""
    if intrinsics.shape != (3, 3):
        raise ValueError("Court pose intrinsics must have shape (3,3).")
    _require_finite(intrinsics, name="Court pose intrinsics")
    if float(intrinsics[0, 0]) <= 0.0 or float(intrinsics[1, 1]) <= 0.0:
        raise ValueError("Court pose focal lengths must be positive.")
    if not math.isclose(
        float(intrinsics[0, 0]),
        float(intrinsics[1, 1]),
        rel_tol=0.0,
        abs_tol=atol,
    ):
        raise ValueError("Court pose intrinsics require fx=fy within 1e-6 pixels.")
    zero_entries = intrinsics.new_tensor(
        (intrinsics[0, 1], intrinsics[1, 0], intrinsics[2, 0], intrinsics[2, 1])
    )
    if bool(torch.any(torch.abs(zero_entries) > atol)):
        raise ValueError("Court pose intrinsics require zero skew/off-axis terms.")
    if not math.isclose(
        float(intrinsics[2, 2]), 1.0, rel_tol=0.0, abs_tol=atol
    ):
        raise ValueError("Court pose intrinsics require K[2,2]=1.")


def validate_proper_rotation(rotation: Tensor, *, atol: float = SO3_ATOL) -> None:
    """Require finite proper SO(3) within the frozen numeric tolerance."""
    if rotation.ndim < 2 or rotation.shape[-2:] != (3, 3):
        raise ValueError("Court pose rotation must end in shape (3,3).")
    _require_finite(rotation, name="Court pose rotation")
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    gram = rotation.transpose(-1, -2) @ rotation
    if not bool(torch.allclose(gram, identity.expand_as(gram), atol=atol, rtol=0.0)):
        raise ValueError("Court pose rotation must be orthonormal within 1e-5.")
    determinant = torch.linalg.det(rotation)
    if not bool(
        torch.allclose(
            determinant,
            torch.ones_like(determinant),
            atol=atol,
            rtol=0.0,
        )
    ):
        raise ValueError("Court pose rotation must have determinant +1 within 1e-5.")


@dataclass(frozen=True, slots=True)
class CourtPoseTarget:
    """One augmentation-aware target in canonical court metres and pixels."""

    translation_m: Tensor  # [3], canonical camera centre
    rotation: Tensor  # [3,3], canonical_from_camera
    log_focal: Tensor  # scalar, augmented pixels
    intrinsics: Tensor  # [3,3], augmented pixel K
    semantic_to_physical: Tensor  # [14], exact V3 authority order

    def __post_init__(self) -> None:
        if self.translation_m.shape != (3,):
            raise ValueError("Court pose translation target must have shape (3,).")
        _require_finite(self.translation_m, name="Court pose translation target")
        validate_proper_rotation(self.rotation)
        if self.log_focal.shape != ():
            raise ValueError("Court pose log-focal target must be scalar.")
        _require_finite(self.log_focal, name="Court pose log-focal target")
        validate_square_intrinsics(self.intrinsics)
        if self.semantic_to_physical.shape != (14,) or self.semantic_to_physical.dtype != torch.long:
            raise ValueError("Court pose semantic_to_physical must be int64 [14].")
        if sorted(self.semantic_to_physical.tolist()) != list(range(14)):
            raise ValueError("Court pose semantic_to_physical must be a 0..13 bijection.")
        expected_log_focal = torch.log(self.intrinsics[0, 0])
        if not bool(torch.allclose(self.log_focal, expected_log_focal, atol=1.0e-6, rtol=0.0)):
            raise ValueError("Court pose log-focal disagrees with augmented K.")

    @property
    def raw_values(self) -> Tensor:
        """Return the exact immutable ten-scalar training target order."""
        return torch.cat(
            (
                self.translation_m,
                self.rotation[:2].reshape(6),
                self.log_focal.reshape(1),
            )
        )

    def to_mapping(self) -> dict[str, Tensor]:
        return {
            "translation_m": self.translation_m,
            "rotation": self.rotation,
            "log_focal": self.log_focal,
            "intrinsics": self.intrinsics,
            "semantic_to_physical": self.semantic_to_physical,
            "raw_pose10d": self.raw_values,
        }


@dataclass(frozen=True, slots=True)
class CourtDecodedPose:
    """Decoded batched query prediction, separate from the raw 10D head."""

    translation_m: Tensor  # [B,3]
    rotation: Tensor  # [B,3,3], canonical_from_camera
    focal_px: Tensor  # [B]
    log_focal: Tensor  # [B]

    def __post_init__(self) -> None:
        if self.translation_m.ndim != 2 or self.translation_m.shape[-1] != 3:
            raise ValueError("Decoded Court translation must have shape (B,3).")
        batch_size = self.translation_m.shape[0]
        if self.rotation.shape != (batch_size, 3, 3):
            raise ValueError("Decoded Court rotation must have shape (B,3,3).")
        if self.focal_px.shape != (batch_size,) or self.log_focal.shape != (batch_size,):
            raise ValueError("Decoded Court focal values must have shape (B,).")
        _require_finite(self.translation_m, name="Decoded Court translation")
        validate_proper_rotation(self.rotation)
        _require_finite(self.log_focal, name="Decoded Court log focal")
        _require_finite(self.focal_px, name="Decoded Court focal")
        if bool(torch.any(self.focal_px <= 0.0)):
            raise ValueError("Decoded Court focal must be positive.")


def decode_pose10d_strict(values: Tensor) -> CourtDecodedPose:
    """Predecode-reject invalid rows, then recover a differentiable proper SO(3)."""
    if values.ndim != 2 or values.shape[1] != len(POSE10D_RAW_ORDER):
        raise ValueError("Raw Court pose must have exact shape (B,10).")
    if values.shape[0] <= 0:
        raise ValueError("Raw Court pose batch size must be positive.")
    _require_finite(values, name="Raw Court pose10d")
    row6d = values[:, 3:9]
    first = row6d[:, :3]
    second = row6d[:, 3:]
    first_norm = torch.linalg.vector_norm(first, dim=-1)
    if bool(torch.any(first_norm < ROTATION_DEGENERACY_EPS)):
        raise ValueError("Raw Court pose first-row norm is below 1e-6.")
    normalized_first = first / first_norm.unsqueeze(-1)
    residual = second - (normalized_first * second).sum(dim=-1, keepdim=True) * normalized_first
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    if bool(torch.any(residual_norm < ROTATION_DEGENERACY_EPS)):
        raise ValueError("Raw Court pose second-row residual norm is below 1e-6.")
    rotation = rotation_6d_to_matrix(row6d)
    validate_proper_rotation(rotation)
    focal = torch.exp(values[:, 9])
    if not bool(torch.isfinite(focal).all()) or bool(torch.any(focal <= 0.0)):
        raise ValueError("Decoded Court focal must be finite and positive.")
    return CourtDecodedPose(
        translation_m=values[:, :3],
        rotation=rotation,
        focal_px=focal,
        log_focal=values[:, 9],
    )


def build_pose_target(
    authority: CourtPoseAuthority,
    *,
    source_to_output: Tensor | None = None,
) -> CourtPoseTarget:
    """Derive the unique V3 target from typed camera/court authority and K."""
    if not isinstance(authority, CourtPoseAuthority):
        raise TypeError("Court pose target requires CourtPoseAuthority.")
    binding = authority.target_court
    court = CourtInstance(
        court_instance_id=binding.court_instance_id,
        candidate_id=binding.candidate_id,
        scene_from_court=binding.scene_from_court,
        court_from_scene=binding.scene_from_court.inverse(),
        fit_status="accepted",
        fit_metrics={},
        holdout_status="accepted",
        holdout_metrics={},
    )
    canonical = camera_view_canonicalization(authority.camera, court)
    intrinsic = torch.tensor(
        authority.camera.intrinsics,
        dtype=torch.float64,
    ).reshape(3, 3)
    validate_square_intrinsics(intrinsic)
    if source_to_output is not None:
        if source_to_output.shape != (3, 3):
            raise ValueError("Court source-to-output calibration must be [3,3].")
        _require_finite(source_to_output, name="Court source-to-output calibration")
        intrinsic = source_to_output.to(dtype=torch.float64) @ intrinsic
        validate_square_intrinsics(intrinsic)
    rotation = torch.from_numpy(
        canonical.camera_from_canonical.matrix()[:3, :3].T.copy()
    )
    translation = torch.tensor(
        canonical.camera_center_canonical_m,
        dtype=torch.float64,
    )
    return CourtPoseTarget(
        translation_m=translation.float(),
        rotation=rotation.float(),
        log_focal=torch.log(intrinsic[0, 0]).float(),
        intrinsics=intrinsic.float(),
        semantic_to_physical=torch.tensor(
            canonical.semantic_to_physical,
            dtype=torch.long,
        ),
    )


def canonical_semantic_court_points(target: CourtPoseTarget) -> Tensor:
    """Return canonical KP14 in the exact semantic order carried by V3."""
    physical = court_keypoints_3d(STANDARD_COURT_CONFIG)[:14].to(dtype=torch.float64)
    ordered = physical.index_select(0, target.semantic_to_physical)
    # Identity or Rz(pi) is uniquely determined by the V3 physical permutation.
    if target.semantic_to_physical.tolist() == list(range(14)):
        transform = torch.eye(3, dtype=torch.float64)
    else:
        transform = torch.diag(torch.tensor((-1.0, -1.0, 1.0), dtype=torch.float64))
    return ordered @ transform.T


def project_canonical_points(target: CourtPoseTarget, points: Tensor) -> Tensor:
    """Project canonical metres with target C, R and augmentation-aware K."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Canonical Court projection points must have shape (N,3).")
    _require_finite(points, name="Canonical Court projection points")
    points64 = points.to(dtype=torch.float64)
    center = target.translation_m.to(dtype=torch.float64)
    rotation_camera_from_canonical = target.rotation.to(dtype=torch.float64).T
    points_camera = (points64 - center) @ rotation_camera_from_canonical.T
    if bool(torch.any(points_camera[:, 2] <= 0.0)):
        raise ValueError("Canonical Court projection requires positive camera depth.")
    homogeneous = points_camera @ target.intrinsics.to(dtype=torch.float64).T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    _require_finite(pixels, name="Canonical Court projected pixels")
    return pixels


def validate_projection_round_trip(
    target: CourtPoseTarget,
    expected_semantic_uv: Tensor,
    *,
    atol_px: float = PROJECTION_ATOL_PX,
) -> None:
    """Require pose/K to reproduce V3 semantic KP14 within 1e-4 px."""
    if expected_semantic_uv.shape != (14, 2):
        raise ValueError("V3 projection round-trip expects semantic UV [14,2].")
    _require_finite(expected_semantic_uv, name="V3 semantic UV")
    projected = project_canonical_points(
        target,
        canonical_semantic_court_points(target),
    )
    if not bool(
        torch.allclose(
            projected,
            expected_semantic_uv.to(dtype=torch.float64),
            atol=atol_px,
            rtol=0.0,
        )
    ):
        max_error = float(
            torch.max(torch.abs(projected - expected_semantic_uv.to(dtype=torch.float64)))
        )
        raise ValueError(
            "Synthetic Court V3 pose/K projection round-trip exceeds 1e-4 px; "
            f"max_error_px={max_error:.6g}."
        )


__all__ = [
    "INTRINSICS_ATOL",
    "POSE10D_RAW_ORDER",
    "POSE10D_SCHEMA",
    "PROJECTION_ATOL_PX",
    "ROTATION_DEGENERACY_EPS",
    "SO3_ATOL",
    "CourtDecodedPose",
    "CourtPoseTarget",
    "build_pose_target",
    "canonical_semantic_court_points",
    "decode_pose10d_strict",
    "project_canonical_points",
    "validate_projection_round_trip",
    "validate_proper_rotation",
    "validate_square_intrinsics",
]

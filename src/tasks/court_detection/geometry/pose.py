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
from src.utils.schema.court import (
    CAMERA_VIEW_HALF_TURN_INDEX,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

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
PROJECTIVE_DEPTH_EPS_M = 1.0e-6
MIN_PROJECTION_REFERENCE_POINTS = 4
PROJECTION_REFERENCE_AREA_EPS_M2 = 1.0e-6


def _pose_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promote reduced-precision predictions to the SO(3) compute authority."""
    if dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


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
    compute_dtype = _pose_compute_dtype(rotation.dtype)
    with torch.autocast(device_type=rotation.device.type, enabled=False):
        authority = rotation.to(dtype=compute_dtype)
        identity = torch.eye(3, dtype=compute_dtype, device=rotation.device)
        # CUDA float32 matmul may use the process-wide TF32 policy.  A reduced-
        # precision Gram matrix can reject an SO(3) result that was recovered
        # accurately in float32, so keep this strict check on elementwise
        # float32 arithmetic without changing the global matmul policy.
        gram = (authority.unsqueeze(-1) * authority.unsqueeze(-2)).sum(dim=-3)
        if not bool(
            torch.allclose(
                gram,
                identity.expand_as(gram),
                atol=atol,
                rtol=0.0,
            )
        ):
            raise ValueError("Court pose rotation must be orthonormal within 1e-5.")
        determinant = torch.linalg.det(authority)
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


@dataclass(frozen=True, slots=True)
class CourtPredictedProjection:
    """Differentiable predicted-pose projection in the model pixel frame."""

    points_xy: Tensor  # [B,14,2], model pixels
    depth_m: Tensor  # [B,14], unmodified signed camera-space depth

    def __post_init__(self) -> None:
        if self.points_xy.ndim != 3 or self.points_xy.shape[1:] != (14, 2):
            raise ValueError("Predicted Court projection points must have shape (B,14,2).")
        if self.depth_m.shape != self.points_xy.shape[:2]:
            raise ValueError("Predicted Court projection depth must have shape (B,14).")
        _require_finite(self.points_xy, name="Predicted Court projected pixels")
        _require_finite(self.depth_m, name="Predicted Court projective depth")


def decode_pose10d_strict(values: Tensor) -> CourtDecodedPose:
    """Decode a differentiable SO(3) at float32 or the higher input precision."""
    if values.ndim != 2 or values.shape[1] != len(POSE10D_RAW_ORDER):
        raise ValueError("Raw Court pose must have exact shape (B,10).")
    if values.shape[0] <= 0:
        raise ValueError("Raw Court pose batch size must be positive.")
    _require_finite(values, name="Raw Court pose10d")
    compute_dtype = _pose_compute_dtype(values.dtype)
    with torch.autocast(device_type=values.device.type, enabled=False):
        authority = values.to(dtype=compute_dtype)
        row6d = authority[:, 3:9]
        first = row6d[:, :3]
        second = row6d[:, 3:]
        first_norm = torch.linalg.vector_norm(first, dim=-1)
        if bool(torch.any(first_norm < ROTATION_DEGENERACY_EPS)):
            raise ValueError("Raw Court pose first-row norm is below 1e-6.")
        normalized_first = first / first_norm.unsqueeze(-1)
        residual = second - (
            (normalized_first * second).sum(dim=-1, keepdim=True)
            * normalized_first
        )
        residual_norm = torch.linalg.vector_norm(residual, dim=-1)
        if bool(torch.any(residual_norm < ROTATION_DEGENERACY_EPS)):
            raise ValueError("Raw Court pose second-row residual norm is below 1e-6.")
        rotation = rotation_6d_to_matrix(row6d)
        validate_proper_rotation(rotation)
        focal = torch.exp(authority[:, 9])
        if not bool(torch.isfinite(focal).all()) or bool(torch.any(focal <= 0.0)):
            raise ValueError("Decoded Court focal must be finite and positive.")
    return CourtDecodedPose(
        translation_m=authority[:, :3],
        rotation=rotation,
        focal_px=focal,
        log_focal=authority[:, 9],
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
        translation_m=translation,
        rotation=rotation,
        log_focal=torch.log(intrinsic[0, 0]),
        intrinsics=intrinsic,
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


def canonical_semantic_court_points_batched(
    semantic_to_physical: Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return V3 canonical KP14 for each exact camera-end semantic order.

    The two accepted orders are the identity and the shared full half-turn.
    This function deliberately consumes only the V3 ordering authority; it
    does not accept or infer a camera pose.
    """
    if (
        semantic_to_physical.ndim != 2
        or semantic_to_physical.shape[1] != 14
        or semantic_to_physical.dtype != torch.long
    ):
        raise ValueError("Court semantic_to_physical must be int64 (B,14).")
    if semantic_to_physical.shape[0] <= 0:
        raise ValueError("Court semantic_to_physical batch size must be positive.")
    output_device = semantic_to_physical.device if device is None else torch.device(device)
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("Canonical Court points require a floating-point dtype.")
    orders = semantic_to_physical.to(device=output_device)
    identity = torch.arange(14, dtype=torch.long, device=output_device)
    half_turn = torch.tensor(
        CAMERA_VIEW_HALF_TURN_INDEX,
        dtype=torch.long,
        device=output_device,
    )
    identity_rows = torch.all(orders == identity, dim=1)
    half_turn_rows = torch.all(orders == half_turn, dim=1)
    if not bool(torch.all(identity_rows | half_turn_rows)):
        raise ValueError(
            "V3 Court semantic_to_physical must be identity or the shared full half-turn."
        )
    physical = court_keypoints_3d(STANDARD_COURT_CONFIG)[:14].to(
        dtype=dtype,
        device=output_device,
    )
    ordered = physical[orders]
    half_turn_sign = ordered.new_tensor((-1.0, -1.0, 1.0))
    signs = torch.where(
        half_turn_rows[:, None],
        half_turn_sign[None, :],
        torch.ones_like(half_turn_sign)[None, :],
    )
    return ordered * signs[:, None, :]


def project_predicted_canonical_points(
    prediction: CourtDecodedPose,
    canonical_points: Tensor,
    principal_point_px: Tensor,
    *,
    depth_epsilon_m: float = PROJECTIVE_DEPTH_EPS_M,
) -> CourtPredictedProjection:
    """Project batched canonical points through a decoded predicted pose.

    Unlike :func:`project_canonical_points`, this prediction-side path keeps
    the autograd graph while using float32 or the higher decoded precision,
    and it does not reject low or negative depth. Near-zero depths use a
    signed finite denominator; the returned depths remain unmodified so the
    loss can apply fixed-visibility cheirality supervision.
    """
    if canonical_points.ndim != 3 or canonical_points.shape[1:] != (14, 3):
        raise ValueError("Predicted Court canonical points must have shape (B,14,3).")
    batch_size = canonical_points.shape[0]
    if batch_size != prediction.translation_m.shape[0]:
        raise ValueError(
            "Predicted Court canonical points must match the pose batch size."
        )
    if principal_point_px.shape != (batch_size, 2):
        raise ValueError("Predicted Court principal point must have shape (B,2).")
    _require_finite(canonical_points, name="Predicted Court canonical points")
    _require_finite(principal_point_px, name="Predicted Court principal point")
    if not math.isfinite(depth_epsilon_m) or depth_epsilon_m <= 0.0:
        raise ValueError("depth_epsilon_m must be finite and positive.")

    dtype = _pose_compute_dtype(prediction.translation_m.dtype)
    device = prediction.translation_m.device
    with torch.autocast(device_type=device.type, enabled=False):
        points = canonical_points.to(device=device, dtype=dtype)
        principal_point = principal_point_px.to(device=device, dtype=dtype)
        translation = prediction.translation_m.to(dtype=dtype)
        rotation = prediction.rotation.to(dtype=dtype)
        focal = prediction.focal_px.to(dtype=dtype)
        offset = points - translation[:, None, :]
        points_camera = torch.bmm(offset, rotation)
        depth = points_camera[:, :, 2]
        epsilon = float(depth_epsilon_m)
        safe_depth = torch.where(
            depth < 0.0,
            torch.clamp(depth, max=-epsilon),
            torch.clamp(depth, min=epsilon),
        )
        normalized_xy = points_camera[:, :, :2] / safe_depth[:, :, None]
        points_xy = (
            focal[:, None, None] * normalized_xy
            + principal_point[:, None, :]
        )
    return CourtPredictedProjection(points_xy=points_xy, depth_m=depth)


def project_canonical_points(target: CourtPoseTarget, points: Tensor) -> Tensor:
    """Project canonical metres at finite, non-degenerate projective depths."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Canonical Court projection points must have shape (N,3).")
    _require_finite(points, name="Canonical Court projection points")
    points64 = points.to(dtype=torch.float64)
    center = target.translation_m.to(dtype=torch.float64)
    rotation_camera_from_canonical = target.rotation.to(dtype=torch.float64).T
    points_camera = (points64 - center) @ rotation_camera_from_canonical.T
    _require_finite(points_camera, name="Canonical Court camera-space points")
    depth = points_camera[:, 2]
    if bool(torch.any(torch.abs(depth) <= PROJECTIVE_DEPTH_EPS_M)):
        raise ValueError(
            "Canonical Court projection has zero/near-zero projective depth "
            "within 1e-6 m."
        )
    homogeneous = points_camera @ target.intrinsics.to(dtype=torch.float64).T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    _require_finite(pixels, name="Canonical Court projected pixels")
    return pixels


def _valid_projection_reference_mask(
    target: CourtPoseTarget,
    points: Tensor,
) -> Tensor:
    """Select the deterministic positive-depth subset used as pose/K evidence."""
    points64 = points.to(dtype=torch.float64)
    center = target.translation_m.to(dtype=torch.float64)
    points_camera = (points64 - center) @ target.rotation.to(dtype=torch.float64)
    _require_finite(points_camera, name="Canonical Court camera-space points")
    return points_camera[:, 2] > PROJECTIVE_DEPTH_EPS_M


def _validate_projection_reference_evidence(
    points: Tensor,
    reference_mask: Tensor,
) -> None:
    """Require four positive-depth references spanning non-collinear court XY."""
    reference_count = int(reference_mask.sum())
    if reference_count < MIN_PROJECTION_REFERENCE_POINTS:
        raise ValueError(
            "Synthetic Court V3 projection round-trip requires at least "
            f"{MIN_PROJECTION_REFERENCE_POINTS} positive-depth references; "
            f"got {reference_count}."
        )
    reference_xy = points.to(dtype=torch.float64)[reference_mask, :2]
    offsets = reference_xy[1:] - reference_xy[0]
    twice_triangle_area = torch.abs(
        offsets[:, None, 0] * offsets[None, :, 1]
        - offsets[:, None, 1] * offsets[None, :, 0]
    )
    if float(torch.max(twice_triangle_area)) <= PROJECTION_REFERENCE_AREA_EPS_M2:
        raise ValueError(
            "Synthetic Court V3 projection round-trip positive-depth references "
            "must contain non-collinear canonical-court evidence."
        )


def _projection_comparison_tolerance(
    expected: Tensor,
    *,
    atol_px: float,
) -> Tensor:
    """Include only the unavoidable half-ULP uncertainty of serialized UV."""
    positive_infinity = torch.full_like(expected, float("inf"))
    negative_infinity = torch.full_like(expected, float("-inf"))
    upper_spacing = torch.nextafter(expected, positive_infinity) - expected
    lower_spacing = expected - torch.nextafter(expected, negative_infinity)
    quantization_radius = 0.5 * torch.maximum(upper_spacing, lower_spacing)
    return quantization_radius.to(dtype=torch.float64) + atol_px


def validate_projection_round_trip(
    target: CourtPoseTarget,
    expected_semantic_uv: Tensor,
    *,
    atol_px: float = PROJECTION_ATOL_PX,
) -> None:
    """Require pose/K to reproduce V3 semantic KP14 within 1e-4 px."""
    if not math.isfinite(atol_px) or atol_px < 0.0:
        raise ValueError(
            "Projection round-trip atol_px must be finite and non-negative."
        )
    if expected_semantic_uv.shape != (14, 2):
        raise ValueError("V3 projection round-trip expects semantic UV [14,2].")
    _require_finite(expected_semantic_uv, name="V3 semantic UV")
    canonical_points = canonical_semantic_court_points(target)
    projected = project_canonical_points(target, canonical_points)
    reference_mask = _valid_projection_reference_mask(target, canonical_points)
    _validate_projection_reference_evidence(canonical_points, reference_mask)
    projected_references = projected[reference_mask]
    expected_references_native = expected_semantic_uv[reference_mask]
    expected_references = expected_references_native.to(dtype=torch.float64)
    absolute_error = torch.abs(projected_references - expected_references)
    tolerance = _projection_comparison_tolerance(
        expected_references_native,
        atol_px=atol_px,
    )
    if not bool(torch.all(absolute_error <= tolerance)):
        max_error = float(torch.max(absolute_error))
        raise ValueError(
            "Synthetic Court V3 pose/K projection round-trip exceeds 1e-4 px; "
            f"max_error_px={max_error:.6g}."
        )


__all__ = [
    "INTRINSICS_ATOL",
    "MIN_PROJECTION_REFERENCE_POINTS",
    "POSE10D_RAW_ORDER",
    "POSE10D_SCHEMA",
    "PROJECTION_ATOL_PX",
    "PROJECTIVE_DEPTH_EPS_M",
    "PROJECTION_REFERENCE_AREA_EPS_M2",
    "ROTATION_DEGENERACY_EPS",
    "SO3_ATOL",
    "CourtDecodedPose",
    "CourtPredictedProjection",
    "CourtPoseTarget",
    "build_pose_target",
    "canonical_semantic_court_points",
    "canonical_semantic_court_points_batched",
    "decode_pose10d_strict",
    "project_canonical_points",
    "project_predicted_canonical_points",
    "validate_projection_round_trip",
    "validate_proper_rotation",
    "validate_square_intrinsics",
]

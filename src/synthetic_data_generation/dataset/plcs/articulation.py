"""Fail-closed streaming articulation witnesses for production PLCS motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from smplx.lbs import batch_rodrigues  # type: ignore[import-untyped]

from src.synthetic_data_generation.dataset.plcs.smplh import (
    SMPLHDeviceClip,
    SMPLHGaussianBatch,
)
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import (
    MotionCategory,
    PLCSMotionClip,
)

_REGION_JOINTS: dict[str, tuple[int, ...]] = {
    "legs": (1, 2, 4, 5, 7, 8, 10, 11),
    "arms": (13, 14, 16, 17, 18, 19, 20, 21),
    "torso": (3, 6, 9, 12, 15),
}


@dataclass(frozen=True, slots=True)
class MotionArticulationReport:
    """Quantitative root-removed motion evidence from bounded CUDA probes."""

    frame_count: int
    category: MotionCategory
    non_root_pose_range_radians: float
    gaussian_nonrigid_residual_m: float
    region_displacement_m: dict[str, float]
    deformed_frame_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the machine-readable production gate result."""
        return {
            "frame_count": self.frame_count,
            "category": self.category.value,
            "non_root_pose_range_radians": self.non_root_pose_range_radians,
            "gaussian_nonrigid_residual_m": self.gaussian_nonrigid_residual_m,
            "region_displacement_m": dict(self.region_displacement_m),
            "deformed_frame_indices": list(self.deformed_frame_indices),
        }


def articulation_probe_indices(clip: PLCSMotionClip) -> tuple[int, ...]:
    """Select deterministic witnesses without retaining full deformed geometry."""
    if clip.frame_count < 2:
        raise ValueError("Production articulated motion requires at least two frames.")
    body = clip.body_pose_axis_angle.astype(np.float64, copy=False).reshape(
        clip.frame_count, 21, 3
    )
    indices = {0, clip.frame_count - 1}
    local = np.concatenate(
        (
            clip.body_pose_axis_angle,
            clip.left_hand_pose_axis_angle,
            clip.right_hand_pose_axis_angle,
        ),
        axis=1,
    ).astype(np.float64, copy=False)
    indices.add(int(np.argmax(np.linalg.norm(local - local[:1], axis=1))))
    for joints in _REGION_JOINTS.values():
        local_joint_indices = tuple(index - 1 for index in joints if 1 <= index <= 21)
        region = body[:, local_joint_indices, :]
        indices.add(int(np.argmax(np.linalg.norm(region - region[:1], axis=(1, 2)))))
    return tuple(sorted(indices))


def validate_articulated_motion(
    clip: PLCSMotionClip,
    device_clip: SMPLHDeviceClip,
    probes: SMPLHGaussianBatch,
    *,
    minimum_local_displacement_m: float = 1.0e-4,
    minimum_pose_range_radians: float = 1.0e-4,
) -> MotionArticulationReport:
    """Reject a fixed-pose rigid transform using bounded on-device witnesses."""
    expected_probes = articulation_probe_indices(clip)
    if probes.source_frame_indices != expected_probes:
        raise ValueError(
            "Articulation probe batch differs from deterministic witnesses."
        )
    if (
        device_clip.source_path != clip.source_path
        or device_clip.frame_count != clip.frame_count
    ):
        raise ValueError(
            "Articulation device clip differs from its lossless source clip."
        )
    displacement_threshold = _positive(
        minimum_local_displacement_m, name="minimum_local_displacement_m"
    )
    pose_threshold = _positive(
        minimum_pose_range_radians, name="minimum_pose_range_radians"
    )
    local_pose = np.concatenate(
        (
            clip.body_pose_axis_angle,
            clip.left_hand_pose_axis_angle,
            clip.right_hand_pose_axis_angle,
        ),
        axis=1,
    ).astype(np.float64, copy=False)
    non_root_pose_range = float(np.max(np.ptp(local_pose, axis=0)))
    if non_root_pose_range <= pose_threshold:
        raise ValueError(
            "PLCS motion is a fixed non-root pose and can only produce rigid motion."
        )

    indices = torch.tensor(
        expected_probes, dtype=torch.int64, device=probes.means_m.device
    )
    orientations = device_clip.full_pose_axis_angle.index_select(0, indices)[:, :3]
    root_rotations = batch_rodrigues(orientations)
    centered_joints = probes.joints_m - probes.joints_m[:, :1, :]
    root_local_joints = centered_joints @ root_rotations
    region_displacement = {
        name: float(
            torch.linalg.vector_norm(
                root_local_joints[:, joint_indices, :]
                - root_local_joints[:1, joint_indices, :],
                dim=-1,
            )
            .max()
            .item()
        )
        for name, joint_indices in _REGION_JOINTS.items()
    }
    if clip.category in {MotionCategory.RUNNING, MotionCategory.WALKING}:
        insufficient = {
            name: value
            for name, value in region_displacement.items()
            if value <= displacement_threshold
        }
        if insufficient:
            raise ValueError(
                f"{clip.category.value} motion lacks required local region movement: "
                f"{insufficient}."
            )

    reference = probes.means_m[0]
    residuals = tuple(
        _rigid_fit_residual(reference, probes.means_m[index])
        for index in range(1, len(expected_probes))
    )
    deformed_indices = tuple(
        frame_index
        for frame_index, residual in zip(expected_probes[1:], residuals, strict=True)
        if residual > displacement_threshold
    )
    if not deformed_indices:
        raise ValueError(
            "PLCS Gaussian avatar is rigid-only after removing its best-fit root transform."
        )
    return MotionArticulationReport(
        frame_count=clip.frame_count,
        category=clip.category,
        non_root_pose_range_radians=non_root_pose_range,
        gaussian_nonrigid_residual_m=max(residuals),
        region_displacement_m=region_displacement,
        deformed_frame_indices=deformed_indices,
    )


def _rigid_fit_residual(source: torch.Tensor, target: torch.Tensor) -> float:
    source_center = source.mean(dim=0)
    target_center = target.mean(dim=0)
    centered_source = source - source_center
    centered_target = target - target_center
    left, _, right_transpose = torch.linalg.svd(centered_source.T @ centered_target)
    rotation = right_transpose.T @ left.T
    if float(torch.linalg.det(rotation).item()) < 0.0:
        right_transpose = right_transpose.clone()
        right_transpose[-1] *= -1.0
        rotation = right_transpose.T @ left.T
    aligned = centered_source @ rotation.T + target_center
    return float(torch.linalg.vector_norm(aligned - target, dim=-1).max().item())


def _positive(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


__all__ = [
    "MotionArticulationReport",
    "articulation_probe_indices",
    "validate_articulated_motion",
]

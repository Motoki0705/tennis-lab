"""Tests for bounded CUDA root-removed articulation witnesses."""

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.plcs.articulation import (
    articulation_probe_indices,
    validate_articulated_motion,
)
from src.synthetic_data_generation.dataset.plcs.smplh import (
    SMPLHDeviceClip,
    SMPLHGaussianBatch,
)
from src.tasks.plcs.generate_dataset.sampling.motion_source import PLCSMotionClip

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _clip(tmp_path: Path, *, category: str, fixed: bool = False) -> PLCSMotionClip:
    poses: NDArray[np.float32] = np.zeros((3, 156), dtype=np.float32)
    if not fixed:
        poses[1, 3:66] = 0.2
        poses[2, 3:66] = 0.4
    return PLCSMotionClip.from_amass_arrays(
        source_path=tmp_path / "motion.npz",
        category=category,
        gender="neutral",
        fps=30.0,
        poses=poses,
        trans=np.zeros((3, 3), dtype=np.float32),
        betas=np.zeros(16, dtype=np.float32),
    )


def _device_clip(clip: PLCSMotionClip) -> SMPLHDeviceClip:
    return SMPLHDeviceClip(
        source_path=clip.source_path,
        full_pose_axis_angle=torch.as_tensor(
            np.array(clip.full_pose_axis_angle(), copy=True),
            device="cuda:0",
        ),
        betas=torch.as_tensor(np.array(clip.betas, copy=True), device="cuda:0"),
    )


def _probes(clip: PLCSMotionClip, *, rigid_only: bool) -> SMPLHGaussianBatch:
    indices = articulation_probe_indices(clip)
    count = len(indices)
    reference = torch.tensor(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=torch.float32,
        device="cuda:0",
    )
    means = reference.unsqueeze(0).repeat(count, 1, 1)
    if not rigid_only and count > 1:
        means[1:, 1, 0] += 0.25
    joints = torch.zeros((count, 52, 3), dtype=torch.float32, device="cuda:0")
    for region in (
        (1, 2, 4, 5, 7, 8, 10, 11),
        (13, 14, 16, 17, 18, 19, 20, 21),
        (3, 6, 9, 12, 15),
    ):
        joints[1:, region, 1] += 0.1
    quaternions = torch.zeros((count, 4, 4), dtype=torch.float32, device="cuda:0")
    quaternions[..., 0] = 1.0
    return SMPLHGaussianBatch(
        source_frame_indices=indices,
        means_m=means,
        quaternions_wxyz=quaternions,
        log_scales_m=torch.zeros((count, 4, 3), dtype=torch.float32, device="cuda:0"),
        joints_m=joints,
    )


def test_running_requires_legs_arms_torso_and_nonrigid_gaussians(
    tmp_path: Path,
) -> None:
    clip = _clip(tmp_path, category="running")
    report = validate_articulated_motion(
        clip,
        _device_clip(clip),
        _probes(clip, rigid_only=False),
    )

    assert set(report.region_displacement_m) == {"legs", "arms", "torso"}
    assert all(value > 0.0 for value in report.region_displacement_m.values())
    assert report.gaussian_nonrigid_residual_m > 0.0


def test_fixed_pose_rigid_motion_is_rejected(tmp_path: Path) -> None:
    clip = _clip(tmp_path, category="general", fixed=True)
    with pytest.raises(ValueError, match="fixed non-root pose"):
        validate_articulated_motion(
            clip,
            _device_clip(clip),
            _probes(clip, rigid_only=True),
        )

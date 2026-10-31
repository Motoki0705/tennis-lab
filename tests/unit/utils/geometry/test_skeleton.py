"""Independent COCO-17 numerical oracles on geometrically defined inputs.

Validity and gradients follow the explicit missing-feature contract.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from src.utils.geometry.skeleton import (
    MIN_BONE_LENGTH_M,
    MIN_PLANE_SINE,
    SkeletonValues,
    compute_bone_lengths,
    compute_joint_angles,
    compute_torsion_angles,
    compute_torso_twist,
)
from tests.support.geometry.skeleton import coco_pose_m


def _transformed_poses(pose: Tensor) -> Tensor:
    # Proper rotation: +X->+Y, +Y->+Z, +Z->+X (determinant +1).
    rotation = pose.new_tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    translation = pose.new_tensor([7, -11, 2])
    rigid = pose @ rotation.T + translation
    # Two leading axes also expose accidental reductions across batch/time.
    return torch.stack(
        [torch.stack([pose, rigid]), torch.stack([2.5 * pose, 2.5 * rigid])]
    )


@pytest.mark.parametrize(
    ("compute", "expected", "length_scale"),
    [
        pytest.param(
            compute_joint_angles,
            # Left elbow: cos(angle) = -3/5; all other joints are right angles.
            [math.acos(-3 / 5)] + [math.pi / 2] * 11,
            False,
            id="joint_angles",
        ),
        pytest.param(
            compute_torsion_angles,
            # Viewed along each shoulder->elbow or hip->knee axis, the plane
            # pairs have quarter turns of opposite sign, or coincide (left leg).
            [math.pi / 2, -math.pi / 2, 0.0, -math.pi / 2],
            False,
            id="torsion_angles",
        ),
        pytest.param(compute_torso_twist, 0.0, False, id="torso_twist"),
        pytest.param(
            compute_bone_lengths,
            # Widths, torso sides, arm bones, leg bones. The left forearm is
            # the hypotenuse of the explicit 3-4-5 triangle in the YZ plane.
            [3.0, 3.0, 4.0, 4.0, 3.0, 5.0, 4.0, 6.0, 3.0, 4.0, 5.0, 12.0],
            True,
            id="bone_lengths",
        ),
    ],
)
def test_known_coco_values_under_rigid_motion_and_positive_scale(
    compute: Callable[[Tensor], SkeletonValues],
    expected: list[float] | float,
    length_scale: bool,
) -> None:
    pose = coco_pose_m()
    known = pose.new_tensor(expected)
    expected_values = torch.stack(
        [torch.stack([known, known]), torch.stack([known, known])]
    )
    if length_scale:
        expected_values[1] *= 2.5
    result = compute(_transformed_poses(pose))
    assert result.valid.all()
    torch.testing.assert_close(result.values, expected_values, atol=1e-12, rtol=0)


def test_joint_angles_at_zero_right_and_straight_angles() -> None:
    # The bones have nonzero length even at the two interior-angle endpoints.
    pose = torch.tensor(
        [
            [[2, 0, 0], [0, 0, 0], [1, 0, 0]],
            [[2, 0, 0], [0, 0, 0], [0, 3, 0]],
            [[2, 0, 0], [0, 0, 0], [-3, 0, 0]],
        ],
        dtype=torch.float64,
    )
    actual = compute_joint_angles(pose, triplets=((0, 1, 2),))
    torch.testing.assert_close(
        actual.values,
        pose.new_tensor([[0], [math.pi / 2], [math.pi]]),
        atol=1e-12,
        rtol=0,
    )

    assert actual.valid.all()


@pytest.mark.parametrize(
    "twist", [math.pi / 2, -math.pi / 2, math.pi / 6, math.pi - 1e-6, -math.pi + 1e-6]
)
def test_torso_twist_has_known_sign_and_magnitude(twist: float) -> None:
    pose = coco_pose_m()
    # Hip axis is +X, torso axis +Z. Rotate just the shoulder line about +Z.
    shoulder_half_axis = pose.new_tensor([math.cos(twist), math.sin(twist), 0])
    center = pose.new_tensor([1.5, 0, 4])
    pose[5] = center - shoulder_half_axis
    pose[6] = center + shoulder_half_axis
    pose.requires_grad_()
    result = compute_torso_twist(_transformed_poses(pose))
    assert result.valid.all()
    torch.testing.assert_close(
        result.values,
        pose.new_full((2, 2), twist),
        atol=1e-12,
        rtol=0,
    )
    result.values.sum().backward()
    assert pose.grad is not None and torch.isfinite(pose.grad).all()
    assert pose.grad.abs().max() > 0


COMPUTATIONS = [
    compute_joint_angles,
    compute_torsion_angles,
    compute_torso_twist,
    compute_bone_lengths,
]


@pytest.mark.parametrize("compute", COMPUTATIONS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mixed_batch_preserves_valid_values_and_gradients(
    compute: Callable[[Tensor], SkeletonValues],
    dtype: torch.dtype,
) -> None:
    single = coco_pose_m().to(dtype).requires_grad_()
    expected = compute(single)
    expected.values.sum().backward()
    mixed = torch.stack([single.detach(), torch.zeros_like(single)]).requires_grad_()
    actual = compute(mixed)
    assert actual.valid[0].all()
    assert not actual.valid[1].any()
    assert actual.values.shape == actual.valid.shape
    assert actual.valid.dtype == torch.bool
    torch.testing.assert_close(actual.values[0], expected.values)
    actual.values.sum().backward()
    assert mixed.grad is not None and single.grad is not None
    torch.testing.assert_close(mixed.grad[0], single.grad)
    torch.testing.assert_close(mixed.grad[1], torch.zeros_like(single))


@pytest.mark.parametrize(
    ("compute", "invalid_indices"),
    [
        (compute_joint_angles, [0, 4]),
        (compute_torsion_angles, [0]),
        (compute_bone_lengths, [4]),
    ],
)
def test_validity_is_per_feature_and_invalid_feature_has_zero_gradient(
    compute: Callable[[Tensor], SkeletonValues], invalid_indices: list[int]
) -> None:
    pose = coco_pose_m()
    pose[7] = pose[5]  # Only the left upper arm collapses.
    pose.requires_grad_()
    result = compute(pose)
    expected_mask = torch.ones_like(result.valid)
    expected_mask[invalid_indices] = False
    torch.testing.assert_close(result.valid, expected_mask)
    result.values[invalid_indices].sum().backward()
    assert pose.grad is not None
    torch.testing.assert_close(pose.grad, torch.zeros_like(pose))


@pytest.mark.parametrize("reason", ["hip", "shoulder", "torso", "projection"])
def test_twist_requires_nonzero_axes_and_transverse_planes(reason: str) -> None:
    pose = coco_pose_m()
    if reason == "hip":
        pose[12] = pose[11]
    elif reason == "shoulder":
        pose[6] = pose[5]
    elif reason == "torso":
        pose[5:7, 2] = 0
    else:
        pose[11] = pose.new_tensor([1.5, 0, -1])
        pose[12] = pose.new_tensor([1.5, 0, 1])
    pose.requires_grad_()
    result = compute_torso_twist(pose)
    assert not result.valid
    result.values.backward()
    assert pose.grad is not None
    torch.testing.assert_close(pose.grad, torch.zeros_like(pose))


def test_straight_limb_has_valid_interior_angle_but_invalid_torsion() -> None:
    pose = coco_pose_m()
    pose[9] = 2 * pose[7] - pose[5]
    interior = compute_joint_angles(pose)
    torsion = compute_torsion_angles(pose)
    assert interior.valid.all()
    assert interior.values[0] == pytest.approx(math.pi)
    torch.testing.assert_close(torsion.valid, torch.tensor([False, True, True, True]))


@pytest.mark.parametrize("angle", [math.pi - 1e-6, -math.pi + 1e-6])
def test_torsion_near_signed_pi_has_defined_value_and_finite_gradient(
    angle: float,
) -> None:
    pose = torch.tensor(
        [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, math.cos(angle), -math.sin(angle)]],
        dtype=torch.float64,
        requires_grad=True,
    )
    result = compute_torsion_angles(pose, quadruplets=((0, 1, 2, 3),))
    assert result.valid.all()
    torch.testing.assert_close(
        result.values, pose.new_tensor([angle]), atol=1e-12, rtol=0
    )
    result.values.sum().backward()
    assert pose.grad is not None and torch.isfinite(pose.grad).all()
    assert pose.grad.abs().max() > 0


def test_length_threshold_is_in_metres_and_plane_threshold_is_scale_invariant() -> None:
    pose = coco_pose_m()
    pose[7] = pose[5] + pose.new_tensor([0, MIN_BONE_LENGTH_M / 2, 0])
    assert not compute_bone_lengths(pose).valid[4]
    assert compute_bone_lengths(4 * pose).valid[4]
    # Almost straight but nonzero bones: plane validity is independent of scale.
    pose = torch.tensor(
        [[0, 1, 0], [0, 0, 0], [1, 0, 0], [2, MIN_PLANE_SINE / 2, 0]],
        dtype=torch.float64,
    )
    for scale in (1.0, 100.0):
        assert not compute_torsion_angles(scale * pose, ((0, 1, 2, 3),)).valid[0]
    pose[3, 1] = 2 * MIN_PLANE_SINE
    assert compute_torsion_angles(pose, ((0, 1, 2, 3),)).valid[0]


@pytest.mark.parametrize("compute", COMPUTATIONS)
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_nonfinite_pose_is_rejected_at_entry(
    compute: Callable[[Tensor], SkeletonValues], nonfinite: float
) -> None:
    pose = coco_pose_m()
    pose[0, 0] = nonfinite  # Even an unused face joint must be finite.
    with pytest.raises(ValueError, match="finite"):
        compute(pose)

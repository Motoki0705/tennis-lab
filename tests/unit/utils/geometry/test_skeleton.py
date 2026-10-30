"""Independent COCO-17 numerical oracles on geometrically defined inputs.

Degenerate bones/planes and signed-angle branch cuts await a separate policy;
these tests do not prescribe their current incidental values or gradients.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from src.utils.geometry.skeleton import (
    compute_bone_lengths,
    compute_joint_angles,
    compute_torsion_angles,
    compute_torso_twist,
)


def _coco_pose() -> Tensor:
    # Explicit COCO body ordering, independent of the production edge tables:
    # shoulders 5/6, elbows 7/8, wrists 9/10, hips 11/12,
    # knees 13/14, ankles 15/16. Face points do not enter these APIs.
    return torch.tensor(
        [
            [1.5, 0.0, 6.0],
            [1.0, 0.0, 6.5],
            [2.0, 0.0, 6.5],
            [0.5, 0.0, 6.0],
            [2.5, 0.0, 6.0],
            [0.0, 0.0, 4.0],
            [3.0, 0.0, 4.0],
            [0.0, 3.0, 4.0],
            [3.0, -4.0, 4.0],
            [0.0, 6.0, 8.0],
            [3.0, -4.0, -2.0],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [3.0, 5.0, 0.0],
            [4.0, 3.0, 0.0],
            [3.0, 5.0, 12.0],
        ],
        dtype=torch.float64,
    )


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
    compute: Callable[[Tensor], Tensor],
    expected: list[float] | float,
    length_scale: bool,
) -> None:
    pose = _coco_pose()
    known = pose.new_tensor(expected)
    expected_values = torch.stack(
        [torch.stack([known, known]), torch.stack([known, known])]
    )
    if length_scale:
        expected_values[1] *= 2.5
    torch.testing.assert_close(
        compute(_transformed_poses(pose)), expected_values, atol=1e-12, rtol=0
    )


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
        actual, pose.new_tensor([[0], [math.pi / 2], [math.pi]]), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("twist", [math.pi / 2, -math.pi / 2, math.pi / 6])
def test_torso_twist_has_known_sign_and_magnitude(twist: float) -> None:
    pose = _coco_pose()
    # Hip axis is +X, torso axis +Z. Rotate just the shoulder line about +Z.
    shoulder_half_axis = pose.new_tensor([math.cos(twist), math.sin(twist), 0])
    center = pose.new_tensor([1.5, 0, 4])
    pose[5] = center - shoulder_half_axis
    pose[6] = center + shoulder_half_axis
    torch.testing.assert_close(
        compute_torso_twist(_transformed_poses(pose)),
        pose.new_full((2, 2), twist),
        atol=1e-12,
        rtol=0,
    )

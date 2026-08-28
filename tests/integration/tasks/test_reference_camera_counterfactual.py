"""Opposite-reference geometry and projection integration contracts."""

from __future__ import annotations

import torch
from torch import Tensor

from src.tasks.base.generate_dataset import (
    CourtReferenceFrameProvenance,
    build_court_view_record,
    build_reference_frame_provenance,
    camera_extrinsics_physical_to_target,
    court_headings_physical_to_target,
    court_points_physical_to_target,
    court_vectors_physical_to_target,
    court_world_joints_physical_to_target,
    resolve_court_keypoint_contract,
)
from src.utils.projection.camera_projector import Camera, project_points


def _reference_pair() -> tuple[
    CourtReferenceFrameProvenance, CourtReferenceFrameProvenance
]:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    views = (
        build_court_view_record(
            camera_id="camera_near",
            camera_center_court_m=(1.0, -12.0, 5.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="camera_far",
            camera_center_court_m=(-1.0, 12.0, 5.0),
            contract=contract,
        ),
    )
    return (
        build_reference_frame_provenance(
            views,
            reference_camera_id="camera_near",
        ),
        build_reference_frame_provenance(
            views,
            reference_camera_id="camera_far",
        ),
    )


def _assert_opposite_reference_contract(
    *, dtype: torch.dtype, atol: float, rtol: float
) -> None:
    first, second = _reference_pair()
    first_matrix = torch.tensor(first.reference_from_physical, dtype=dtype)
    second_matrix = torch.tensor(second.reference_from_physical, dtype=dtype)
    relative_rotation = second_matrix @ first_matrix.transpose(-1, -2)

    blcs_position = torch.tensor(
        [[[1.5, -2.0, 0.7], [-0.5, 4.0, 1.2]]], dtype=dtype
    )
    blcs_vector = torch.tensor(
        [[[0.2, 1.1, -0.3], [-1.5, 0.4, 0.8]]], dtype=dtype
    )
    plcs_position = torch.tensor(
        [[[2.0, -3.0, 0.0], [-1.0, 2.5, 0.0]]], dtype=dtype
    )
    plcs_heading = torch.tensor([[[0.6, 0.8], [-0.8, 0.6]]], dtype=dtype)
    world_joints = torch.tensor(
        [[[[2.0, -3.0, 1.0], [2.2, -2.8, 1.8]]]], dtype=dtype
    )

    quantities = (
        (
            court_points_physical_to_target(blcs_position, first),
            court_points_physical_to_target(blcs_position, second),
            relative_rotation,
        ),
        (
            court_vectors_physical_to_target(blcs_vector, first),
            court_vectors_physical_to_target(blcs_vector, second),
            relative_rotation,
        ),
        (
            court_points_physical_to_target(plcs_position, first),
            court_points_physical_to_target(plcs_position, second),
            relative_rotation,
        ),
        (
            court_headings_physical_to_target(plcs_heading, first),
            court_headings_physical_to_target(plcs_heading, second),
            relative_rotation[:2, :2],
        ),
        (
            court_world_joints_physical_to_target(world_joints, first),
            court_world_joints_physical_to_target(world_joints, second),
            relative_rotation,
        ),
    )
    for first_value, second_value, rotation in quantities:
        assert isinstance(first_value, Tensor)
        assert isinstance(second_value, Tensor)
        torch.testing.assert_close(
            second_value,
            first_value @ rotation.transpose(-1, -2),
            atol=atol,
            rtol=rtol,
        )

    object_uv = torch.tensor(
        [[[[0.2, 0.3], [0.7, 0.6]], [[0.4, 0.8], [0.1, 0.2]]]],
        dtype=dtype,
    )
    canonical_pose = torch.arange(18, dtype=dtype).reshape(1, 2, 3, 3)
    torch.testing.assert_close(object_uv.clone(), object_uv, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        canonical_pose.clone(), canonical_pose, atol=0.0, rtol=0.0
    )

    physical_points = torch.tensor(
        [[1.0, 2.0, 10.0], [-2.0, -1.0, 8.0]], dtype=dtype
    )
    camera_center = torch.zeros(3, dtype=dtype)
    camera_rotation = torch.eye(3, dtype=dtype)
    physical_camera = Camera(
        C=camera_center,
        R=camera_rotation,
        f=800.0,
        cx=640.0,
        cy=360.0,
        w=1280,
        h=720,
    )
    physical_uv, physical_visible = project_points(physical_camera, physical_points)
    assert physical_visible.all()
    for provenance in (first, second):
        transformed_points = court_points_physical_to_target(
            physical_points, provenance
        )
        transformed_center, transformed_rotation = (
            camera_extrinsics_physical_to_target(
                camera_center,
                camera_rotation,
                provenance,
            )
        )
        assert isinstance(transformed_points, Tensor)
        assert isinstance(transformed_center, Tensor)
        assert isinstance(transformed_rotation, Tensor)
        transformed_uv, transformed_visible = project_points(
            Camera(
                C=transformed_center,
                R=transformed_rotation,
                f=physical_camera.f,
                cx=physical_camera.cx,
                cy=physical_camera.cy,
                w=physical_camera.w,
                h=physical_camera.h,
            ),
            transformed_points,
        )
        assert torch.equal(transformed_visible, physical_visible)
        torch.testing.assert_close(
            transformed_uv,
            physical_uv,
            atol=atol,
            rtol=rtol,
        )


def test_pure_geometry_counterfactual_uses_float64_issue_tolerance() -> None:
    _assert_opposite_reference_contract(
        dtype=torch.float64,
        atol=1.0e-9,
        rtol=1.0e-9,
    )


def test_runtime_counterfactual_uses_float32_issue_tolerance() -> None:
    _assert_opposite_reference_contract(
        dtype=torch.float32,
        atol=1.0e-6,
        rtol=1.0e-5,
    )

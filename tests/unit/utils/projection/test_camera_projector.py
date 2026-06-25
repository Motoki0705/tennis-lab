"""Unit tests for :mod:`src.utils.projection.camera_projector`."""

from __future__ import annotations

import math

import pytest
import torch

from src.utils.projection.camera_projector import (
    Camera,
    CameraConfig,
    CameraProjector,
    CameraView,
    make_look_at_camera,
    project_points,
)


class TestMakeLookAtCamera:
    def test_rotation_is_orthonormal(self) -> None:
        cam = make_look_at_camera(center=(2.0, -3.0, 4.0), look_at=(0.0, 0.0, 0.0))
        rtr = cam.R @ cam.R.t()
        assert torch.allclose(rtr, torch.eye(3), atol=1e-5)

    def test_principal_point_is_image_center(self) -> None:
        cam = make_look_at_camera(center=(0.0, 0.0, 5.0), image_size=(1280, 720))
        assert cam.cx == pytest.approx(640.0)
        assert cam.cy == pytest.approx(360.0)
        assert cam.w == 1280 and cam.h == 720

    def test_focal_length_from_hfov(self) -> None:
        cam = make_look_at_camera(
            center=(0.0, 0.0, 5.0), image_size=(1000, 500), hfov_deg=90.0
        )
        expected_f = 0.5 * 1000 / math.tan(math.radians(45.0))
        assert cam.f == pytest.approx(expected_f, rel=1e-5)


class TestProjectPoints:
    def test_empty_input(self) -> None:
        cam = make_look_at_camera(center=(0.0, 0.0, 5.0))
        uv, mask = project_points(cam, torch.zeros(0, 3))
        assert uv.shape == (0, 2)
        assert mask.shape == (0,)

    def test_look_at_target_maps_to_principal_point(self) -> None:
        cam = make_look_at_camera(center=(0.0, 0.0, 5.0), look_at=(0.0, 0.0, 0.0))
        uv, mask = project_points(cam, torch.tensor([[0.0, 0.0, 0.0]]))
        assert bool(mask[0])
        assert uv[0, 0].item() == pytest.approx(cam.cx, abs=1e-3)
        assert uv[0, 1].item() == pytest.approx(cam.cy, abs=1e-3)

    def test_point_behind_camera_is_masked(self) -> None:
        cam = make_look_at_camera(center=(0.0, 0.0, 5.0), look_at=(0.0, 0.0, 0.0))
        # Camera looks down -z; a point above it (further +z) is behind.
        _, mask = project_points(cam, torch.tensor([[0.0, 0.0, 10.0]]))
        assert not bool(mask[0])


class TestCameraProjector:
    def test_project_points_to_uv_normalized_and_visible(self) -> None:
        projector = CameraProjector()
        cam = make_look_at_camera(center=(0.0, 0.0, 5.0), look_at=(0.0, 0.0, 0.0))
        uv, visible = projector.project_points_to_uv(
            torch.tensor([[0.0, 0.0, 0.0]]), cam
        )
        assert uv.shape == (1, 2)
        assert torch.allclose(uv[0], torch.tensor([0.5, 0.5]), atol=1e-3)
        assert bool(visible[0])

    def test_project_points_to_uv_empty(self) -> None:
        projector = CameraProjector()
        cam = make_look_at_camera(center=(0.0, 0.0, 5.0))
        uv, visible = projector.project_points_to_uv(torch.zeros(0, 3), cam)
        assert uv.shape == (0, 2)
        assert visible.shape == (0,)

    def test_project_points_preserves_leading_dims(self) -> None:
        projector = CameraProjector()
        cam = make_look_at_camera(center=(0.0, 0.0, 5.0))
        pts = torch.rand(4, 7, 3)
        uv, visible = projector.project_points_to_uv(pts, cam)
        assert uv.shape == (4, 7, 2)
        assert visible.shape == (4, 7)

    def test_project_court_keypoints_shape(self) -> None:
        projector = CameraProjector()
        cam = make_look_at_camera(center=(0.0, -15.0, 5.0), look_at=(0.0, 0.0, 0.0))
        uv, visible = projector.project_court_keypoints(cam)
        assert uv.shape == (20, 2)
        assert visible.shape == (20,)
        assert visible.dtype == torch.bool

    def test_generate_camera_view_serializes_params(self) -> None:
        projector = CameraProjector()
        cam = make_look_at_camera(center=(0.0, -15.0, 5.0), look_at=(0.0, 0.0, 0.0))
        view = projector.generate_camera_view(torch.rand(3, 3), cam)
        assert isinstance(view, CameraView)
        assert set(view.camera_params) == {"C", "R", "f", "cx", "cy", "w", "h"}
        assert view.points_uv is not None and view.points_uv.shape == (3, 2)

    def test_fixed_cameras_deterministic_without_noise(self) -> None:
        config = CameraConfig(
            fixed_position_noise_radius=0.0,
            fixed_look_at_xy_radius=0.0,
        )
        projector = CameraProjector(config=config)
        cams_a = projector.fixed_cameras()
        cams_b = projector.fixed_cameras()
        assert len(cams_a) == 6
        for a, b in zip(cams_a, cams_b, strict=True):
            assert torch.allclose(a.C, b.C)
            assert torch.allclose(a.R, b.R)


class TestCameraDataclass:
    def test_is_constructible(self) -> None:
        cam = Camera(
            C=torch.zeros(3), R=torch.eye(3), f=1.0, cx=0.5, cy=0.5, w=2, h=2
        )
        assert cam.f == 1.0

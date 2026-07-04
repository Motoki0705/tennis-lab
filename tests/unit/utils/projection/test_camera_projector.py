"""Unit tests for :mod:`src.utils.projection.camera_projector`."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from src.utils.projection.camera_projector import (
    BROADCAST_LAYOUT,
    FIXED_LAYOUT,
    Camera,
    CameraConfig,
    CameraProjector,
    CameraView,
    camera_config_from_mapping,
    make_look_at_camera,
    project_points,
)
from src.utils.schema.court import HALF_LENGTH


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


def _deterministic_broadcast_config(**overrides: object) -> CameraConfig:
    """Broadcast config with all per-scene jitter disabled."""
    params: dict[str, object] = dict(
        layout=BROADCAST_LAYOUT,
        broadcast_position_noise_radius=0.0,
        broadcast_look_at_xy_radius=0.0,
        broadcast_hfov_jitter_deg=0.0,
    )
    params.update(overrides)
    return CameraConfig(**params)  # type: ignore[arg-type]


class TestBroadcastCameras:
    def test_returns_two_cameras(self) -> None:
        projector = CameraProjector(config=_deterministic_broadcast_config())
        cams = projector.broadcast_cameras()
        assert len(cams) == 2

    def test_cameras_sit_behind_each_baseline(self) -> None:
        cfg = _deterministic_broadcast_config(broadcast_setback=20.0)
        projector = CameraProjector(config=cfg)
        near, far = projector.broadcast_cameras()
        expected_y = HALF_LENGTH + 20.0
        # Centred on the court, one behind each baseline (mirror in Y).
        assert near.C[0].item() == pytest.approx(0.0, abs=1e-5)
        assert far.C[0].item() == pytest.approx(0.0, abs=1e-5)
        assert near.C[1].item() == pytest.approx(-expected_y, abs=1e-4)
        assert far.C[1].item() == pytest.approx(+expected_y, abs=1e-4)
        # Behind the baselines, not merely on them.
        assert abs(near.C[1].item()) > HALF_LENGTH
        assert abs(far.C[1].item()) > HALF_LENGTH

    def test_mirror_pair_shares_height_and_intrinsics(self) -> None:
        projector = CameraProjector(config=_deterministic_broadcast_config())
        near, far = projector.broadcast_cameras()
        assert near.C[2].item() == pytest.approx(far.C[2].item(), abs=1e-6)
        assert near.f == pytest.approx(far.f, rel=1e-6)
        assert (near.w, near.h) == (far.w, far.h)

    def test_deterministic_without_noise(self) -> None:
        projector = CameraProjector(config=_deterministic_broadcast_config())
        a = projector.broadcast_cameras()
        b = projector.broadcast_cameras()
        for ca, cb in zip(a, b, strict=True):
            assert torch.allclose(ca.C, cb.C)
            assert torch.allclose(ca.R, cb.R)
            assert ca.f == pytest.approx(cb.f)

    def test_both_cameras_frame_the_full_court(self) -> None:
        projector = CameraProjector(config=_deterministic_broadcast_config())
        for cam in projector.broadcast_cameras():
            _, visible = projector.project_court_keypoints(cam)
            assert int(visible.sum()) == 20

    def test_is_telephoto_relative_to_default_wide_fov(self) -> None:
        # 35 deg broadcast FOV -> longer focal length than the 60 deg default.
        broadcast = CameraProjector(config=_deterministic_broadcast_config())
        default_cam = make_look_at_camera(
            center=(0.0, 0.0, 5.0), image_size=(1280, 720), hfov_deg=60.0
        )
        assert broadcast.broadcast_cameras()[0].f > default_cam.f

    def test_noise_produces_variation(self) -> None:
        cfg = CameraConfig(
            layout=BROADCAST_LAYOUT,
            broadcast_position_noise_radius=1.0,
            broadcast_look_at_xy_radius=1.0,
            broadcast_hfov_jitter_deg=2.0,
        )
        projector = CameraProjector(config=cfg)
        a = projector.broadcast_cameras()[0]
        b = projector.broadcast_cameras()[0]
        assert not torch.allclose(a.C, b.C)


class TestCamerasDispatch:
    def test_fixed_layout_returns_six(self) -> None:
        cfg = CameraConfig(
            layout=FIXED_LAYOUT,
            fixed_position_noise_radius=0.0,
            fixed_look_at_xy_radius=0.0,
        )
        assert len(CameraProjector(config=cfg).cameras()) == 6

    def test_broadcast_layout_returns_two(self) -> None:
        projector = CameraProjector(config=_deterministic_broadcast_config())
        assert len(projector.cameras()) == 2

    def test_unknown_layout_raises(self) -> None:
        projector = CameraProjector(config=CameraConfig(layout="satellite"))
        with pytest.raises(ValueError, match="Unknown camera layout"):
            projector.cameras()

    def test_default_layout_is_fixed(self) -> None:
        assert CameraConfig().layout == FIXED_LAYOUT
        assert len(CameraProjector().cameras()) == 6


class TestCameraConfigFromMapping:
    def test_maps_all_broadcast_fields(self) -> None:
        cfg = camera_config_from_mapping(
            {
                "layout": "broadcast",
                "image_size": [640, 360],
                "broadcast_setback": 18.0,
                "broadcast_height": 9.0,
                "broadcast_hfov_deg": 33.0,
                "broadcast_look_at_y": 1.0,
                "broadcast_look_at_height": 0.75,
                "broadcast_position_noise_radius": 0.5,
                "broadcast_look_at_xy_radius": 0.5,
                "broadcast_hfov_jitter_deg": 1.5,
            }
        )
        assert cfg.layout == "broadcast"
        assert cfg.image_size == (640, 360)
        assert cfg.broadcast_setback == 18.0
        assert cfg.broadcast_height == 9.0
        assert cfg.broadcast_hfov_deg == 33.0
        assert cfg.broadcast_look_at_height == 0.75
        assert cfg.broadcast_hfov_jitter_deg == 1.5

    def test_missing_keys_fall_back_to_defaults(self) -> None:
        defaults = CameraConfig()
        cfg = camera_config_from_mapping({"z_min": 4.0})
        assert cfg.z_min == 4.0
        # Everything unspecified keeps the dataclass default (fixed layout).
        assert cfg.layout == defaults.layout
        assert cfg.broadcast_setback == defaults.broadcast_setback
        assert cfg.image_size == defaults.image_size

    def test_empty_mapping_yields_defaults(self) -> None:
        assert camera_config_from_mapping({}) == CameraConfig()

    def test_accepts_omegaconf_dictconfig(self) -> None:
        from omegaconf import OmegaConf

        cfg = camera_config_from_mapping(
            OmegaConf.create({"layout": "broadcast", "broadcast_height": 6.5})
        )
        assert cfg.layout == "broadcast"
        assert cfg.broadcast_height == 6.5

    def test_non_mapping_raises(self) -> None:
        invalid_cfg: Any = 42
        with pytest.raises(TypeError, match="mapping-like"):
            camera_config_from_mapping(invalid_cfg)

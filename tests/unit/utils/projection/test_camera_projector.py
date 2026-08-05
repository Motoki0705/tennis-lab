"""Unit tests for :mod:`src.utils.projection.camera_projector`."""

from __future__ import annotations

import math

import pytest
import torch

from src.utils.projection.camera_projector import (
    BROADCAST_LAYOUT,
    FIXED_LAYOUT,
    Camera,
    CameraView,
    project_points,
)
from src.utils.projection.camera_projector import (
    CameraConfig as _CameraConfig,
)
from src.utils.projection.camera_projector import (
    CameraProjector as _CameraProjector,
)
from src.utils.projection.camera_projector import (
    make_look_at_camera as _make_look_at_camera,
)
from src.utils.schema.court import HALF_LENGTH, STANDARD_COURT_CONFIG


def make_look_at_camera(
    center: tuple[float, float, float],
    *,
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
    image_size: tuple[int, int] = (1280, 720),
    hfov_deg: float = 60.0,
) -> Camera:
    return _make_look_at_camera(
        center,
        look_at=look_at,
        image_size=image_size,
        hfov_deg=hfov_deg,
    )


def CameraConfig(**overrides: object) -> _CameraConfig:
    params: dict[str, object] = {
        "z_min": 3.0,
        "z_max": 5.0,
        "hfov_deg": 60.0,
        "image_size": (1280, 720),
        "fixed_look_at": (0.0, 0.0, 0.0),
        "fixed_baseline_clear_extra": 0.0,
        "fixed_position_noise_radius": 0.0,
        "fixed_look_at_xy_radius": 0.0,
        "layout": FIXED_LAYOUT,
        "broadcast_setback": 20.0,
        "broadcast_height": 7.0,
        "broadcast_hfov_deg": 35.0,
        "broadcast_look_at_y": 0.0,
        "broadcast_look_at_height": 0.5,
        "broadcast_position_noise_radius": 0.0,
        "broadcast_look_at_xy_radius": 0.0,
        "broadcast_hfov_jitter_deg": 0.0,
        "broadcast_setback_range": None,
        "broadcast_height_range": None,
        "broadcast_court_width_frac_range": None,
    }
    params.update(overrides)
    return _CameraConfig(**params)  # type: ignore[arg-type]


def CameraProjector(config: _CameraConfig | None = None) -> _CameraProjector:
    return _CameraProjector(
        config=config or CameraConfig(),
        court_config=STANDARD_COURT_CONFIG,
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


def _deterministic_broadcast_config(**overrides: object) -> _CameraConfig:
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


class TestProjectionChirality:
    """Regression tests for the mirrored-camera bug (sim-to-real collapse).

    A physical camera behind the near baseline (looking toward +y) must show
    world +x on the image right; the far-baseline camera shows it on the
    left. The pre-fix basis (x = up × z with a v-flip) produced left-right
    mirrored images, undetectable on the symmetric court geometry but fatal
    for keypoint-indexed models fed real footage.
    """

    def _corner_uv(self, cam: Camera, x: float, y: float) -> tuple[float, float]:
        uv, in_front = project_points(cam, torch.tensor([[x, y, 0.0]]))
        assert bool(in_front[0])
        return float(uv[0, 0]), float(uv[0, 1])

    def test_near_camera_shows_world_plus_x_on_image_right(self) -> None:
        projector = CameraProjector(config=_deterministic_broadcast_config())
        near, far = projector.broadcast_cameras()
        u_plus, _ = self._corner_uv(near, +5.485, -HALF_LENGTH)
        u_minus, _ = self._corner_uv(near, -5.485, -HALF_LENGTH)
        assert u_plus > u_minus

    def test_far_camera_shows_world_plus_x_on_image_left(self) -> None:
        projector = CameraProjector(config=_deterministic_broadcast_config())
        near, far = projector.broadcast_cameras()
        u_plus, _ = self._corner_uv(far, +5.485, +HALF_LENGTH)
        u_minus, _ = self._corner_uv(far, -5.485, +HALF_LENGTH)
        assert u_plus < u_minus

    def test_far_side_of_court_appears_above_near_side(self) -> None:
        projector = CameraProjector(config=_deterministic_broadcast_config())
        near, _ = projector.broadcast_cameras()
        _, v_far = self._corner_uv(near, 0.0, +HALF_LENGTH)
        _, v_near = self._corner_uv(near, 0.0, -HALF_LENGTH)
        assert v_far < v_near  # image v grows downward

    def test_rotation_stays_proper(self) -> None:
        cam = make_look_at_camera(center=(2.0, -30.0, 8.0), look_at=(0.0, 0.0, 0.5))
        assert torch.det(cam.R).item() == pytest.approx(1.0, abs=1e-5)


class TestBroadcastRanges:
    def _range_config(self, **overrides: object) -> _CameraConfig:
        params: dict[str, object] = dict(
            layout=BROADCAST_LAYOUT,
            broadcast_position_noise_radius=0.0,
            broadcast_look_at_xy_radius=0.0,
            broadcast_hfov_jitter_deg=0.0,
            broadcast_setback_range=(15.0, 40.0),
            broadcast_height_range=(5.0, 15.0),
            broadcast_court_width_frac_range=(0.5, 0.9),
        )
        params.update(overrides)
        return CameraConfig(**params)  # type: ignore[arg-type]

    def test_sampled_geometry_within_ranges(self) -> None:
        projector = CameraProjector(config=self._range_config())
        for _ in range(20):
            for cam in projector.broadcast_cameras():
                setback = abs(cam.C[1].item()) - HALF_LENGTH
                assert 15.0 <= setback <= 40.0
                assert 5.0 <= cam.C[2].item() <= 15.0

    def test_framing_fraction_is_realized(self) -> None:
        projector = CameraProjector(config=self._range_config())
        for _ in range(10):
            near, far = projector.broadcast_cameras()
            for cam, side in ((near, -1.0), (far, +1.0)):
                uv, in_front = project_points(
                    cam,
                    torch.tensor(
                        [
                            [-5.485, side * HALF_LENGTH, 0.0],
                            [+5.485, side * HALF_LENGTH, 0.0],
                        ]
                    ),
                )
                assert bool(in_front.all())
                frac = abs(uv[1, 0].item() - uv[0, 0].item()) / cam.w
                assert 0.5 - 1e-4 <= frac <= 0.9 + 1e-4

    def test_frac_range_conflicts_with_hfov_jitter(self) -> None:
        projector = CameraProjector(
            config=self._range_config(broadcast_hfov_jitter_deg=2.0)
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            projector.broadcast_cameras()

    def test_explicit_config_preserves_ranges(self) -> None:
        cfg = CameraConfig(
            layout="broadcast",
            broadcast_setback_range=(15.0, 40.0),
            broadcast_height_range=(5.0, 15.0),
            broadcast_court_width_frac_range=(0.5, 0.9),
        )
        assert cfg.broadcast_setback_range == (15.0, 40.0)
        assert cfg.broadcast_height_range == (5.0, 15.0)
        assert cfg.broadcast_court_width_frac_range == (0.5, 0.9)

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

    def test_explicit_fixed_layout(self) -> None:
        assert CameraConfig().layout == FIXED_LAYOUT
        assert len(CameraProjector().cameras()) == 6


class TestCameraConfigContract:
    def test_direct_constructor_requires_every_field(self) -> None:
        with pytest.raises(TypeError, match="required positional arguments"):
            _CameraConfig()  # type: ignore[call-arg]

    def test_direct_constructor_preserves_explicit_fields(self) -> None:
        cfg = CameraConfig(
            layout="broadcast",
            image_size=(640, 360),
            broadcast_height=9.0,
        )
        assert cfg.layout == "broadcast"
        assert cfg.image_size == (640, 360)
        assert cfg.broadcast_height == 9.0

"""Independent rays and post-composition resampling for a wide POV lens."""

import numpy as np
import pytest

from src.tennis_scene.visualization.player_view.head_camera import HeadCameraConfig
from src.tennis_scene.visualization.player_view.lenses import (
    EquidistantLensConfig,
    remap_linear_rgb,
    remap_srgb_bgr8,
)


def test_159_degree_lens_has_known_axis_angles_and_enclosing_pinhole() -> None:
    lens = EquidistantLensConfig(width=640, height=360)
    # Equidistant angular extents scale with the image dimensions / diagonal.
    assert lens.horizontal_fov_degrees == pytest.approx(138.580510, abs=1e-4)
    assert lens.vertical_fov_degrees == pytest.approx(77.951537, abs=1e-4)
    source = lens.pinhole_config(HeadCameraConfig())
    assert (source.width, source.height) == (2560, 1440)
    assert 155 < source.horizontal_fov_degrees < 157
    assert source.face_offset_head_m == (0, 0.08, 0.12)
    assert source.near_m == 0.05 and source.far_m == 100
    # A physical camera ray at +30 degrees goes f*theta pixels from the centre.
    theta = np.pi / 6
    uv = lens.project_directions(np.array([[np.sin(theta), 0, np.cos(theta)]]))
    np.testing.assert_allclose(uv, [[320 + lens.focal_pixels * theta, 180]], atol=1e-12)


def test_optical_axis_and_source_pixel_center_convention() -> None:
    lens = EquidistantLensConfig(width=9, height=7)
    x, y = lens.pinhole_sampling_map()
    # Odd output dimensions have a pixel centred on the optical axis. The even
    # 36x28 source centre is between indices 17/18 and 13/14, not at 18/14.
    assert x[3, 4] == pytest.approx(17.5)
    assert y[3, 4] == pytest.approx(13.5)
    source = lens.pinhole_config(HeadCameraConfig())
    fx = source.intrinsic_matrix()[0, 0]
    ray = np.stack(
        [
            (x + 0.5 - source.width / 2) / fx,
            (y + 0.5 - source.height / 2) / fx,
            np.ones_like(x),
        ],
        axis=-1,
    )
    uv = lens.project_directions(ray)
    expected_x: np.ndarray
    expected_y: np.ndarray
    expected_x, expected_y = np.meshgrid(np.arange(9) + 0.5, np.arange(7) + 0.5)
    np.testing.assert_allclose(uv[..., 0], expected_x, atol=1e-12)
    np.testing.assert_allclose(uv[..., 1], expected_y, atol=1e-12)
    assert x.min() >= 0 and x.max() <= source.width - 1
    assert y.min() >= 0 and y.max() <= source.height - 1


def test_compose_before_warping_preserves_a_known_occlusion_boundary() -> None:
    lens = EquidistantLensConfig(width=9, height=7)
    source = lens.pinhole_config(HeadCameraConfig())
    # The opaque foreground is green on the left. Red background remains on
    # the right. The optical-axis sample straddles this boundary equally.
    composed = np.zeros((source.height, source.width, 3), np.float32)
    composed[:, :18, 1] = 1
    composed[:, 18:, 0] = 1
    result = remap_linear_rgb(composed, lens)
    np.testing.assert_allclose(result[3, 4], [0.5, 0.5, 0], atol=1e-7)
    assert (result >= 0).all() and (result <= 1).all()


def test_linear_radiance_is_resampled_without_display_gamma() -> None:
    lens = EquidistantLensConfig(width=9, height=7)
    source = lens.pinhole_config(HeadCameraConfig())
    xx: np.ndarray
    yy: np.ndarray
    xx, yy = np.meshgrid(np.arange(source.width), np.arange(source.height))
    image = np.stack([xx / 40, yy / 40, np.ones_like(xx) * 0.25], axis=-1).astype(
        np.float32
    )
    result = remap_linear_rgb(image, lens)
    np.testing.assert_allclose(result[3, 4], [17.5 / 40, 13.5 / 40, 0.25], atol=1e-7)


@pytest.mark.parametrize("bad", ["backward", "zero"])
def test_unsupported_directions_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="forward"):
        EquidistantLensConfig().project_directions(
            np.array([[0.0, 0.0, -1.0 if bad == "backward" else 0.0]])
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"diagonal_fov_degrees": 180},
        {"width": 0},
        {"render_scale": 1},
        {"diagonal_fov_degrees": float("nan")},
    ],
)
def test_invalid_or_underresolved_lens_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EquidistantLensConfig(**kwargs)


def test_png_resampling_decodes_srgb_before_blending() -> None:
    lens = EquidistantLensConfig(width=9, height=7)
    source = lens.pinhole_config(HeadCameraConfig())
    image = np.zeros((source.height, source.width, 3), np.uint8)
    image[:, 18:] = 255
    # Half black and half white in linear light is sRGB188, not byte128.
    np.testing.assert_array_equal(remap_srgb_bgr8(image, lens)[3, 4], [188, 188, 188])


def test_enclosing_lens_remains_a_valid_nht_pinhole_request() -> None:
    from src.synthetic_data_generation.rendering.nht.contracts import NHTRenderCamera
    from src.synthetic_data_generation.scene_contract import RigidTransform

    source = EquidistantLensConfig().pinhole_config(HeadCameraConfig())
    camera = NHTRenderCamera(
        camera_id="wide",
        width=source.width,
        height=source.height,
        intrinsics=tuple(float(v) for v in source.intrinsic_matrix().ravel()),
        camera_to_scene=RigidTransform.from_matrix(np.eye(4)),
    )
    intrinsic = camera.to_dict()["intrinsics"]
    assert isinstance(intrinsic, dict)
    assert intrinsic["model"] == "PINHOLE"
    assert intrinsic["distortion_model"] == "NONE"


def test_remap_refuses_depth_and_nonfinite_images() -> None:
    lens = EquidistantLensConfig(width=9, height=7)
    source = lens.pinhole_config(HeadCameraConfig())
    with pytest.raises(ValueError, match="RGB"):
        remap_linear_rgb(np.ones((source.height, source.width, 1), np.float32), lens)
    with pytest.raises(ValueError, match="finite"):
        remap_linear_rgb(np.full((source.height, source.width, 3), np.nan), lens)

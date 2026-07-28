import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.plcs.avatar_asset import (
    AvatarGaussianAsset,
    build_surface_gaussian_asset,
    deform_avatar_gaussians,
)


def _tetrahedron_asset(*, count: int = 64) -> AvatarGaussianAsset:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.int64,
    )
    weights = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    return build_surface_gaussian_asset(
        vertices,
        faces=faces,
        vertex_joint_weights=weights,
        gaussian_count=count,
        seed=17,
    )


def _covariances(
    quaternions: NDArray[np.float64],
    log_scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    rotations = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*quaternions.shape[:-1], 3, 3)
    variances = np.exp(2.0 * log_scales)
    result: NDArray[np.float64] = np.einsum(
        "...ij,...j,...kj->...ik", rotations, variances, rotations
    )
    return result


def test_surface_asset_is_deterministic_metric_and_read_only() -> None:
    first = _tetrahedron_asset()
    second = _tetrahedron_asset()

    for name in (
        "means_m",
        "quaternions_wxyz",
        "log_scales_m",
        "opacity_logits",
        "point_joint_weights",
        "face_indices",
        "barycentric_coordinates",
    ):
        left = getattr(first, name)
        np.testing.assert_array_equal(left, getattr(second, name))
        assert not left.flags.writeable
    assert first.gaussian_count == 64
    np.testing.assert_allclose(
        np.linalg.norm(first.quaternions_wxyz, axis=1),
        1.0,
    )
    np.testing.assert_allclose(first.point_joint_weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(first.barycentric_coordinates.sum(axis=1), 1.0)


def test_identity_deformation_preserves_means_and_covariance() -> None:
    asset = _tetrahedron_asset()
    transforms = np.broadcast_to(np.eye(4), (1, 2, 4, 4)).copy()
    result = deform_avatar_gaussians(asset, joint_transforms=transforms)

    np.testing.assert_allclose(result.means_m[0], asset.means_m, atol=1.0e-12)
    canonical = _covariances(asset.quaternions_wxyz, asset.log_scales_m)
    deformed = _covariances(
        result.quaternions_wxyz[0],
        result.log_scales_m[0],
    )
    np.testing.assert_allclose(deformed, canonical, atol=1.0e-12)


def test_deformation_pushes_covariance_and_translation() -> None:
    asset = _tetrahedron_asset(count=16)
    transforms = np.broadcast_to(np.eye(4), (1, 2, 4, 4)).copy()
    angle = np.pi / 2.0
    transforms[0, 1, :3, :3] = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = deform_avatar_gaussians(
        asset,
        joint_transforms=transforms,
        translations_m=np.asarray([[2.0, -1.0, 0.5]]),
    )

    assert np.isfinite(result.means_m).all()
    assert np.isfinite(result.log_scales_m).all()
    np.testing.assert_allclose(
        np.linalg.norm(result.quaternions_wxyz, axis=-1),
        1.0,
    )
    assert float(result.means_m[..., 0].min()) > 1.0


def test_surface_builder_rejects_degenerate_mesh() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        build_surface_gaussian_asset(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            faces=np.asarray([[0, 1, 2]]),
            vertex_joint_weights=np.ones((3, 1)),
            gaussian_count=8,
            seed=0,
        )


def test_deformation_rejects_joint_count_mismatch() -> None:
    asset = _tetrahedron_asset()
    with pytest.raises(ValueError, match="joint counts differ"):
        deform_avatar_gaussians(
            asset,
            joint_transforms=np.broadcast_to(np.eye(4), (1, 3, 4, 4)),
        )

import numpy as np
import pytest

from src.synthetic_data_generation.dataset.plcs.components.avatar_asset import (
    AvatarGaussianAsset,
    build_surface_gaussian_asset,
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
    weights = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    return build_surface_gaussian_asset(
        vertices,
        faces=faces,
        vertex_joint_weights=weights,
        gaussian_count=count,
        seed=17,
    )


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


def test_surface_builder_rejects_degenerate_mesh() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        build_surface_gaussian_asset(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            faces=np.asarray([[0, 1, 2]]),
            vertex_joint_weights=np.ones((3, 1)),
            gaussian_count=8,
            seed=0,
        )

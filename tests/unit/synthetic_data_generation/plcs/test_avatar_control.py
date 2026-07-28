from collections.abc import Callable

import numpy as np
import pytest

from src.synthetic_data_generation.plcs.avatar_control import (
    NeighborBlend,
    apply_joint_linear_blend_skinning,
    apply_vertex_transform_blend,
    embed_points_on_posed_mesh,
    hugs_topk_neighbor_blend,
    interpolate_face_attributes,
)


def test_face_interpolation_and_mesh_embedding_preserve_attachment() -> None:
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    barycentric = np.asarray([[0.2, 0.3, 0.5]])
    attributes = np.asarray([[1.0, 0.0], [0.0, 2.0], [4.0, 4.0]])
    interpolated = interpolate_face_attributes(
        attributes,
        faces=faces,
        face_indices=np.asarray([0]),
        barycentric_coordinates=barycentric,
    )
    np.testing.assert_allclose(interpolated, [[2.2, 2.6]])

    vertices = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )
    embedded = embed_points_on_posed_mesh(
        vertices,
        faces=faces,
        face_indices=np.asarray([0]),
        barycentric_coordinates=barycentric,
    )
    np.testing.assert_allclose(embedded[:, 0], [[0.3, 0.5, 0.0], [1.3, 0.5, 0.0]])
    assert not embedded.flags.writeable


def test_joint_lbs_uses_declared_weights_and_translation() -> None:
    transforms = np.broadcast_to(np.eye(4), (1, 2, 4, 4)).copy()
    transforms[0, 1, :3, 3] = [2.0, 0.0, 0.0]
    result = apply_joint_linear_blend_skinning(
        np.asarray([[1.0, 2.0, 3.0]]),
        point_joint_weights=np.asarray([[0.25, 0.75]]),
        joint_transforms=transforms,
        translations_m=np.asarray([[0.0, 1.0, 0.0]]),
    )
    np.testing.assert_allclose(result[0, 0], [2.5, 3.0, 3.0])


def test_vertex_transform_blend_matches_explicit_neighbors() -> None:
    transforms = np.broadcast_to(np.eye(4), (2, 3, 4, 4)).copy()
    transforms[:, 1, 0, 3] = [1.0, 2.0]
    transforms[:, 2, 1, 3] = [4.0, 6.0]
    blend = NeighborBlend(
        indices=np.asarray([[1, 2]], dtype=np.int64),
        weights=np.asarray([[0.25, 0.75]]),
    )
    result = apply_vertex_transform_blend(
        np.asarray([[0.0, 0.0, 0.0]]),
        vertex_transforms=transforms,
        neighbor_blend=blend,
    )
    np.testing.assert_allclose(result[:, 0], [[0.25, 3.0, 0.0], [0.5, 4.5, 0.0]])


def test_hugs_topk_blend_is_deterministic_normalized_and_confidence_gated() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]]
    )
    joint_weights = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    first = hugs_topk_neighbor_blend(
        np.asarray([[0.011, 0.0, 0.0]]),
        template_vertices_m=vertices,
        vertex_joint_weights=joint_weights,
        k=3,
    )
    second = hugs_topk_neighbor_blend(
        np.asarray([[0.011, 0.0, 0.0]]),
        template_vertices_m=vertices,
        vertex_joint_weights=joint_weights,
        k=3,
    )
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_allclose(first.weights.sum(axis=1), 1.0)
    rejected_index = int(np.where(first.indices[0] == 2)[0][0])
    assert first.weights[0, rejected_index] == 0.0


def test_neighbor_blend_accepts_float32_smpl_rounding_without_normalizing() -> None:
    weights = np.asarray([[0.2, 0.3, 0.50000006]], dtype=np.float32)
    blend = NeighborBlend(
        indices=np.asarray([[0, 1, 2]], dtype=np.int64),
        weights=weights,
    )

    assert blend.weights.sum() != 1.0
    np.testing.assert_array_equal(blend.weights, weights.astype(np.float64))


@pytest.mark.parametrize(
    ("operation", "match"),
    [
        (
            lambda: NeighborBlend(
                indices=np.asarray([[0, 1]]),
                weights=np.asarray([[0.4, 0.4]]),
            ),
            "sum to one",
        ),
        (
            lambda: interpolate_face_attributes(
                np.ones((3, 2)),
                faces=np.asarray([[0, 1, 3]]),
                face_indices=np.asarray([0]),
                barycentric_coordinates=np.asarray([[0.2, 0.3, 0.5]]),
            ),
            "out-of-range",
        ),
        (
            lambda: apply_joint_linear_blend_skinning(
                np.zeros((1, 3)),
                point_joint_weights=np.ones((1, 1)),
                joint_transforms=np.asarray(
                    [[np.diag([-1.0, 1.0, 1.0, 1.0])]]
                ),
            ),
            "reflect or scale",
        ),
    ],
)
def test_avatar_control_rejects_ambiguous_geometry(
    operation: Callable[[], object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        operation()
